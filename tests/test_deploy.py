"""Exercise deployment state transitions with fake Docker/curl; no live host touched."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/deploy.sh"
IMAGE = "ghcr.io/16minsun-afk/yolo-api@sha256:" + "a" * 64
pytestmark = pytest.mark.skipif(not shutil.which("bash") or not shutil.which("flock"), reason="Linux shell tools required")


@pytest.fixture
def host(tmp_path):
    app = tmp_path / "yolo-api"
    model = app / "models/new"
    model.mkdir(parents=True)
    (model / "best.pt").write_bytes(b"dummy")
    (model / "manifest.json").write_text("{}")
    (app / ".env").write_text("API_TOKEN=not-a-real-key")
    (app / "compose.yaml").write_text("services: {}")
    (app / "deployment.env").write_text(f"IMAGE_REF={IMAGE}\nMODEL_VERSION=old\n")
    bins = tmp_path / "bin"
    bins.mkdir()
    docker = bins / "docker"
    docker.write_text(r"""#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
files = [args[i + 1] for i, value in enumerate(args) if value == '--env-file']
state = Path(files[-1]).read_text() if files else ''
with open(os.environ['LOG'], 'a') as log:
    log.write(' '.join(args) + ' MODEL=' + state.split('MODEL_VERSION=')[-1].strip() + '\n')
if os.environ.get('FAIL_WARMUP') == '1' and 'run' in args:
    sys.exit(1)
if os.environ.get('FAIL_NEW') == '1' and 'up' in args and 'MODEL_VERSION=new' in state:
    sys.exit(1)
""")
    docker.chmod(0o755)
    curl = bins / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n")
    curl.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path), "PATH": str(bins) + os.pathsep + os.environ["PATH"],
           "LOG": str(tmp_path / "docker.log")}
    return app, env


def run(host, **extra):
    app, env = host
    return subprocess.run(["bash", str(SCRIPT), IMAGE, "new"], env={**env, **extra}, capture_output=True, text=True)


def test_success_changes_image_and_model(host):
    result = run(host)
    assert result.returncode == 0, result.stderr
    assert "MODEL_VERSION=new" in (host[0] / "deployment.env").read_text()
    assert "MODEL_VERSION=old" in (host[0] / "deployment.env.previous").read_text()


def test_failed_new_service_rolls_back(host):
    result = run(host, FAIL_NEW="1")
    assert result.returncode != 0
    assert "MODEL_VERSION=old" in (host[0] / "deployment.env").read_text()
    assert "rollback" in result.stdout.lower()


def test_failed_preflight_preserves_running_release(host):
    result = run(host, FAIL_WARMUP="1")
    assert result.returncode != 0
    assert "MODEL_VERSION=old" in (host[0] / "deployment.env").read_text()


def test_invalid_image_is_rejected_before_docker(host):
    result = subprocess.run(["bash", str(SCRIPT), "untrusted:latest", "new"], env=host[1], capture_output=True)
    assert result.returncode == 2
    assert not Path(host[1]["LOG"]).exists()
