#!/usr/bin/env bash
# Executed as the deploy user; no root login, no Docker socket exposed remotely.
set -Eeuo pipefail
umask 077
IMAGE_REF=${1:?Provide immutable image reference}
MODEL_VERSION=${2:?Provide a model directory name}
[[ "$IMAGE_REF" =~ ^ghcr\.io/16minsun-afk/yolo-api@sha256:[a-f0-9]{64}$ ]] || exit 2
[[ "$MODEL_VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || exit 2
cd "$HOME/yolo-api"
exec 9>.deploy.lock
flock -n 9 || { echo 'Another deployment is running'; exit 1; }
[[ -f .env && -f compose.yaml && -f "models/$MODEL_VERSION/best.pt" && -f "models/$MODEL_VERSION/manifest.json" ]] || {
  echo 'Missing server setup or model bundle. Follow docs/SETUP_KO.md.'; exit 1;
}
printf 'IMAGE_REF=%s\nMODEL_VERSION=%s\n' "$IMAGE_REF" "$MODEL_VERSION" > deployment.env.next
compose() { docker compose --env-file .env --env-file "$1" "${@:2}"; }
# Validate config without printing interpolated secrets.
compose deployment.env.next config --quiet
compose deployment.env.next pull
# Verify the mounted model and actual warm-up using the NEW image, before stopping the old service.
compose deployment.env.next run --rm --no-deps api python -c 'from server.app.main import load_predictor; load_predictor(); print("Model warm-up OK")'
if [[ -f deployment.env ]]; then cp deployment.env deployment.env.previous; else rm -f deployment.env.previous; fi
rollback() {
  local status=$?
  trap - ERR
  echo 'Deployment failed; attempting previous image/model rollback.'
  if [[ -f deployment.env.previous ]]; then
    cp deployment.env.previous deployment.env
    compose deployment.env up -d --wait --wait-timeout 180 || echo 'ROLLBACK FAILED: inspect docker compose logs'
  else
    compose deployment.env stop api || true
    echo 'First deployment failed; no previous release exists.'
  fi
  exit "$status"
}
trap rollback ERR
mv deployment.env.next deployment.env
compose deployment.env up -d --wait --wait-timeout 180
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8000/readyz
trap - ERR
echo "Deployed model $MODEL_VERSION using $IMAGE_REF"
