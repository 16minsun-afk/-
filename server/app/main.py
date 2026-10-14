"""CPU inference API. The static key is for a private pilot, not public-app login."""
import io
import logging
import os
import secrets
import threading
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from pipeline.artifacts import verify_bundle

MAX_FILE = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
log = logging.getLogger(__name__)


class BodyLimitMiddleware:
    """Bound the complete request BEFORE multipart parsing, including chunked bodies."""
    def __init__(self, app, max_bytes=MAX_FILE + 65536):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        if length < 0 or length > self.max_bytes:
            return await JSONResponse({"detail": "Request too large"}, 413)(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                return await JSONResponse({"detail": "Request too large"}, 413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def load_predictor():
    directory = Path(os.environ.get("MODEL_DIR", "/models"))
    minimum = float(os.environ.get("MODEL_MIN_MAP", "0.5"))
    manifest = verify_bundle(directory, minimum, allow_smoke=os.environ.get("ALLOW_SMOKE_MODEL") == "1")
    from ultralytics import YOLO

    model = YOLO(str(directory / "best.pt"))
    size = int(manifest.get("imgsz", 640))
    if size < 32 or size > 2048:
        raise ValueError("Model inference image size must be between 32 and 2048")
    model.predict(Image.new("RGB", (64, 64)), device="cpu", imgsz=size, verbose=False)

    def predict(image):
        result = model.predict(image, device="cpu", imgsz=size, verbose=False)[0]
        return [
            {"class_id": int(box.cls.item()), "class_name": result.names[int(box.cls.item())],
             "confidence": float(box.conf.item()), "xyxy": box.xyxy[0].tolist()}
            for box in result.boxes
        ]

    return predict


def create_app(predictor_factory=None):
    factory = predictor_factory or load_predictor
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        token = os.environ.get("API_TOKEN", "")
        if len(token) < 32:
            raise RuntimeError("Set a random API_TOKEN of at least 32 characters")
        app.state.token = token
        app.state.predictor = await run_in_threadpool(factory)
        yield
        app.state.predictor = None

    app = FastAPI(title="YOLO CPU API", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(BodyLimitMiddleware)

    def authenticate(x_api_key: str | None = Header(default=None)):
        if not x_api_key or not secrets.compare_digest(x_api_key.encode(), app.state.token.encode()):
            raise HTTPException(401, "Invalid API key")

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/readyz")
    def ready():
        if not getattr(app.state, "predictor", None):
            raise HTTPException(503, "Model not ready")
        return {"status": "ready"}

    def infer(content):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as source:
                    if source.format not in {"JPEG", "PNG", "WEBP"}:
                        raise HTTPException(415, "Unsupported image format")
                    if source.width * source.height > MAX_PIXELS:
                        raise HTTPException(413, "Too many image pixels")
                    image = ImageOps.exif_transpose(source).convert("RGB")
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(413, "Too many image pixels") from None
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(400, "Invalid image") from None
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "Inference busy; retry later", headers={"Retry-After": "1"})
        try:
            return {"detections": app.state.predictor(image)}
        except Exception:
            log.exception("Inference failed")
            raise HTTPException(500, "Inference failed") from None
        finally:
            lock.release()

    @app.post("/predict", dependencies=[Depends(authenticate)])
    async def predict(file: UploadFile = File(...)):
        try:
            if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(415, "Unsupported content type")
            content = await file.read(MAX_FILE + 1)
            if len(content) > MAX_FILE:
                raise HTTPException(413, "Image too large")
            return await run_in_threadpool(infer, content)
        finally:
            await file.close()

    return app


app = create_app()
