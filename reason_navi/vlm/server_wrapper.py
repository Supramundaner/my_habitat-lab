"""
Server wrapper utilities for object detection services
独立于VLFM的服务器包装器
"""

import base64
import hashlib
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


REQUEST_ATTEMPTS = 10
LOCK_STALE_SECONDS = 120


class VLMServiceError(RuntimeError):
    """Raised when a local VLM service cannot complete a request."""


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        return jsonify(model.process_payload(payload))

    # These model services are intentionally local-only. Do not make the bind
    # address configurable without adding authentication first.
    app.run(host="127.0.0.1", port=port, use_reloader=False)


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked


def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


def send_request(url: str, **kwargs: Any) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            return _send_request(url, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < REQUEST_ATTEMPTS:
                delay = 20 + random.random() * 10
                print(
                    f"VLM request to {url} failed ({exc}); retrying in "
                    f"{delay:.1f} seconds..."
                )
                time.sleep(delay)

    raise VLMServiceError(
        f"VLM request to {url} failed after {REQUEST_ATTEMPTS} attempts"
    ) from last_error


def _lockfile_path(url: str) -> Path:
    """Return an absolute, per-user lock path outside the process cwd."""
    uid = getattr(os, "getuid", lambda: "unknown")()
    lockfiles_dir = Path(tempfile.gettempdir()) / f"reason-navi-vlm-{uid}"
    lockfiles_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    url_digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return lockfiles_dir / f"{url_digest}.lock"


@contextmanager
def _request_lock(url: str):
    filename = _lockfile_path(url)
    token = f"{os.getpid()}-{random.randint(0, 1_000_000)}"

    while True:
        try:
            fd = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                if time.time() - filename.stat().st_mtime > LOCK_STALE_SECONDS:
                    filename.unlink()
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.05)
            continue

        with os.fdopen(fd, "w") as lockfile:
            lockfile.write(token)
        break

    try:
        yield
    finally:
        try:
            if filename.read_text() == token:
                filename.unlink()
        except FileNotFoundError:
            pass


def _send_request(url: str, **kwargs: Any) -> dict:
    with _request_lock(url):
        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v

        # Set the headers
        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as exc:
            raise VLMServiceError(f"Request to {url} failed: {exc}") from exc
        except ValueError as exc:
            raise VLMServiceError(f"Service at {url} returned invalid JSON") from exc

    if not isinstance(result, dict):
        raise VLMServiceError(
            f"Service at {url} returned {type(result).__name__}; expected a JSON object"
        )
    return result
