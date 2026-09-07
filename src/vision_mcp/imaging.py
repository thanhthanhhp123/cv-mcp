"""Image I/O helpers: load (path / URL / base64 data URI), resize, encode, annotate.

Everything downstream works on an RGB ``uint8`` ``H x W x 3`` numpy array.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import get_settings

_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(?P<data>.+)$", re.DOTALL)


class ImageLoadError(ValueError):
    """Raised when an image spec cannot be turned into pixels. Callers convert
    this into a clean MCP error the agent can act on."""


def _from_bytes(raw: bytes) -> np.ndarray:
    settings = get_settings()
    if len(raw) > settings.max_input_bytes:
        raise ImageLoadError(
            f"image is {len(raw)} bytes, over the {settings.max_input_bytes}-byte limit"
        )
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"could not decode image bytes: {exc}") from exc
    return np.asarray(pil.convert("RGB"))


def load_image(spec: str, *, timeout: float = 15.0) -> np.ndarray:
    """Load an image from a local path, an http(s) URL, or a base64 data URI."""

    spec = spec.strip()

    m = _DATA_URI_RE.match(spec)
    if m or _looks_like_bare_base64(spec):
        payload = m.group("data") if m else spec
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageLoadError(f"invalid base64 image data: {exc}") from exc
        return _from_bytes(raw)

    if spec.startswith(("http://", "https://")):
        import httpx

        try:
            resp = httpx.get(spec, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ImageLoadError(f"could not fetch {spec!r}: {exc}") from exc
        return _from_bytes(resp.content)

    path = Path(spec).expanduser()
    if not path.is_file():
        raise ImageLoadError(f"no such image file: {spec!r}")
    return _from_bytes(path.read_bytes())


def _looks_like_bare_base64(spec: str) -> bool:
    if len(spec) < 64 or len(spec) % 4 != 0:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", spec) is not None


def resize_max_edge(image: np.ndarray, max_edge: int | None = None) -> tuple[np.ndarray, float]:
    """Resize so the longest edge is ``max_edge``. Returns ``(resized, scale)``
    where ``scale`` maps resized coords back to the original (multiply by it)."""

    max_edge = max_edge or get_settings().max_edge_px
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image, 1.0
    ratio = max_edge / longest
    resized = cv2.resize(
        image, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA
    )
    return resized, 1.0 / ratio


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ImageLoadError("failed to PNG-encode image")
    return buf.tobytes()


def to_base64_png(image: np.ndarray) -> str:
    return base64.b64encode(encode_png(image)).decode("ascii")


_PALETTE = [
    (239, 71, 111),
    (17, 138, 178),
    (6, 214, 160),
    (255, 209, 102),
    (155, 93, 229),
]


def annotate(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    labels: list[str] | None = None,
    *,
    color_idx: int = 0,
) -> np.ndarray:
    """Draw boxes (+ optional labels) on a copy of ``image``."""

    canvas = image.copy()
    color = _PALETTE[color_idx % len(_PALETTE)]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(canvas, p1, p2, color, 2)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(
                canvas,
                labels[i],
                (p1[0], max(0, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas
