"""``read_price_tags`` — OCR price labels and link each to the nearest product.

The OCR backend is a stub; this tool wires it up and returns a clear
``not_implemented`` status (not an error) so agents and ``shelf_report`` can
handle it gracefully.
"""

from __future__ import annotations

from ..backends.ocr import get_ocr
from ..imaging import ImageLoadError, load_image
from ..schemas import ImageInput, ImageMeta, ReadPriceTagsOutput
from ..storage import get_storage
from ._common import ToolInputError


def read_price_tags(params: ImageInput) -> ReadPriceTagsOutput:
    try:
        image = load_image(params.image)
    except ImageLoadError as exc:
        raise ToolInputError(str(exc)) from exc

    h, w = image.shape[:2]
    meta = ImageMeta(width=w, height=h, image_id=get_storage().put_image(image))

    result = get_ocr().run(image, {})
    if result.status == "not_implemented":
        return ReadPriceTagsOutput(
            status="not_implemented",
            image=meta,
            notes=[result.message or "OCR backend not implemented."],
        )
    if result.status == "error":
        raise ToolInputError(result.message or "OCR backend failed")

    # Real path (once the backend lands): map result.detections -> PriceTag.
    return ReadPriceTagsOutput(status="ok", tags=[], image=meta)
