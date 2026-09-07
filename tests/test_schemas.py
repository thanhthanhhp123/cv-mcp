from __future__ import annotations

import pytest
from pydantic import ValidationError

from vision_mcp.schemas import ImageInput, PlanogramInput, PlanogramSlot


def test_image_input_defaults():
    m = ImageInput(image="x.png")
    assert m.annotate is True and m.conf is None


def test_image_input_rejects_unknown_field():
    with pytest.raises(ValidationError):
        ImageInput(image="x.png", bogus=1)


def test_image_input_conf_bounds():
    with pytest.raises(ValidationError):
        ImageInput(image="x.png", conf=1.5)


def test_planogram_requires_slots():
    with pytest.raises(ValidationError):
        PlanogramInput(image="x.png", slots=[])


def test_planogram_ok():
    m = PlanogramInput(
        image="x.png",
        slots=[PlanogramSlot(label="cola", region=(0.0, 0.0, 0.2, 0.5), order=0)],
    )
    assert m.slots[0].label == "cola"
