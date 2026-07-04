import PIL.Image as Image

from roll.pipeline.agentic.env.medical_grounding.toolbox import ViewportToolBox


def test_answer_bbox_relative_to_viewport():
    image = Image.new("RGB", (200, 100), color="white")
    toolbox = ViewportToolBox(image)

    # Simulate a viewport covering the right half of the image.
    toolbox.viewport = toolbox._clamp(100.0, 0.0, 200.0, 100.0)

    # Answer bbox is specified relative to the current viewport.
    action = "<answer>bbox[0.25, 0.25, 0.75, 0.75]</answer>"
    result = toolbox.parse_and_execute(action)

    assert result["done"] is True
    assert result["tool_name"] == "answer"
    assert result["tool_success"] is True
    assert result["format_error"] is False

    # Expected original-image normalized bbox coordinates.
    # Viewport spans x=100..200, y=0..100, so x coords shift by +0.5 in normalized space.
    assert result["predicted_bbox"] == (0.625, 0.25, 0.875, 0.75)
