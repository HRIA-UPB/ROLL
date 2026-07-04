"""Viewport navigation toolbox for medical image grounding."""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import PIL.Image as Image
from PIL import ImageDraw


@dataclass
class Viewport:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def normalized(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        return self.x1 / img_w, self.y1 / img_h, self.x2 / img_w, self.y2 / img_h

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop((int(self.x1), int(self.y1), int(self.x2), int(self.y2)))


class ViewportToolBox:
    """Tracks viewport state and executes navigation tools.

    Coordinates are in absolute pixels internally. All tool inputs are
    normalized [0, 1] relative to the *current* viewport.
    """

    _ANSWER_RE = re.compile(
        r"<answer>bbox\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]</answer>"
    )
    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _ZOOM_RE = re.compile(r"zoom\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]")
    _PAN_RE = re.compile(r"pan\[([\d.]+),\s*([\d.]+)\]")
    _ZOOMOUT_RE = re.compile(r"zoomout\[([\d.]+)\]")
    _RESETZOOM_RE = re.compile(r"resetzoom")

    def __init__(self, image: Image.Image, max_aspect_ratio: float = 200.0) -> None:
        self.original_image = image
        self.max_aspect_ratio = max_aspect_ratio
        self._reset_viewport()

    def reset(self, image: Image.Image) -> None:
        self.original_image = image
        self._reset_viewport()

    def _reset_viewport(self) -> None:
        w, h = self.original_image.size
        self.viewport = Viewport(0.0, 0.0, float(w), float(h))

    @property
    def img_w(self) -> int:
        return self.original_image.width

    @property
    def img_h(self) -> int:
        return self.original_image.height

    def get_viewport_image(self) -> Image.Image:
        return self.viewport.crop(self.original_image)

    def get_viewport_normalized(self) -> Tuple[float, float, float, float]:
        return self.viewport.normalized(self.img_w, self.img_h)

    def get_viewport_abs(self) -> Tuple[float, float, float, float]:
        return self.viewport.x1, self.viewport.y1, self.viewport.x2, self.viewport.y2

    def is_at_full_size(self) -> bool:
        """Check if the viewport currently shows the full original image."""
        vp = self.viewport
        return (
            vp.x1 == 0.0
            and vp.y1 == 0.0
            and vp.x2 == float(self.img_w)
            and vp.y2 == float(self.img_h)
        )

    def _check_aspect_ratio(self, width: float, height: float) -> bool:
        """Return True if width/height ratio is within the allowed limit."""
        if min(width, height) <= 0:
            return False
        return max(width, height) / min(width, height) <= self.max_aspect_ratio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _norm_to_abs(
        self, nx1: float, ny1: float, nx2: float, ny2: float
    ) -> Tuple[float, float, float, float]:
        """Convert normalized viewport coords to absolute image coords."""
        vp = self.viewport
        return (
            vp.x1 + nx1 * vp.width,
            vp.y1 + ny1 * vp.height,
            vp.x1 + nx2 * vp.width,
            vp.y1 + ny2 * vp.height,
        )

    def _clamp(self, x1: float, y1: float, x2: float, y2: float) -> Viewport:
        """Clamp a viewport to image bounds and enforce minimum 10px size."""
        w, h = float(self.img_w), float(self.img_h)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10:
            x2 = min(x1 + 10, w)
        if y2 - y1 < 10:
            y2 = min(y1 + 10, h)
        return Viewport(x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Tool executors
    # ------------------------------------------------------------------

    def _do_zoom(self, nx1: float, ny1: float, nx2: float, ny2: float) -> Tuple[bool, Optional[str]]:
        if not (0.0 <= nx1 < nx2 <= 1.0 and 0.0 <= ny1 < ny2 <= 1.0):
            return False, "Invalid normalized zoom coordinates"
        viewport = self._clamp(*self._norm_to_abs(nx1, ny1, nx2, ny2))
        if not self._check_aspect_ratio(viewport.width, viewport.height):
            return False, (
                f"Aspect ratio too large for zoom viewport: "
                f"width={viewport.width:.4f}, height={viewport.height:.4f}, "
                f"max={self.max_aspect_ratio}"
            )
        self.viewport = viewport
        return True, None

    def _do_pan(self, nx: float, ny: float) -> Tuple[bool, Optional[str]]:
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            return False, "Invalid normalized pan coordinates"
        if self.is_at_full_size():
            return False, "Cannot pan when viewport already shows the full image"
        vp = self.viewport
        cx = vp.x1 + nx * vp.width
        cy = vp.y1 + ny * vp.height
        self.viewport = self._clamp(cx - vp.width / 2, cy - vp.height / 2, cx + vp.width / 2, cy + vp.height / 2)
        return True, None

    def _do_zoomout(self, factor: float) -> Tuple[bool, Optional[str]]:
        if factor <= 1.0:
            return False, "zoomout factor must be > 1.0"
        if self.is_at_full_size():
            return False, "Cannot zoom out when viewport already shows the full image"
        vp = self.viewport
        cx, cy = (vp.x1 + vp.x2) / 2, (vp.y1 + vp.y2) / 2
        viewport = self._clamp(cx - vp.width * factor / 2, cy - vp.height * factor / 2,
                               cx + vp.width * factor / 2, cy + vp.height * factor / 2)
        if not self._check_aspect_ratio(viewport.width, viewport.height):
            return False, (
                f"Aspect ratio too large for zoomout viewport: "
                f"width={viewport.width:.4f}, height={viewport.height:.4f}, "
                f"max={self.max_aspect_ratio}"
            )
        self.viewport = viewport
        return True, None

    def _do_resetzoom(self) -> Tuple[bool, Optional[str]]:
        self._reset_viewport()
        return True, None

    # ------------------------------------------------------------------
    # Annotation helper
    # ------------------------------------------------------------------

    def answer_bbox_viewport_to_original_norm(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """Convert a viewport-relative answer bbox to original image normalized coords."""
        vp = self.viewport
        ax1 = vp.x1 + bbox[0] * vp.width
        ay1 = vp.y1 + bbox[1] * vp.height
        ax2 = vp.x1 + bbox[2] * vp.width
        ay2 = vp.y1 + bbox[3] * vp.height
        return ax1 / self.img_w, ay1 / self.img_h, ax2 / self.img_w, ay2 / self.img_h

    def annotate_viewport(
        self,
        gt_bbox: Optional[Tuple[float, float, float, float]] = None,
        pred_bbox_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> Image.Image:
        """Return the current viewport image with optional annotations.

        Args:
            gt_bbox: Ground-truth bbox in absolute pixel coords (drawn red).
            pred_bbox_norm: Predicted bbox normalized [0, 1] to the original
                image (drawn green). Only for the final answer step.
        """
        img = self.get_viewport_image().copy()
        draw = ImageDraw.Draw(img)
        vp = self.viewport

        def _abs_to_view(ax1: float, ay1: float, ax2: float, ay2: float):
            rx1 = (ax1 - vp.x1) / vp.width * img.width
            ry1 = (ay1 - vp.y1) / vp.height * img.height
            rx2 = (ax2 - vp.x1) / vp.width * img.width
            ry2 = (ay2 - vp.y1) / vp.height * img.height
            return rx1, ry1, rx2, ry2

        if gt_bbox is not None:
            rx1, ry1, rx2, ry2 = _abs_to_view(*gt_bbox)
            if rx2 > 0 and ry2 > 0 and rx1 < img.width and ry1 < img.height:
                draw.rectangle([rx1, ry1, rx2, ry2], outline="green", width=3)

        if pred_bbox_norm is not None:
            ax1 = pred_bbox_norm[0] * self.img_w
            ay1 = pred_bbox_norm[1] * self.img_h
            ax2 = pred_bbox_norm[2] * self.img_w
            ay2 = pred_bbox_norm[3] * self.img_h
            rx1, ry1, rx2, ry2 = _abs_to_view(ax1, ay1, ax2, ay2)
            if rx2 > 0 and ry2 > 0 and rx1 < img.width and ry1 < img.height:
                draw.rectangle([rx1, ry1, rx2, ry2], outline="red", width=3)

        return img

    def annotate_original(
        self,
        viewport_abs: Tuple[float, float, float, float],
        gt_bbox: Optional[Tuple[float, float, float, float]] = None,
        pred_bbox_norm: Optional[Tuple[float, float, float, float]] = None,
        draw_viewport: bool = True,
        draw_pred: bool = True,
    ) -> Image.Image:
        img = self.original_image.copy()
        draw = ImageDraw.Draw(img)

        if draw_viewport:
            vx1, vy1, vx2, vy2 = viewport_abs
            draw.rectangle([vx1, vy1, vx2, vy2], outline="blue", width=3)

        if gt_bbox is not None:
            gx1, gy1, gx2, gy2 = gt_bbox
            draw.rectangle([gx1, gy1, gx2, gy2], outline="green", width=4)

        if draw_pred and pred_bbox_norm is not None:
            px1 = pred_bbox_norm[0] * self.img_w
            py1 = pred_bbox_norm[1] * self.img_h
            px2 = pred_bbox_norm[2] * self.img_w
            py2 = pred_bbox_norm[3] * self.img_h
            draw.rectangle([px1, py1, px2, py2], outline="red", width=3)

        return img

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def parse_and_execute(self, action: str) -> Dict[str, Any]:
        """Parse an action string, execute the matching tool, and return results.

        Returns a dict with:
            done (bool): True if the episode should end (answer given).
            predicted_bbox (Optional[tuple]): Normalized [0,1] answer bbox.
            tool_name (Optional[str]): Which tool was called.
            tool_args (List[float]): Arguments passed to the tool.
            tool_success (bool): Whether the tool executed successfully.
            format_error (bool): True if no valid action was found.
            viewport_image (PIL.Image): Current viewport after execution.
            error_msg (Optional[str]): Human-readable error if applicable.
        """
        # Check for final answer
        answer_m = self._ANSWER_RE.search(action)
        if answer_m:
            try:
                bbox = tuple(float(v) for v in answer_m.groups())
            except ValueError:
                return {
                    "done": False,
                    "predicted_bbox": None,
                    "tool_name": None,
                    "tool_args": [],
                    "tool_success": False,
                    "format_error": True,
                    "viewport_image": self.get_viewport_image(),
                    "error_msg": f"Invalid float values in answer bbox: {answer_m.groups()}",
                }
            if not (0.0 <= bbox[0] < bbox[2] <= 1.0 and 0.0 <= bbox[1] < bbox[3] <= 1.0):
                return {
                    "done": False,
                    "predicted_bbox": None,
                    "tool_name": "answer",
                    "tool_args": list(bbox),
                    "tool_success": False,
                    "format_error": True,
                    "viewport_image": self.get_viewport_image(),
                    "error_msg": (
                        "Answer bbox must satisfy 0.0 <= x1 < x2 <= 1.0 and "
                        "0.0 <= y1 < y2 <= 1.0"
                    ),
                }
            bbox_norm = self.answer_bbox_viewport_to_original_norm(bbox)
            width = bbox_norm[2] - bbox_norm[0]
            height = bbox_norm[3] - bbox_norm[1]
            if not self._check_aspect_ratio(width, height):
                return {
                    "done": False,
                    "predicted_bbox": None,
                    "tool_name": "answer",
                    "tool_args": list(bbox),
                    "tool_success": False,
                    "format_error": True,
                    "viewport_image": self.get_viewport_image(),
                    "error_msg": (
                        f"Aspect ratio too large for answer bbox: width={width:.4f}, "
                        f"height={height:.4f}, max={self.max_aspect_ratio}"
                    ),
                }
            return {
                "done": True,
                "predicted_bbox": bbox_norm,
                "tool_name": "answer",
                "tool_args": list(bbox),
                "tool_success": True,
                "format_error": False,
                "viewport_image": self.get_viewport_image(),
                "error_msg": None,
            }

        # Check for a tool call block
        call_m = self._TOOL_CALL_RE.search(action)
        if not call_m:
            return {
                "done": False,
                "predicted_bbox": None,
                "tool_name": None,
                "tool_args": [],
                "tool_success": False,
                "format_error": True,
                "viewport_image": self.get_viewport_image(),
                "error_msg": "No <tool_call>...</tool_call> or <answer>bbox[...]</answer> found.",
            }

        content = call_m.group(1).strip()
        tool_handlers = [
            ("zoom", self._ZOOM_RE, lambda m: self._do_zoom(*[float(v) for v in m.groups()])),
            ("pan", self._PAN_RE, lambda m: self._do_pan(*[float(v) for v in m.groups()])),
            ("zoomout", self._ZOOMOUT_RE, lambda m: self._do_zoomout(float(m.group(1)))),
            ("resetzoom", self._RESETZOOM_RE, lambda _: self._do_resetzoom()),
        ]

        for tool_name, pattern, handler in tool_handlers:
            m = pattern.search(content)
            if m:
                try:
                    args: List[float] = [float(v) for v in m.groups()] if m.groups() else []
                except ValueError:
                    return {
                        "done": False,
                        "predicted_bbox": None,
                        "tool_name": tool_name,
                        "tool_args": [],
                        "tool_success": False,
                        "format_error": True,
                        "viewport_image": self.get_viewport_image(),
                        "error_msg": f"Invalid float values for {tool_name}: {m.groups()}",
                    }
                try:
                    result = handler(m)
                except ValueError as err:
                    return {
                        "done": False,
                        "predicted_bbox": None,
                        "tool_name": tool_name,
                        "tool_args": args,
                        "tool_success": False,
                        "format_error": True,
                        "viewport_image": self.get_viewport_image(),
                        "error_msg": str(err),
                    }
                success, error_msg = result if isinstance(result, tuple) else (bool(result), None)
                return {
                    "done": False,
                    "predicted_bbox": None,
                    "tool_name": tool_name,
                    "tool_args": args,
                    "tool_success": bool(success),
                    "format_error": not bool(success),
                    "viewport_image": self.get_viewport_image(),
                    "error_msg": None if success else (error_msg or f"Invalid args for {tool_name}: {args}"),
                }

        return {
            "done": False,
            "predicted_bbox": None,
            "tool_name": None,
            "tool_args": [],
            "tool_success": False,
            "format_error": True,
            "viewport_image": self.get_viewport_image(),
            "error_msg": f"Unrecognized tool command: {content}",
        }
