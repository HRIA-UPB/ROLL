"""Reward functions for medical image grounding."""
import math
from typing import Tuple


def compute_iou(
    pred: Tuple[float, float, float, float],
    gt: Tuple[float, float, float, float],
    img_w: int = None,
    img_h: int = None,
) -> float:
    """Compute Intersection-over-Union between two bboxes in [x1, y1, x2, y2] format.
    
    When intersection is 0, returns a distance-based score if image dimensions are provided.
    """
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt

    inter_x1, inter_y1 = max(px1, gx1), max(py1, gy1)
    inter_x2, inter_y2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter

    if inter > 0.0:
        return inter / union
    
    # No intersection: use distance-based score if image dimensions provided
    if img_w is not None and img_h is not None:
        # Compute center points
        pred_cx = (px1 + px2) / 2.0
        pred_cy = (py1 + py2) / 2.0
        gt_cx = (gx1 + gx2) / 2.0
        gt_cy = (gy1 + gy2) / 2.0
        
        # Euclidean distance between centers
        distance = math.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)
        
        # Normalize by image diagonal
        diagonal = math.sqrt(img_w ** 2 + img_h ** 2)
        normalized_distance = distance / diagonal
        
        # Convert to reward: exp(-k * distance) where k controls decay
        # Using k=5 gives reasonable decay: distance=0.1*diag -> ~0.61, distance=0.5*diag -> ~0.08
        return math.exp(-5.0 * normalized_distance)
    
    return 0.0


def viewport_iou(
    viewport_abs: Tuple[float, float, float, float],
    gt_bbox_abs: Tuple[float, float, float, float],
    img_w: int = None,
    img_h: int = None,
) -> float:
    """IoU between current viewport and GT bbox, both in absolute pixel coords."""
    return compute_iou(viewport_abs, gt_bbox_abs, img_w, img_h)


def prediction_iou(
    predicted_bbox_norm: Tuple[float, float, float, float],
    gt_bbox_abs: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> float:
    """IoU between predicted bbox (normalized [0,1] to original image) and GT bbox (absolute px)."""
    px1 = predicted_bbox_norm[0] * img_w
    py1 = predicted_bbox_norm[1] * img_h
    px2 = predicted_bbox_norm[2] * img_w
    py2 = predicted_bbox_norm[3] * img_h
    return compute_iou((px1, py1, px2, py2), gt_bbox_abs, img_w, img_h)
