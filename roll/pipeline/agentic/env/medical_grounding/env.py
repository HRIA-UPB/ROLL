"""Medical Imaging Grounding Environment for RL agent training.

The agent navigates a medical image using zoom/pan tools and must
localize the region described by a referring expression.

Observation format (multimodal dict compatible with VLTrajEnvManager):
    {
        "prompt": str,          # step prompt with viewport info + <image> placeholder
        "image": [PIL.Image],   # current viewport image
    }

Action format:
    Tool calls (normalized [0, 1] to *current* viewport):
        <tool_call>zoom[x1, y1, x2, y2]</tool_call>
        <tool_call>pan[x, y]</tool_call>
        <tool_call>zoomout[factor]</tool_call>
        <tool_call>resetzoom</tool_call>
    Final answer (normalized [0, 1] to *original* image):
        <answer>bbox[x1, y1, x2, y2]</answer>

Reward structure:
    - Per-step penalty:      step_penalty (< 0, default -0.01)
    - Format violation:      format_penalty (< 0, default -0.5)
    - Tool step reward:      viewport_iou_weight * IoU(viewport, GT)
    - Final answer reward:   answer_iou_weight * IoU(prediction, GT)
"""
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PIL.Image as Image
from gem import Env

from roll.pipeline.agentic.env.medical_grounding.dataset import MedicalGroundingDataset
from roll.pipeline.agentic.env.medical_grounding.rewards import prediction_iou, viewport_iou
from roll.pipeline.agentic.env.medical_grounding.toolbox import Viewport, ViewportToolBox
from roll.utils.constants import EpisodeStopReason
from roll.utils.logging import get_logger

logger = get_logger()

# Image placeholder that VLTrajEnvManager replaces with the model's image token.
_IMAGE_PLACEHOLDER = "<image>"


_STEP_TEMPLATE = """\
Current observation:
  Referring Expression : {expression}
  Native Resolution    : {width}x{height}
  Current Viewport     : [{vx1:.4f}, {vy1:.4f}, {vx2:.4f}, {vy2:.4f}] (normalized)
  Steps remaining      : {steps_left}
  Available actions    : zoom[x1,y1,x2,y2]  pan[x,y]  zoomout[factor]  resetzoom
                         <answer>bbox[x1,y1,x2,y2]</answer>
  Note: pan and zoomout are unavailable when viewport shows full image

{image_placeholder}
"""


class MedicalGroundingEnv(Env):
    """RL environment for referring-expression grounding in medical images.

    Args:
        data_path: Path to the JSON/JSONL annotation file.
        image_dir: Directory containing images referenced by ``image_id``.
        mode: ``"train"`` (random sampling) or ``"val"`` (sequential).
        seed: Dataset RNG seed.
        max_steps: Maximum tool-call steps per episode before truncation.
        step_penalty: Negative reward applied every step.
        format_penalty: Extra negative reward when the model outputs no valid
            action or answer.
        iou_threshold: IoU above which the episode counts as a success.
        viewport_iou_weight: Reward coefficient for intermediate viewport IoU.
        answer_iou_weight: Reward coefficient for the final answer IoU.
        trajectory_log_dir: If set, trajectories are saved here as JSON + PNG
            files for offline inspection and wandb image logging.
        max_logged_trajectories: Maximum number of trajectories saved per env
            instance (prevents disk bloat during long runs).
        max_image_width: If set, resize images to this max width while preserving
            aspect ratio. If None, images are loaded at original resolution.
    """

    image_placeholder: str = _IMAGE_PLACEHOLDER

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        mode: str = "train",
        seed: Optional[int] = None,
        max_steps: int = 5,
        step_penalty: float = -0.01,
        format_penalty: float = -0.5,
        iou_threshold: float = 0.5,
        viewport_iou_weight: float = 0.3,
        answer_iou_weight: float = 1.0,
        trajectory_log_dir: Optional[str] = None,
        max_logged_trajectories: int = 20,
        max_image_width: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.format_penalty = format_penalty
        self.iou_threshold = iou_threshold
        self.viewport_iou_weight = viewport_iou_weight
        self.answer_iou_weight = answer_iou_weight
        self.trajectory_log_dir = trajectory_log_dir
        self.max_logged_trajectories = max_logged_trajectories

        dataset_mode = "sample" if mode == "train" else "traversal"
        self.dataset = MedicalGroundingDataset(
            data_path=data_path,
            image_dir=image_dir,
            mode=dataset_mode,
            seed=seed,
            max_image_width=max_image_width,
        )

        # Episode state (reset on each call to reset())
        self._data_item: Optional[Dict] = None
        self._toolbox: Optional[ViewportToolBox] = None
        self._step_count: int = 0
        self._trajectory: List[Dict] = []

        # Logging state
        self._logged_count = 0
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Start a new episode.

        Returns:
            obs: First multimodal observation dict.
            info: ``{"env_instruction": system_prompt}`` prepended to the
                first user message by the env manager.
        """
        data = self.dataset.get_item(seed if seed is not None else 0)
        self._data_item = data
        self._toolbox = ViewportToolBox(data["image"])
        self._step_count = 0
        self._trajectory = []

        obs = self._build_obs()

        # Add initial state as step 0 to trajectory
        vp_norm = self._toolbox.get_viewport_normalized()
        vp_abs = self._toolbox.get_viewport_abs()
        self._trajectory.append({
            "step": 0,
            "prompt": obs["prompt"],
            "response": None,
            "tool_name": None,
            "tool_args": [],
            "tool_success": False,
            "format_error": False,
            "error_msg": None,
            "viewport_norm": list(vp_norm),
            "viewport_abs": list(vp_abs),
            "viewport": list(vp_abs),
            "viewport_iou": 0.0,
            "answer_iou": 0.0,
            "step_reward": 0.0,
            "predicted_bbox": None,
            "predicted_viewport": None,
            "gt_bbox": list(self._data_item["gt_bbox"]),
            "success": False,
        })

        # env_instruction is prepended to the first user message by VLTrajEnvManager.
        # We leave it empty and rely on agent_system_template in the YAML config
        # to inject the system prompt into the LLM's system role instead.
        return obs, {}

    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute one agent action and return the next state.

        Args:
            action: Raw model output string.

        Returns:
            Standard gym (obs, reward, terminated, truncated, info) tuple.
            obs is ``""`` (empty string) when the episode is done.
        """
        self._step_count += 1

        if isinstance(action, EpisodeStopReason) and action == EpisodeStopReason.MAX_LENGTH:
            logger.info(f"[MAX_LENGTH] Episode terminated due to MAX_LENGTH, step_count={self._step_count}")
            done = True
            truncated = True
            reward = 0.0
            info = {}
            return "", reward, done, truncated, info

        # Capture the prompt that was shown to the agent before it responded
        current_obs = self._build_obs()
        current_prompt = current_obs["prompt"]

        result = self._toolbox.parse_and_execute(action)
        tool_name: Optional[str] = result["tool_name"]
        tool_success: bool = result["tool_success"]
        format_error: bool = result["format_error"]
        done: bool = result["done"]
        predicted_bbox: Optional[Tuple] = result["predicted_bbox"]

        data = self._data_item
        gt_bbox: Tuple = data["gt_bbox"]
        img_w: int = data["width"]
        img_h: int = data["height"]

        # Compute rewards
        vp_abs = self._toolbox.get_viewport_abs()
        vp_iou = viewport_iou(vp_abs, gt_bbox, img_w, img_h)
        ans_iou = 0.0

        step_reward = self.step_penalty
        if format_error:
            step_reward += self.format_penalty
        if done and predicted_bbox is not None:
            ans_iou = prediction_iou(predicted_bbox, gt_bbox, img_w, img_h)
            step_reward += self.answer_iou_weight * ans_iou
        elif tool_success:
            step_reward += self.viewport_iou_weight * vp_iou

        success = ans_iou >= self.iou_threshold if done else False

        # Track for logging
        vp_norm = self._toolbox.get_viewport_normalized()
        self._trajectory.append({
            "step": self._step_count,
            "prompt": current_prompt,
            "response": action,
            "tool_name": tool_name,
            "tool_args": result.get("tool_args", []),
            "tool_success": tool_success,
            "format_error": format_error,
            "error_msg": result.get("error_msg"),
            "viewport_norm": list(vp_norm),
            "viewport_abs": list(vp_abs),
            "viewport": list(vp_abs),
            "viewport_iou": round(vp_iou, 4),
            "answer_iou": round(ans_iou, 4),
            "step_reward": round(step_reward, 4),
            "predicted_bbox": list(predicted_bbox) if predicted_bbox else None,
            "predicted_viewport": list(predicted_bbox) if predicted_bbox else None,
            "gt_bbox": list(gt_bbox),
            "success": success,
        })

        # Truncate at max_steps if the agent hasn't answered yet
        truncated = False
        if self._step_count >= self.max_steps and not done:
            done = True
            truncated = True

        if done:
            self._maybe_save_trajectory()

        next_obs = "" if done else self._build_obs()

        metrics = {
            "tool_success": float(tool_success),
            "format_error": float(format_error),
            "viewport_iou": vp_iou,
            "answer_iou": ans_iou,
            "success": float(success),
        }
        metrics_agg_mode = {
            "tool_success": "mean",
            "format_error": "mean",
            "viewport_iou": "mean",
            "answer_iou": "last",
            "success": "last",
        }
        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "action_desc": (
                f"step={self._step_count} tool={tool_name or 'invalid'} "
                f"vp_iou={vp_iou:.3f} ans_iou={ans_iou:.3f} r={step_reward:.3f}"
            ),
        }
        return next_obs, step_reward, done, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> Dict:
        """Construct the current multimodal observation."""
        data = self._data_item
        vx1, vy1, vx2, vy2 = self._toolbox.get_viewport_normalized()
        prompt = _STEP_TEMPLATE.format(
            expression=data["expression"],
            width=data["width"],
            height=data["height"],
            vx1=vx1,
            vy1=vy1,
            vx2=vx2,
            vy2=vy2,
            steps_left=self.max_steps - self._step_count,
            image_placeholder=_IMAGE_PLACEHOLDER,
        )
        return {
            "prompt": prompt,
            "image": [self._toolbox.get_viewport_image()],
        }

    def _maybe_save_trajectory(self) -> None:
        """Save this episode's trajectory to disk if logging is configured."""
        if not self.trajectory_log_dir:
            return
        with self._log_lock:
            if self._logged_count >= self.max_logged_trajectories:
                return
            self._logged_count += 1

        traj_id = str(uuid.uuid4())[:8]
        img_id = self._data_item["image_id"].replace("/", "_").replace(".", "_")
        traj_name = f"{img_id}_{self._data_item['bbox_id']}_{traj_id}"
        out_dir = Path(self.trajectory_log_dir) / traj_name
        out_dir.mkdir(parents=True, exist_ok=True)

        gt_bbox = self._data_item["gt_bbox"]

        # Save annotated viewport image per step
        original_toolbox = ViewportToolBox(self._data_item["image"])
        for step_info in self._trajectory:
            vabs = step_info["viewport_abs"]
            original_toolbox.viewport = Viewport(*vabs)
            pred_norm = (
                tuple(step_info["predicted_bbox"]) if step_info["predicted_bbox"] else None
            )
            annotated = original_toolbox.annotate_viewport(
                gt_bbox=gt_bbox,
                pred_bbox_norm=pred_norm,
            )
            annotated.save(out_dir / f"step_{step_info['step']:02d}.png")

            current_overlay = original_toolbox.annotate_original(
                viewport_abs=tuple(vabs),
                gt_bbox=gt_bbox,
                pred_bbox_norm=None,
                draw_viewport=True,
                draw_pred=False,
            )
            current_overlay.save(out_dir / f"step_{step_info['step']:02d}_orig_current.png")

            if pred_norm is not None:
                pred_overlay = original_toolbox.annotate_original(
                    viewport_abs=tuple(vabs),
                    gt_bbox=gt_bbox,
                    pred_bbox_norm=pred_norm,
                    draw_viewport=False,
                    draw_pred=True,
                )
                pred_overlay.save(out_dir / f"step_{step_info['step']:02d}_orig_pred.png")

        # Save trajectory JSON
        traj_data = {
            "image_id": self._data_item["image_id"],
            "bbox_id": self._data_item["bbox_id"],
            "expression": self._data_item["expression"],
            "gt_bbox": list(gt_bbox),
            "native_resolution": f"{self._data_item['width']}x{self._data_item['height']}",
            "steps": self._trajectory,
            "episode_reward": sum(s["step_reward"] for s in self._trajectory),
            "final_answer_iou": self._trajectory[-1]["answer_iou"] if self._trajectory else 0.0,
            "success": self._trajectory[-1]["success"] if self._trajectory else False,
        }
        with open(out_dir / "trajectory.json", "w") as f:
            json.dump(traj_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"[MedicalGroundingEnv] Saved trajectory to {out_dir}")
