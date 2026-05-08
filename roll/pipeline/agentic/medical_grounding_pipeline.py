"""Medical Grounding Pipeline.

Extends AgenticPipeline with wandb trajectory image logging.

During evaluation (``val()``), the pipeline reads trajectory JSON files and
viewport PNG images saved by ``MedicalGroundingEnv`` and logs them as wandb
Tables so that model responses, tool calls, viewport images (annotated with
the GT bbox in red and the predicted bbox in green), and per-step rewards are
all visible in the wandb UI.

During training, trajectories are saved to disk by the env workers for offline
inspection; only scalar metrics are logged to wandb per training step.

Requires:
    - ``trajectory_log_dir`` configured under ``env_config`` in the YAML.
    - A shared filesystem accessible from both env workers and the pipeline
      process (local disk on a single-node setup, NFS on multi-node).
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
from roll.utils.logging import get_logger

logger = get_logger()

_TRAJ_TABLE_COLUMNS = [
    "step",
    "expression",
    "gt_bbox",
    "prompt",
    "tool_name",
    "tool_args",
    "viewport",
    "viewport_iou",
    "answer_iou",
    "step_reward",
    "format_error",
    "response",
    "viewport_image",
    "prediction_image",
]


class MedicalGroundingPipeline(AgenticPipeline):
    """AgenticPipeline subclass that adds wandb image logging for medical grounding."""

    def __init__(self, pipeline_config: Any) -> None:
        super().__init__(pipeline_config)
        self._trajectory_log_dir: Optional[str] = self._resolve_traj_log_dir()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_traj_log_dir(self) -> Optional[str]:
        """Pull trajectory_log_dir from custom_envs config if present."""
        for env_cfg in self.pipeline_config.custom_envs.values():
            tld = (env_cfg.get("env_config") or {}).get("trajectory_log_dir")
            if tld:
                return tld
        return None

    def _log_trajectory_files_to_wandb(
        self,
        global_step: int,
        tag: str,
        max_trajs: int = 5,
    ) -> None:
        """Read saved trajectory dirs and log them as a wandb Table.

        Args:
            global_step: Current training step (used as the wandb x-axis value).
            tag: Either ``"train"`` or ``"val"`` – determines the subdirectory
                that is scanned and the wandb table key prefix.
            max_trajs: Maximum number of trajectories to log per call.
        """
        try:
            import wandb
        except ImportError:
            logger.warning("[MedicalGroundingPipeline] wandb not installed; skipping image logging.")
            return

        if wandb.run is None:
            return

        if not self._trajectory_log_dir:
            return

        search_dir = Path(self._trajectory_log_dir)
        if not search_dir.exists():
            return

        # Each trajectory is a directory containing trajectory.json + step_NN.png files.
        traj_dirs = sorted(
            [d for d in search_dir.iterdir() if d.is_dir() and (d / "trajectory.json").exists()]
        )[:max_trajs]

        if not traj_dirs:
            return

        table = wandb.Table(columns=_TRAJ_TABLE_COLUMNS)

        for traj_dir in traj_dirs:
            with open(traj_dir / "trajectory.json") as f:
                traj: Dict = json.load(f)

            for step_info in traj.get("steps", []):
                step_idx = step_info["step"]

                img_path = traj_dir / f"step_{step_idx:02d}_orig_current.png"
                if not img_path.exists():
                    img_path = traj_dir / f"step_{step_idx:02d}.png"
                wb_image = wandb.Image(str(img_path)) if img_path.exists() else None

                pred_img_path = traj_dir / f"step_{step_idx:02d}_orig_pred.png"
                wb_pred_image = (
                    wandb.Image(str(pred_img_path)) if pred_img_path.exists() else None
                )

                response = step_info.get("response", "")
                prompt = step_info.get("prompt", "")

                table.add_data(
                    step_info.get("step", -1),
                    traj.get("expression", ""),
                    str(traj.get("gt_bbox", [])),
                    prompt,
                    step_info.get("tool_name", ""),
                    str(step_info.get("tool_args", [])),
                    str(step_info.get("viewport", step_info.get("viewport_abs", []))),
                    step_info.get("viewport_iou", 0.0),
                    step_info.get("answer_iou", 0.0),
                    step_info.get("step_reward", 0.0),
                    step_info.get("format_error", False),
                    response,
                    wb_image,
                    wb_pred_image,
                )

        self.tracker.run.log(
            {f"{tag}/trajectories": table},
            step=global_step,
        )
        logger.info(
            f"[MedicalGroundingPipeline] Logged {len(traj_dirs)} trajectory/ies "
            f"({tag}) to wandb at step {global_step}."
        )

        # Remove processed trajectory directories to avoid re-logging on the
        # next eval/logging step.
        for traj_dir in traj_dirs:
            import shutil
            shutil.rmtree(traj_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def val(self, global_step: int) -> Dict:
        """Run validation and log trajectory images to wandb."""
        metrics = super().val(global_step)
        self._log_trajectory_files_to_wandb(global_step, tag="val")
        return metrics
