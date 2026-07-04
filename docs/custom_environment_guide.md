# Creating Custom ROLL Reinforcement Learning Environments

This guide explains how to create a custom reinforcement learning environment for the ROLL (Reinforcement Learning with Large Language Models) framework, using the medical grounding environment as a reference implementation.

## Overview

ROLL provides a framework for training vision-language models (VLMs) through RL on agentic tasks. The framework consists of several core components that work together:

- **Environment**: Implements the RL environment (Gym API) with custom state, action, and reward logic
- **Dataset**: Loads and manages training/evaluation data
- **Toolbox**: Parses and executes agent tool calls (if using tool-based interactions)
- **Rewards**: Computes reward signals for the agent
- **Pipeline**: Orchestrates training, rollout collection, and model updates
- **EnvManager**: Manages environment instances and handles multimodal observations

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     AgenticPipeline                           │
│  (orchestrates training, rollouts, model updates)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  VLTrajEnvManager                            │
│  (manages env instances, handles multimodal observations)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Custom Environment (gem.Env)                    │
│  - reset(): Initialize episode                               │
│  - step(action): Execute action, return (obs, reward, ...)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │ Dataset │   │ Toolbox  │   │ Rewards  │
   └─────────┘   └──────────┘   └──────────┘
```

## Step 1: Create the Dataset Class

The dataset class loads your training/evaluation data and provides reproducible sampling.

**File**: `roll/pipeline/agentic/env/{your_env}/dataset.py`

```python
"""Dataset loading for your custom environment."""
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import PIL.Image as Image


def _load_records(data_path: str) -> List[Dict]:
    """Load records from a JSON array file or a JSONL file."""
    path = Path(data_path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


class CustomDataset:
    """Reproducible sequential / random-access dataset.

    Each record must follow your custom schema.
    
    Args:
        data_path: Path to JSON/JSONL annotation file.
        image_dir: Directory containing images (if applicable).
        mode: "sample" for random access (train) or "traversal" for sequential (eval).
        seed: RNG seed for reproducibility.
        max_image_width: Optional image resizing.
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        mode: str = "sample",
        seed: Optional[int] = None,
        max_image_width: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.image_dir = image_dir
        self.mode = mode
        self.max_image_width = max_image_width
        self.records = _load_records(data_path)
        self._rng = random.Random(seed)
        self._idx = 0
        self._seed_map: Dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.records)

    def get_item(self, seed: int) -> Dict:
        """Return the record associated with seed.
        
        Ensures all workers in the same group see identical data.
        """
        if seed not in self._seed_map:
            if self.mode == "traversal":
                idx = self._idx % len(self.records)
                self._idx += 1
            else:
                idx = self._rng.randint(0, len(self.records) - 1)
            self._seed_map[seed] = idx

        record = self.records[self._seed_map[seed]]
        
        # Load and process your data here
        # Example: load image, process text, etc.
        
        return {
            "image": image,
            "gt_target": target,
            "query": query,
            # ... other fields
        }
```

**Key Points**:
- Use `seed`-based indexing for reproducibility across workers
- Support both "sample" (random) and "traversal" (sequential) modes
- Return a dict with all data needed for the environment

## Step 2: Create Reward Functions

Define how to compute rewards for your task.

**File**: `roll/pipeline/agentic/env/{your_env}/rewards.py`

```python
"""Reward functions for your custom environment."""
import math
from typing import Tuple, Optional


def compute_custom_reward(
    prediction: Any,
    ground_truth: Any,
    **kwargs: Any,
) -> float:
    """Compute reward based on your task requirements.
    
    Examples:
    - IoU for bounding box tasks
    - Accuracy for classification
    - BLEU/ROUGE for text generation
    - Custom domain-specific metrics
    """
    # Implement your reward logic here
    reward = 0.0
    
    # Example: simple accuracy
    if prediction == ground_truth:
        reward = 1.0
    
    return reward
```

## Step 3: Create the Toolbox (Optional)

If your environment uses tool-based interactions (like the medical grounding example with zoom/pan tools), create a toolbox to parse and execute tool calls.

**File**: `roll/pipeline/agentic/env/{your_env}/toolbox.py`

```python
"""Toolbox for parsing and executing agent tool calls."""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import PIL.Image as Image


class CustomToolBox:
    """Tracks state and executes tools for your environment."""

    # Define regex patterns for your tools
    _TOOL_CALL_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
    _TOOL1_RE = re.compile(r"tool1\[([\d.]+)\]")
    _TOOL2_RE = re.compile(r"tool2\[([\d.]+),\s*([\d.]+)\]")
    _ANSWER_RE = re.compile(r"<answer>(.*?)</answer>")

    def __init__(self, initial_state: Any) -> None:
        self.state = initial_state
        # Initialize any state tracking here

    def parse_and_execute(self, action: str) -> Dict[str, Any]:
        """Parse an action string and execute the matching tool.
        
        Returns a dict with:
            done (bool): True if episode should end
            result (Any): Tool execution result
            tool_name (Optional[str]): Which tool was called
            tool_args (List): Arguments passed to tool
            tool_success (bool): Whether tool executed successfully
            format_error (bool): True if no valid action found
            error_msg (Optional[str]): Error message if applicable
        """
        # Check for final answer
        answer_m = self._ANSWER_RE.search(action)
        if answer_m:
            answer = answer_m.group(1)
            return {
                "done": True,
                "result": answer,
                "tool_name": "answer",
                "tool_args": [answer],
                "tool_success": True,
                "format_error": False,
                "error_msg": None,
            }

        # Check for tool call block
        call_m = self._TOOL_CALL_RE.search(action)
        if not call_m:
            return {
                "done": False,
                "result": None,
                "tool_name": None,
                "tool_args": [],
                "tool_success": False,
                "format_error": True,
                "error_msg": "No tool call or answer found",
            }

        content = call_m.group(1).strip()
        
        # Define your tool handlers
        tool_handlers = [
            ("tool1", self._TOOL1_RE, lambda m: self._do_tool1(float(m.group(1)))),
            ("tool2", self._TOOL2_RE, lambda m: self._do_tool2(float(m.group(1)), float(m.group(2)))),
        ]

        for tool_name, pattern, handler in tool_handlers:
            m = pattern.search(content)
            if m:
                try:
                    args = [float(v) for v in m.groups()] if m.groups() else []
                    result = handler(m)
                    return {
                        "done": False,
                        "result": result,
                        "tool_name": tool_name,
                        "tool_args": args,
                        "tool_success": True,
                        "format_error": False,
                        "error_msg": None,
                    }
                except (ValueError, Exception) as e:
                    return {
                        "done": False,
                        "result": None,
                        "tool_name": tool_name,
                        "tool_args": [],
                        "tool_success": False,
                        "format_error": True,
                        "error_msg": str(e),
                    }

        return {
            "done": False,
            "result": None,
            "tool_name": None,
            "tool_args": [],
            "tool_success": False,
            "format_error": True,
            "error_msg": f"Unrecognized tool command: {content}",
        }

    def _do_tool1(self, arg: float) -> bool:
        """Execute tool1 with validation."""
        # Validate arguments
        if not (0.0 <= arg <= 1.0):
            return False
        # Update state
        self.state = self._apply_tool1(arg)
        return True

    def _do_tool2(self, arg1: float, arg2: float) -> bool:
        """Execute tool2 with validation."""
        # Validate arguments
        if not (0.0 <= arg1 <= 1.0 and 0.0 <= arg2 <= 1.0):
            return False
        # Update state
        self.state = self._apply_tool2(arg1, arg2)
        return True
```

**Key Points**:
- Use regex to parse tool calls from model output
- Return structured results with success/failure information
- Handle format errors gracefully (don't crash on invalid model output)

## Step 4: Create the Environment Class

The core environment implements the Gym API (`reset`, `step`) and defines your task logic.

**File**: `roll/pipeline/agentic/env/{your_env}/env.py`

```python
"""Custom RL Environment for ROLL."""
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PIL.Image as Image
from gem import Env

from roll.pipeline.agentic.env.{your_env}.dataset import CustomDataset
from roll.pipeline.agentic.env.{your_env}.rewards import compute_custom_reward
from roll.pipeline.agentic.env.{your_env}.toolbox import CustomToolBox
from roll.utils.constants import EpisodeStopReason
from roll.utils.logging import get_logger

logger = get_logger()

# Image placeholder that VLTrajEnvManager replaces with model's image token
_IMAGE_PLACEHOLDER = "<image>"


_STEP_TEMPLATE = """\
Current observation:
  Query: {query}
  State: {state}
  Steps remaining: {steps_left}

{image_placeholder}
"""


class CustomEnv(Env):
    """RL environment for your custom task.

    Args:
        data_path: Path to annotation file.
        image_dir: Directory containing images.
        mode: "train" or "val".
        seed: Dataset RNG seed.
        max_steps: Maximum steps per episode.
        step_penalty: Negative reward per step.
        format_penalty: Extra penalty for invalid actions.
        trajectory_log_dir: Optional directory for saving trajectories.
        max_logged_trajectories: Max trajectories to save per env.
        max_image_width: Optional image resizing.
        **kwargs: Additional arguments.
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
        trajectory_log_dir: Optional[str] = None,
        max_logged_trajectories: int = 20,
        max_image_width: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.format_penalty = format_penalty
        self.trajectory_log_dir = trajectory_log_dir
        self.max_logged_trajectories = max_logged_trajectories

        dataset_mode = "sample" if mode == "train" else "traversal"
        self.dataset = CustomDataset(
            data_path=data_path,
            image_dir=image_dir,
            mode=dataset_mode,
            seed=seed,
            max_image_width=max_image_width,
        )

        # Episode state
        self._data_item: Optional[Dict] = None
        self._toolbox: Optional[CustomToolBox] = None
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
            obs: First observation dict.
            info: Optional info dict (can be empty).
        """
        data = self.dataset.get_item(seed if seed is not None else 0)
        self._data_item = data
        self._toolbox = CustomToolBox(data["initial_state"])
        self._step_count = 0
        self._trajectory = []

        obs = self._build_obs()

        # Add initial state to trajectory
        self._trajectory.append({
            "step": 0,
            "prompt": obs["prompt"],
            "response": None,
            "tool_name": None,
            "tool_args": [],
            "tool_success": False,
            "format_error": False,
            "error_msg": None,
            "state": self._toolbox.state,
            "step_reward": 0.0,
            "predicted": None,
            "gt_target": data["gt_target"],
            "success": False,
        })

        return obs, {}

    def step(self, action: str) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute one agent action.

        Args:
            action: Raw model output string.

        Returns:
            (obs, reward, terminated, truncated, info) tuple.
            obs is "" when episode is done.
        """
        self._step_count += 1

        # Handle MAX_LENGTH stop reason
        if isinstance(action, EpisodeStopReason) and action == EpisodeStopReason.MAX_LENGTH:
            logger.info(f"[MAX_LENGTH] Episode terminated, step_count={self._step_count}")
            done = True
            truncated = True
            reward = 0.0
            info = {}
            return "", reward, done, truncated, info

        # Capture current observation
        current_obs = self._build_obs()
        current_prompt = current_obs["prompt"]

        # Parse and execute action
        result = self._toolbox.parse_and_execute(action)
        tool_name = result["tool_name"]
        tool_success = result["tool_success"]
        format_error = result["format_error"]
        done = result["done"]
        prediction = result["result"]

        data = self._data_item
        gt_target = data["gt_target"]

        # Compute reward
        step_reward = self.step_penalty
        if format_error:
            step_reward += self.format_penalty
        if done and prediction is not None:
            step_reward += compute_custom_reward(prediction, gt_target)
        elif tool_success:
            # Optional: intermediate rewards for tool usage
            step_reward += 0.0  # Add intermediate reward logic here

        success = compute_custom_reward(prediction, gt_target) > 0.5 if done else False

        # Track trajectory
        self._trajectory.append({
            "step": self._step_count,
            "prompt": current_prompt,
            "response": action,
            "tool_name": tool_name,
            "tool_args": result.get("tool_args", []),
            "tool_success": tool_success,
            "format_error": format_error,
            "error_msg": result.get("error_msg"),
            "state": self._toolbox.state,
            "step_reward": round(step_reward, 4),
            "predicted": prediction,
            "gt_target": gt_target,
            "success": success,
        })

        # Truncate at max_steps
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
            "success": float(success),
        }
        metrics_agg_mode = {
            "tool_success": "mean",
            "format_error": "mean",
            "success": "last",
        }
        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "action_desc": f"step={self._step_count} tool={tool_name or 'invalid'} r={step_reward:.3f}",
        }

        return next_obs, step_reward, done, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> Dict:
        """Construct the current observation."""
        data = self._data_item
        state = self._toolbox.state
        
        prompt = _STEP_TEMPLATE.format(
            query=data["query"],
            state=state,
            steps_left=self.max_steps - self._step_count,
            image_placeholder=_IMAGE_PLACEHOLDER,
        )
        
        return {
            "prompt": prompt,
            "image": [data["image"]],  # List of PIL Images
        }

    def _maybe_save_trajectory(self) -> None:
        """Save trajectory to disk if logging is configured."""
        if not self.trajectory_log_dir:
            return
        
        with self._log_lock:
            if self._logged_count >= self.max_logged_trajectories:
                return
            self._logged_count += 1

        traj_id = str(uuid.uuid4())[:8]
        traj_name = f"{self._data_item.get('id', 'unknown')}_{traj_id}"
        out_dir = Path(self.trajectory_log_dir) / traj_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save trajectory data
        traj_data = {
            "query": self._data_item["query"],
            "gt_target": self._data_item["gt_target"],
            "steps": self._trajectory,
            "episode_reward": sum(s["step_reward"] for s in self._trajectory),
            "success": self._trajectory[-1]["success"] if self._trajectory else False,
        }
        
        with open(out_dir / "trajectory.json", "w") as f:
            json.dump(traj_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"[CustomEnv] Saved trajectory to {out_dir}")
```

**Key Points**:
- Inherit from `gem.Env` (the GYM-like interface used by ROLL)
- Implement `reset(seed)` and `step(action)` methods
- Return observations as a dict with `"prompt"` and `"image"` keys
- Return metrics in `info["metrics"]` with aggregation modes in `info["metrics_agg_mode"]`
- Handle `EpisodeStopReason.MAX_LENGTH` for sequence length limits
- Support trajectory logging for debugging and visualization

## Step 5: Register Your Environment

Register your environment with the GEM framework so it can be instantiated by name.

**File**: `roll/pipeline/agentic/env/{your_env}/__init__.py`

```python
"""Custom environment package."""
from gem import register

from roll.pipeline.agentic.env.{your_env}.env import CustomEnv

# Register the environment with GEM
register(
    env_id="your_env_name",
    entry_point="roll.pipeline.agentic.env.{your_env}.env:CustomEnv",
)
```

## Step 6: Create Configuration File

Create a YAML configuration file for your environment.

**File**: `examples/{your_env}/{your_env}.yaml`

```yaml
defaults:
  - ../config/envs@_here_
  - ../config/deepspeed_zero@_here_

hydra:
  run:
    dir: .
  output_subdir: null

# Use base AgenticPipeline or create a custom subclass
pipeline_cls: roll.pipeline.agentic.agentic_pipeline.AgenticPipeline

exp_name: "your_experiment_name"
seed: 42
logging_dir: ./output/logs
output_dir: ./output

checkpoint_config:
  type: file_system
  output_dir: /data/checkpoints/${exp_name}

track_with: wandb  # or tensorboard, mlflow, etc.
tracker_kwargs:
  project: your-project-name
  name: ${exp_name}

num_gpus_per_node: 8

max_steps: 1000
save_steps: 100
logging_steps: 1
eval_steps: 50

rollout_batch_size: 256
val_batch_size: 64
response_length: 4096
sequence_length: 20480

reward_clip: 10
advantage_clip: 5.0
ppo_epochs: 1
adv_estimator: "grpo"  # or "gae"
whiten_advantages: false

pretrain: your-model-name

# Actor training configuration
actor_train:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: false
    dtype: bf16
  training_args:
    learning_rate: 1.0e-6
    lr_scheduler_type: constant
    weight_decay: 1.0e-2
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 64
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      tensor_model_parallel_size: 1
      sequence_parallel: true

# Actor inference configuration
actor_infer:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 1.0
    temperature: 1.0
  strategy_args:
    strategy_name: vllm
    strategy_config:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.85

max_actions_per_traj: 5

reward_normalization:
  grouping: traj_group_id
  method: mean_std

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
custom_envs:
  your_env_name:
    env_type: your_env_name  # Must match registered env_id
    max_steps: ${max_actions_per_traj}
    max_tokens_per_step: ${response_length}
    env_manager_cls: roll.pipeline.agentic.env_manager.vl_traj_env_manager.VLTrajEnvManager
    use_thread_lock: true

    # System prompt for the agent
    agent_system_template: |
      You are a specialized AI Agent for your task.
      
      Describe the task and available tools here.
      
      Example turn:
        <thinking>Reasoning...</thinking>
        ```python
        tool1[0.5]
        ```

    # Text injected before/after each observation turn
    pre_step_template: ""
    next_step_template: ""

    env_config:
      # --- REQUIRED: update these paths ---
      data_path: /path/to/your/data.jsonl
      image_dir: /path/to/your/images
      # ---

      mode: train
      seed: ${seed}
      max_steps: ${max_actions_per_traj}
      max_image_width: 512

      # Reward shaping
      step_penalty: -0.01
      format_penalty: -0.5

      # Trajectory logging
      trajectory_log_dir: ./output/trajectories
      max_logged_trajectories: 1

train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 32
  group_size: 8
  tags: [your_env_name]
  num_groups_partition:
    - 32

val_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: ${val_batch_size}
  group_size: 1
  tags: [your_env_name]
  num_groups_partition:
    - ${val_batch_size}
```

## Step 7: (Optional) Create Custom Pipeline

If you need custom logging or other pipeline-level customizations, create a pipeline subclass.

**File**: `roll/pipeline/agentic/{your_env}_pipeline.py`

```python
"""Custom pipeline for your environment."""
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline


class CustomPipeline(AgenticPipeline):
    """AgenticPipeline subclass with custom behavior."""

    def __init__(self, pipeline_config: Any) -> None:
        super().__init__(pipeline_config)
        # Add custom initialization here

    def val(self, global_step: int) -> Dict:
        """Run validation with custom logging."""
        metrics = super().val(global_step)
        # Add custom validation logic here
        return metrics
```

Then update your YAML:
```yaml
pipeline_cls: roll.pipeline.agentic.{your_env}_pipeline.CustomPipeline
```

## Step 8: Run Training

```bash
python examples/start_agentic_pipeline.py \
    --config_path your_env \
    --config_name your_env
```

## Key Design Patterns

### Observation Format

The environment must return observations in this format:

```python
{
    "prompt": str,          # Text prompt with <image> placeholder
    "image": [PIL.Image],   # List of PIL Images (even for single image)
}
```

The `<image>` placeholder is replaced by the VLTrajEnvManager with the model's image token.

### Action Format

The environment receives raw model output strings. You can define any format:
- Tool-based: ````python tool[arg1, arg2]````
- Direct answer: `<answer>result</answer>`
- Free-form: plain text

Parse the action in your `step()` method or toolbox.

### Reward Structure

Return per-step rewards in the `step()` method. Common patterns:
- Step penalty: negative reward for each step to encourage efficiency
- Format penalty: extra penalty for invalid actions
- Intermediate rewards: reward for good tool usage
- Final reward: reward based on final answer quality

### Metrics

Return metrics in `info["metrics"]` with aggregation modes:
```python
info = {
    "metrics": {
        "success": 1.0,      # Scalar metric
        "tool_success": 0.8, # Mean across steps
    },
    "metrics_agg_mode": {
        "success": "last",      # Take last value
        "tool_success": "mean", # Average across steps
    },
}
```

Supported aggregation modes: `"mean"`, `"last"`, `"sum"`, `"max"`, `"min"`.

### Trajectory Logging

Save trajectories for debugging and visualization:
- JSON file with step-by-step data
- PNG images for visual tasks
- Must use thread-safe writes (multiple workers)

## Common Pitfalls

1. **Not handling MAX_LENGTH**: Always check for `EpisodeStopReason.MAX_LENGTH` in `step()`
2. **Invalid float conversion**: Use try-except when parsing numeric arguments from model output
3. **Thread safety**: Use locks when writing trajectories (multiple workers)
4. **Image format**: Always return images as a list, even for single images
5. **Seed reproducibility**: Use seed-based indexing in dataset for reproducible rollouts
6. **Metric aggregation**: Always provide `metrics_agg_mode` for custom metrics

## Testing Your Environment

Create a simple test script:

```python
from roll.pipeline.agentic.env.{your_env}.env import CustomEnv

env = CustomEnv(
    data_path="test_data.jsonl",
    image_dir="test_images/",
    mode="val",
    seed=42,
)

# Test reset
obs, info = env.reset(seed=0)
print("Initial obs:", obs["prompt"][:100])

# Test step
action = "<answer>test_answer</answer>"
obs, reward, done, truncated, info = env.step(action)
print(f"Reward: {reward}, Done: {done}")
print("Metrics:", info["metrics"])
```

## Additional Resources

- Base environment manager: `roll/pipeline/agentic/env_manager/vl_traj_env_manager.py`
- Base pipeline: `roll/pipeline/agentic/agentic_pipeline.py`
- Medical grounding example: `roll/pipeline/agentic/env/medical_grounding/`
- Configuration examples: `examples/medical_grounding/`
