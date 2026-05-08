"""Dataset loading and sampling for medical image grounding."""
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


class MedicalGroundingDataset:
    """Reproducible sequential / random-access dataset for medical grounding.

    Each record must follow the schema::

        {
            "bbox": [x1, y1, x2, y2],   # absolute pixel coords
            "image_id": "filename.jpg",
            "height": int,
            "width": int,
            "bbox_id": int,
            "expression": str | List[str]
        }

    Args:
        data_path: Path to JSON/JSONL annotation file.
        image_dir: Directory containing the images referenced by ``image_id``.
        mode: ``"sample"`` for random access (train) or ``"traversal"`` for
            sequential access (eval).
        seed: RNG seed for the random sampler.
        max_image_width: If set, resize images to this max width while preserving
            aspect ratio. If None, images are loaded at original resolution.
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        mode: str = "sample",
        seed: Optional[int] = None,
        max_image_width: Optional[int] = None,
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
        """Return the record and loaded image associated with *seed*.

        Repeated calls with the same seed return the same record, ensuring
        that all workers in the same group see identical data.
        """
        if seed not in self._seed_map:
            if self.mode == "traversal":
                idx = self._idx % len(self.records)
                self._idx += 1
            else:
                idx = self._rng.randint(0, len(self.records) - 1)
            self._seed_map[seed] = idx

        record = self.records[self._seed_map[seed]]
        expression = record["expression"]
        if isinstance(expression, list):
            expression = expression[0]

        image_path = os.path.join(self.image_dir, record["image_id"])
        image = Image.open(image_path).convert("RGB")

        # Resize image to max width while preserving aspect ratio
        if self.max_image_width is not None:
            w, h = image.size
            if w > self.max_image_width:
                scale = self.max_image_width / w
                new_w = self.max_image_width
                new_h = int(h * scale)
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return {
            "image": image,
            "gt_bbox": tuple(float(v) for v in record["bbox"]),  # (x1, y1, x2, y2) abs px
            "expression": expression,
            "image_id": record["image_id"],
            "height": int(record["height"]),
            "width": int(record["width"]),
            "bbox_id": int(record.get("bbox_id", 0)),
        }
