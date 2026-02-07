from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import DetectorConfig


@dataclass
class Detection:
    frame_index: int
    timestamp: float
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str


class BaseDetector:
    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()

    def detect(self, frame: np.ndarray, frame_index: int, timestamp: float) -> List[Detection]:
        raise NotImplementedError


class YoloDetector(BaseDetector):
    """YOLO-based detector wrapped behind a simple interface.

    By default uses a generic model from ultralytics. For better drone
    performance plug in a drone-specific checkpoint via DetectorConfig.model_name.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        super().__init__(config)
        from ultralytics import YOLO  # lazy import to avoid hard dependency at import time

        self._model = YOLO(self.config.model_name)

        # Map target class names to model class IDs if specified
        self._target_ids = None
        if self.config.target_classes is not None:
            name_to_id = {name: i for i, name in self._model.model.names.items()}
            self._target_ids = {
                name_to_id[name]
                for name in self.config.target_classes
                if name in name_to_id
            }

    def detect(self, frame: np.ndarray, frame_index: int, timestamp: float) -> List[Detection]:
        results = self._model.predict(
            source=frame,
            verbose=False,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if self._target_ids is not None and cls_id not in self._target_ids:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0]) if box.conf is not None else 0.0
            label = result.names.get(cls_id, str(cls_id))

            detections.append(
                Detection(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=score,
                    label=label,
                )
            )

        return detections
