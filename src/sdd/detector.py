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
    area: float
    area_norm: float
    threat_score: float
    threat_level: str


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

        height, width = frame.shape[:2]
        frame_area = float(width * height) if width > 0 and height > 0 else 0.0

        def _compute_threat(x1: float, y1: float, x2: float, y2: float, score: float) -> tuple[float, str, float, float]:
            # Obszar i znormalizowany obszar detekcji
            area = max(0.0, (x2 - x1) * (y2 - y1))
            area_norm = area / frame_area if frame_area > 0 else 0.0

            # Pozycja względem środka kadru (bliżej środka = potencjalnie większe zagrożenie)
            if width > 0 and height > 0:
                cx = ((x1 + x2) / 2.0) / width
                cy = ((y1 + y2) / 2.0) / height
                dx = cx - 0.5
                dy = cy - 0.5
                dist_center = (dx * dx + dy * dy) ** 0.5
                proximity = 1.0 - min(dist_center / 0.5, 1.0)
            else:
                proximity = 0.0

            # Prosta, heurystyczna funkcja punktowa 0-1
            # - score (pewność modelu)
            # - area_norm (jak duży obiekt w kadrze)
            # - proximity (jak blisko środka kadru)
            threat_score = (
                0.6 * max(0.0, min(1.0, score))
                + 0.3 * max(0.0, min(1.0, area_norm * 4.0))  # wzmocnienie dużych obiektów
                + 0.1 * max(0.0, min(1.0, proximity))
            )

            if threat_score >= 0.6:
                threat_level = "high"
            elif threat_score >= 0.3:
                threat_level = "medium"
            else:
                threat_level = "low"

            return threat_score, threat_level, area, area_norm

        for box in boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if self._target_ids is not None and cls_id not in self._target_ids:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0]) if box.conf is not None else 0.0
            label = result.names.get(cls_id, str(cls_id))

            threat_score, threat_level, area, area_norm = _compute_threat(
                float(x1), float(y1), float(x2), float(y2), score
            )

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
                    area=area,
                    area_norm=area_norm,
                    threat_score=threat_score,
                    threat_level=threat_level,
                )
            )

        return detections
