from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FrameGeometry:
    width: int
    height: int

    @property
    def area(self) -> float:
        return float(max(0, self.width) * max(0, self.height))


def classify_label_safety(label: str) -> str:
    """Klasyfikuje etykietę na kategorie bezpieczeństwa.

    Ma to być prosta, domenowa heurystyka:
    - "danger": obiekty latające / potencjalnie niebezpieczne
      (airplane, helicopter, bird, kite, balloon, drone, paraglider, hang glider),
    - "medium": pojazdy lądowe / infrastruktura (car, truck, bus, train, boat,
      ship, motorcycle, bicycle, parking meter),
    - "safe": reszta.
    """

    name = (label or "").strip().lower()

    dangerous_keywords = {
        "airplane",
        "helicopter",
        "bird",
        "kite",
        "balloon",
        "drone",
        "paraglider",
        "hang glider",
    }

    medium_keywords = {
        "car",
        "truck",
        "bus",
        "train",
        "boat",
        "ship",
        "motorcycle",
        "bicycle",
        "parking meter",
    }

    if name in dangerous_keywords:
        return "danger"
    if name in medium_keywords:
        return "medium"
    return "safe"


def compute_threat_metrics(
    label: str,
    score: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame: FrameGeometry,
) -> Tuple[float, str, float, float]:
    """Wylicza threat_score, threat_level oraz metryki geometryczne.

    Zasada działania (0–1):
    - główne źródło: pewność modelu (score),
    - dodatkowo: wielkość obiektu w kadrze (area_norm),
    - położenie względem środka (proximity),
    - lekka korekta zależna od klasy (danger/medium/safe).
    """

    frame_area = frame.area

    # Obszar i znormalizowany obszar detekcji
    area = max(0.0, (x2 - x1) * (y2 - y1))
    area_norm = area / frame_area if frame_area > 0 else 0.0

    # Pozycja względem środka kadru (bliżej środka = potencjalnie większe zagrożenie)
    if frame.width > 0 and frame.height > 0:
        cx = ((x1 + x2) / 2.0) / frame.width
        cy = ((y1 + y2) / 2.0) / frame.height
        dx = cx - 0.5
        dy = cy - 0.5
        dist_center = (dx * dx + dy * dy) ** 0.5
        proximity = 1.0 - min(dist_center / 0.5, 1.0)
    else:
        proximity = 0.0

    # Korekta zależna od typu obiektu
    safety = classify_label_safety(label)
    if safety == "danger":
        safety_bias = 0.20
    elif safety == "medium":
        safety_bias = 0.10
    else:
        safety_bias = 0.0

    score_clamped = max(0.0, min(1.0, score))
    area_term = max(0.0, min(1.0, area_norm * 4.0))  # wzmocnienie dużych obiektów
    proximity_term = max(0.0, min(1.0, proximity))

    threat_score = (
        0.55 * score_clamped
        + 0.25 * area_term
        + 0.10 * proximity_term
        + 0.10 * safety_bias
    )

    # Dodatkowe obcięcie do [0, 1]
    threat_score = max(0.0, min(1.0, threat_score))

    if threat_score >= 0.65:
        threat_level = "high"
    elif threat_score >= 0.35:
        threat_level = "medium"
    else:
        threat_level = "low"

    return threat_score, threat_level, area, area_norm
