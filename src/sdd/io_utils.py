from __future__ import annotations

import json
import csv
from dataclasses import asdict
from typing import Iterable, Dict
from collections import defaultdict

from .detector import Detection


def save_detections_json(detections: Iterable[Detection], path: str) -> None:
    data = [asdict(d) for d in detections]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_detections_csv(detections: Iterable[Detection], path: str) -> None:
    detections = list(detections)
    if not detections:
        # create empty file with header
        fieldnames = [
            "frame_index",
            "timestamp",
            "x1",
            "y1",
            "x2",
            "y2",
            "score",
            "label",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    fieldnames = list(asdict(detections[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in detections:
            writer.writerow(asdict(d))


def save_events_summary(detections: Iterable[Detection], path: str) -> None:
    """Save aggregated per-class event summary for ML analysis.

    For each label we store:
    - count of detections
    - first/last frame index
    - first/last timestamp
    - min/max/mean score
    """

    detections_list = list(detections)

    fieldnames = [
        "label",
        "count",
        "first_frame_index",
        "last_frame_index",
        "first_timestamp",
        "last_timestamp",
        "min_score",
        "max_score",
        "mean_score",
        "max_threat_score",
        "mean_threat_score",
        "dominant_threat_level",
    ]

    if not detections_list:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    stats: Dict[str, Dict[str, float]] = defaultdict(dict)

    for det in detections_list:
        label = det.label
        s = stats.get(label)
        if not s:
            s = {
                "count": 0,
                "first_frame_index": det.frame_index,
                "last_frame_index": det.frame_index,
                "first_timestamp": det.timestamp,
                "last_timestamp": det.timestamp,
                "min_score": det.score,
                "max_score": det.score,
                "score_sum": 0.0,
                "max_threat_score": det.threat_score,
                "threat_score_sum": 0.0,
                "threat_level_counts": {"low": 0, "medium": 0, "high": 0},
            }
            stats[label] = s

        s["count"] += 1
        if det.frame_index < s["first_frame_index"]:
            s["first_frame_index"] = det.frame_index
        if det.frame_index > s["last_frame_index"]:
            s["last_frame_index"] = det.frame_index

        if det.timestamp < s["first_timestamp"]:
            s["first_timestamp"] = det.timestamp
        if det.timestamp > s["last_timestamp"]:
            s["last_timestamp"] = det.timestamp

        if det.score < s["min_score"]:
            s["min_score"] = det.score
        if det.score > s["max_score"]:
            s["max_score"] = det.score

        s["score_sum"] += det.score

        # Threat-related stats
        if det.threat_score > s["max_threat_score"]:
            s["max_threat_score"] = det.threat_score
        s["threat_score_sum"] += det.threat_score
        if det.threat_level in s["threat_level_counts"]:
            s["threat_level_counts"][det.threat_level] += 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in sorted(stats.keys()):
            s = stats[label]
            mean_score = s["score_sum"] / s["count"] if s["count"] > 0 else 0.0
            mean_threat_score = (
                s["threat_score_sum"] / s["count"] if s["count"] > 0 else 0.0
            )

            level_counts = s["threat_level_counts"]
            dominant_level = max(
                ("low", "medium", "high"),
                key=lambda lvl: level_counts.get(lvl, 0),
            )
            writer.writerow(
                {
                    "label": label,
                    "count": int(s["count"]),
                    "first_frame_index": int(s["first_frame_index"]),
                    "last_frame_index": int(s["last_frame_index"]),
                    "first_timestamp": float(s["first_timestamp"]),
                    "last_timestamp": float(s["last_timestamp"]),
                    "min_score": float(s["min_score"]),
                    "max_score": float(s["max_score"]),
                    "mean_score": float(mean_score),
                    "max_threat_score": float(s["max_threat_score"]),
                    "mean_threat_score": float(mean_threat_score),
                    "dominant_threat_level": dominant_level,
                }
            )
