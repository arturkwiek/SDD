from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os


# Domyślny katalog z próbkami wideo / obrazów.
# - w natywnym Windows: C:\\VideoSource
# - w WSL: /mnt/c/VideoSource
if os.name == "nt":
    SAMPLE_PATH: Path = Path("C:/VideoSource")
else:
    SAMPLE_PATH: Path = Path("/mnt/c/VideoSource")


@dataclass
class DetectorConfig:
    model_name: str = "yolov8n.pt"  # lightweight default model
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    target_classes: Optional[List[str]] = None  # e.g. ["drone"] when using a dedicated model


@dataclass
class RuntimeConfig:
    show_preview: bool = True
    save_video: bool = False
    output_video_path: str = "output_detection.mp4"
    output_json_path: str = "detections.json"
    output_csv_path: str = "detections.csv"
