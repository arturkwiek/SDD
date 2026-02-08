from __future__ import annotations

import argparse
import time
from typing import Optional, List

import cv2

from .config import DetectorConfig, RuntimeConfig, SAMPLE_PATH
from .detector import YoloDetector, Detection
from .io_utils import save_detections_json, save_detections_csv, save_events_summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Drone Detector")

    parser.add_argument(
        "source",
        help=(
            "Źródło wideo/obrazu: ścieżka do pliku (jpg/mp4 itp.), "
            "indeks kamery (0,1,...) lub URL strumienia (RTSP/HTTP)."
        ),
    )

    parser.add_argument("--model", default="yolov8n.pt", help="Nazwa/ścieżka modelu YOLO.")
    parser.add_argument("--conf", type=float, default=0.35, help="Próg pewności detekcji.")
    parser.add_argument("--iou", type=float, default=0.45, help="Próg IoU dla NMS.")
    parser.add_argument(
        "--target-classes",
        nargs="*",
        default=None,
        help="Lista nazw klas do filtrowania (np. drone) – zależy od modelu.",
    )

    parser.add_argument("--no-preview", action="store_true", help="Wyłącza podgląd video.")
    parser.add_argument("--save-video", action="store_true", help="Zapisuje wideo z narysowanymi detekcjami.")
    parser.add_argument("--video-out", default="output_detection.mp4", help="Ścieżka zapisu wideo.")
    parser.add_argument("--json-out", default="detections.json", help="Ścieżka zapisu wyników JSON.")
    parser.add_argument("--csv-out", default="detections.csv", help="Ścieżka zapisu wyników CSV.")

    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Wyłącza wyświetlanie statystyk (klatka, czas, liczba detekcji, FPS) na obrazie.",
    )

    parser.add_argument(
        "--from-samples",
        action="store_true",
        help=(
            "Traktuje parametr 'source' jako nazwę pliku w katalogu "
            f"z próbkami ({SAMPLE_PATH})."
        ),
    )

    return parser.parse_args(argv)


def open_source(source: str) -> cv2.VideoCapture:
    # jeżeli podano liczbę, traktujemy jako indeks kamery
    if source.isdigit():
        cam_index = int(source)
        print(f"[SDD][INFO] Źródło interpretowane jako kamera USB o indeksie {cam_index}.")
        cap = cv2.VideoCapture(cam_index)
    else:
        print(f"[SDD][INFO] Źródło interpretowane jako plik/strumień: {source}")
        cap = cv2.VideoCapture(source)

    return cap


def classify_label_safety(label: str) -> str:
    """Heurystyczna kategoryzacja etykiety na safe/medium/danger.

    Używana do kolorowania ramek wokół obiektów na wideo.
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


def draw_detections(frame, detections: List[Detection]):
    for det in detections:
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])

        safety = classify_label_safety(det.label)
        if safety == "danger":
            color = (0, 0, 255)  # red
        elif safety == "medium":
            color = (0, 255, 255)  # yellow
        else:
            color = (0, 255, 0)  # green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.label} {det.score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_stats_overlay(
    frame,
    frame_index: int,
    timestamp: float,
    frame_detections: int,
    total_detections: int,
    source_fps: float,
    frame_threat_level: str,
    frame_max_threat_score: float,
    total_frames: int,
):
    """Rysuje proste statystyki (w tym poziom zagrożenia) w lewym górnym rogu klatki."""

    processing_fps = (frame_index / timestamp) if timestamp > 0 else 0.0

    lines = [
        f"Frame: {frame_index}",
        f"Time: {timestamp:0.2f}s",
        f"Detections: {frame_detections} (total {total_detections})",
    ]

    if total_frames and total_frames > 0:
        # Użyj frame_index+1, bo index startuje od 0
        current = frame_index + 1
        denom = max(total_frames, 1)
        progress = 100.0 * current / denom
        lines.append(f"Progress: {current}/{total_frames} ({progress:0.1f}%)")

    if frame_detections > 0:
        lines.append(
            f"Threat: {frame_threat_level.upper()} ({frame_max_threat_score:0.2f})"
        )

    if source_fps and source_fps > 0:
        lines.append(f"Source FPS: {source_fps:0.1f}")
    if processing_fps > 0:
        lines.append(f"Proc FPS: {processing_fps:0.1f}")

    x = 10
    y_start = 20
    dy = 18
    for i, text in enumerate(lines):
        y = y_start + i * dy
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def run() -> None:
    args = parse_args()

    source = args.source
    if args.from_samples:
        # użycie nazw plików względnie do domyślnego katalogu z próbkami
        source = str(SAMPLE_PATH / source)

    print("[SDD] Start detekcji")
    print(f"[SDD] Źródło: {source}")

    det_cfg = DetectorConfig(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        target_classes=args.target_classes,
    )
    rt_cfg = RuntimeConfig(
        show_preview=not args.no_preview,
        save_video=args.save_video,
        output_video_path=args.video_out,
        output_json_path=args.json_out,
        output_csv_path=args.csv_out,
        show_stats=not args.no_stats,
    )

    print(f"[SDD] Model: {det_cfg.model_name}, conf>={det_cfg.conf_threshold}, iou>={det_cfg.iou_threshold}")
    if det_cfg.target_classes:
        print(f"[SDD] Filtrowane klasy: {det_cfg.target_classes}")

    detector = YoloDetector(det_cfg)

    cap = open_source(source)
    if not cap.isOpened():
        print(f"[SDD][ERROR] Nie można otworzyć źródła: {source}")
        return

    writer: Optional[cv2.VideoWriter] = None
    all_detections: List[Detection] = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(
        f"[SDD][INFO] Parametry źródła: fps={fps:.2f} (0=nieznane), "
        f"rozmiar={width}x{height}, "
        f"liczba_klatek={total_frames if total_frames > 0 else 'nieznana'}"
    )

    if rt_cfg.save_video and fps > 0 and width > 0 and height > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(rt_cfg.output_video_path, fourcc, fps, (width, height))
        if writer is None or not writer.isOpened():
            print(
                f"[SDD][WARN] Nie udało się utworzyć pliku wideo wyjściowego: "
                f"{rt_cfg.output_video_path}"
            )
            writer = None
    elif rt_cfg.save_video:
        print(
            "[SDD][WARN] Zapis wideo włączony, ale brak poprawnych parametrów FPS/rozmiaru; "
            "pomijam zapis wideo."
        )

    frame_index = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_index == 0:
                print("[SDD][ERROR] Nie udało się odczytać żadnej klatki ze źródła.")
            else:
                print("[SDD][INFO] Odczyt źródła zakończony (brak kolejnych klatek).")
            break

        timestamp = time.time() - start_time
        detections = detector.detect(frame, frame_index, timestamp)
        all_detections.extend(detections)

        if frame_index % 30 == 0:
            print(
                f"[SDD] Klatka {frame_index}: {len(detections)} detekcji, "
                f"czas {timestamp:0.2f}s"
            )

        if rt_cfg.show_preview or writer is not None:
            draw_detections(frame, detections)
            if rt_cfg.show_stats:
                if detections:
                    level_order = {"low": 0, "medium": 1, "high": 2}
                    top_det = max(
                        detections,
                        key=lambda d: (
                            level_order.get(d.threat_level, 0),
                            d.threat_score,
                        ),
                    )
                    frame_level = top_det.threat_level
                    frame_max_threat = top_det.threat_score
                else:
                    frame_level = "none"
                    frame_max_threat = 0.0

                draw_stats_overlay(
                    frame,
                    frame_index,
                    timestamp,
                    len(detections),
                    len(all_detections),
                    fps,
                    frame_level,
                    frame_max_threat,
                    total_frames,
                )

        if writer is not None:
            writer.write(frame)

        if rt_cfg.show_preview:
            cv2.imshow("Simple Drone Detector", frame)
            # ESC or q to quit
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    if rt_cfg.show_preview:
        cv2.destroyAllWindows()

    if not all_detections:
        print("[SDD][WARN] Brak detekcji w całym materiale.")

    print(f"[SDD] Zapisuję wyniki: {rt_cfg.output_json_path}, {rt_cfg.output_csv_path}")
    save_detections_json(all_detections, rt_cfg.output_json_path)
    save_detections_csv(all_detections, rt_cfg.output_csv_path)

    # dodatkowy zagregowany log zdarzeń per klasa (pod ML / analitykę)
    if rt_cfg.output_csv_path.lower().endswith(".csv"):
        base = rt_cfg.output_csv_path[:-4]
        events_path = f"{base}_events.csv"
    else:
        events_path = f"{rt_cfg.output_csv_path}_events.csv"

    print(f"[SDD] Zapisuję podsumowanie zdarzeń: {events_path}")
    save_events_summary(all_detections, events_path)

    print(f"[SDD] Zakończono. Łącznie klatek: {frame_index}, detekcji: {len(all_detections)}")


def main():  # entry point for setuptools console_scripts or `python -m sdd`
    run()


if __name__ == "__main__":  # entry when called as `python -m sdd.cli`
    main()
