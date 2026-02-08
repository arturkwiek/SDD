from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple motion analysis based on detections.csv. "
            "Estimates per-class motion speed using consecutive detections."
        )
    )
    parser.add_argument(
        "detections_csv",
        nargs="?",
        default="detections.csv",
        help="Ścieżka do pliku detections.csv (domyślnie detections.csv)",
    )
    parser.add_argument(
        "--max-dt",
        type=float,
        default=1.0,
        help=(
            "Maksymalny odstęp czasu [s] między kolejnymi detekcjami tej samej "
            "klasy, aby liczyć je jako potencjalnie ten sam obiekt (domyślnie 1s)."
        ),
    )
    return parser.parse_args()


def load_detections(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def estimate_frame_size(detections: List[Dict[str, str]]) -> Tuple[float, float]:
    if not detections:
        return 1.0, 1.0

    min_x = min(float(d["x1"]) for d in detections)
    max_x = max(float(d["x2"]) for d in detections)
    min_y = min(float(d["y1"]) for d in detections)
    max_y = max(float(d["y2"]) for d in detections)

    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    return width, height


def analyze_motion(
    detections: List[Dict[str, str]], max_dt: float
) -> Dict[str, Dict[str, float]]:
    """Szacuje podstawowe metryki ruchu dla każdej klasy.

    Uproszczenie: nie mamy śledzenia ID, więc zakładamy, że kolejne
    detekcje tej samej klasy w czasie reprezentują przybliżony ruch
    (może łączyć różne obiekty, ale daje obraz "dynamiki" klasy).
    """

    if not detections:
        return {}

    width, height = estimate_frame_size(detections)

    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for d in detections:
        by_label[d.get("label", "?")].append(d)

    stats: Dict[str, Dict[str, float]] = {}

    for label, rows in by_label.items():
        # sortujemy po czasie, a w razie remisu po indeksie klatki
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                float(r.get("timestamp", 0.0)),
                int(r.get("frame_index", 0)),
            ),
        )

        pair_count = 0
        sum_speed_px = 0.0
        max_speed_px = 0.0
        sum_speed_norm = 0.0

        for prev, curr in zip(rows_sorted, rows_sorted[1:]):
            try:
                t0 = float(prev.get("timestamp", 0.0))
                t1 = float(curr.get("timestamp", 0.0))
            except ValueError:
                continue

            dt = t1 - t0
            if dt <= 0 or dt > max_dt:
                continue

            try:
                x1_0 = float(prev["x1"])  # type: ignore[index]
                y1_0 = float(prev["y1"])  # type: ignore[index]
                x2_0 = float(prev["x2"])  # type: ignore[index]
                y2_0 = float(prev["y2"])  # type: ignore[index]

                x1_1 = float(curr["x1"])  # type: ignore[index]
                y1_1 = float(curr["y1"])  # type: ignore[index]
                x2_1 = float(curr["x2"])  # type: ignore[index]
                y2_1 = float(curr["y2"])  # type: ignore[index]
            except (KeyError, ValueError):
                continue

            cx0 = 0.5 * (x1_0 + x2_0)
            cy0 = 0.5 * (y1_0 + y2_0)
            cx1 = 0.5 * (x1_1 + x2_1)
            cy1 = 0.5 * (y1_1 + y2_1)

            dx = cx1 - cx0
            dy = cy1 - cy0
            dist_px = math.hypot(dx, dy)

            # znormalizowany ruch względem rozmiaru kadru
            dx_norm = dx / width
            dy_norm = dy / height
            dist_norm = math.hypot(dx_norm, dy_norm)

            speed_px = dist_px / dt
            speed_norm = dist_norm / dt

            pair_count += 1
            sum_speed_px += speed_px
            sum_speed_norm += speed_norm
            if speed_px > max_speed_px:
                max_speed_px = speed_px

        if pair_count == 0:
            continue

        stats[label] = {
            "pairs": float(pair_count),
            "mean_speed_px": sum_speed_px / pair_count,
            "max_speed_px": max_speed_px,
            "mean_speed_norm": sum_speed_norm / pair_count,
        }

    return stats


def main() -> None:
    args = parse_args()
    path = args.detections_csv

    try:
        detections = load_detections(path)
    except FileNotFoundError:
        print(f"[SDD][ERROR] Nie znaleziono pliku: {path}")
        return

    if not detections:
        print(f"[SDD][WARN] Plik {path} jest pusty.")
        return

    print(f"[SDD][INFO] Wczytano {len(detections)} detekcji z {path}.")

    motion_stats = analyze_motion(detections, max_dt=args.max_dt)
    if not motion_stats:
        print("[SDD][WARN] Brak par detekcji spełniających kryteria do analizy ruchu.")
        return

    # sortujemy klasy wg średniej prędkości (malejąco)
    sorted_items = sorted(
        motion_stats.items(), key=lambda kv: kv[1]["mean_speed_px"], reverse=True
    )

    print()
    print(
        "Label; pairs; mean_speed_px; max_speed_px; "
        "mean_speed_norm [1/s]"
    )
    print("-----; -----; -------------; ------------; -------------------")

    for label, s in sorted_items:
        print(
            f"{label}; {int(s['pairs'])}; "
            f"{s['mean_speed_px']:11.3f}; {s['max_speed_px']:10.3f}; "
            f"{s['mean_speed_norm']:19.5f}"
        )

    print()
    print("[SDD] Analiza ruchu zakończona.")


if __name__ == "__main__":
    main()
