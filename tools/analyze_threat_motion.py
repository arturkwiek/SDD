from __future__ import annotations

import argparse
import csv
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine threat metrics from *_events.csv with motion metrics "
            "estimated from detections.csv to rank moving threats."
        )
    )
    parser.add_argument(
        "events_csv",
        nargs="?",
        default="detections_events.csv",
        help="Ścieżka do pliku *_events.csv (domyślnie detections_events.csv)",
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
            "klasy przy estymacji ruchu (domyślnie 1s)."
        ),
    )
    return parser.parse_args()


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def estimate_frame_size(detections: List[Dict[str, str]]) -> tuple[float, float]:
    if not detections:
        return 1.0, 1.0

    min_x = min(float(d["x1"]) for d in detections)
    max_x = max(float(d["x2"]) for d in detections)
    min_y = min(float(d["y1"]) for d in detections)
    max_y = max(float(d["y2"]) for d in detections)

    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    return width, height


def compute_motion_per_label(
    detections: List[Dict[str, str]], max_dt: float
) -> Dict[str, Dict[str, float]]:
    import math
    from collections import defaultdict

    if not detections:
        return {}

    width, height = estimate_frame_size(detections)

    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for d in detections:
        by_label[d.get("label", "?")].append(d)

    stats: Dict[str, Dict[str, float]] = {}

    for label, rows in by_label.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                float(r.get("timestamp", 0.0)),
                int(r.get("frame_index", 0)),
            ),
        )

        pair_count = 0
        sum_speed_norm = 0.0
        max_speed_norm = 0.0

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

            dx = (cx1 - cx0) / width
            dy = (cy1 - cy0) / height
            dist_norm = math.hypot(dx, dy)

            speed_norm = dist_norm / dt

            pair_count += 1
            sum_speed_norm += speed_norm
            if speed_norm > max_speed_norm:
                max_speed_norm = speed_norm

        if pair_count == 0:
            continue

        stats[label] = {
            "pairs": float(pair_count),
            "mean_speed_norm": sum_speed_norm / pair_count,
            "max_speed_norm": max_speed_norm,
        }

    return stats


def main() -> None:
    args = parse_args()

    # Wczytanie zagregowanych metryk zagrożenia
    try:
        events_rows = load_csv(args.events_csv)
    except FileNotFoundError:
        print(f"[SDD][ERROR] Nie znaleziono pliku zdarzeń: {args.events_csv}")
        return

    if not events_rows:
        print(f"[SDD][WARN] Plik {args.events_csv} jest pusty.")
        return

    # Wczytanie detekcji do analizy ruchu
    try:
        detections = load_csv(args.detections_csv)
    except FileNotFoundError:
        print(f"[SDD][ERROR] Nie znaleziono pliku detekcji: {args.detections_csv}")
        return

    if not detections:
        print(f"[SDD][WARN] Plik {args.detections_csv} jest pusty.")
        return

    print(
        f"[SDD][INFO] Wczytano {len(events_rows)} wierszy z {args.events_csv} "
        f"i {len(detections)} detekcji z {args.detections_csv}."
    )

    motion_stats = compute_motion_per_label(detections, max_dt=args.max_dt)

    # Mapujemy etykiety -> threat metrics
    threat_by_label: Dict[str, Dict[str, float | str]] = {}
    for r in events_rows:
        label = r.get("label", "?")
        try:
            mean_threat = float(r.get("mean_threat_score", 0.0))
        except ValueError:
            mean_threat = 0.0
        dom_level = r.get("dominant_threat_level", "?")
        try:
            count = int(r.get("count", 0))
        except ValueError:
            count = 0

        threat_by_label[label] = {
            "mean_threat": mean_threat,
            "dominant_level": dom_level,
            "count": count,
        }

    combined: List[tuple[str, Dict[str, float | str]]] = []

    for label, m in motion_stats.items():
        t = threat_by_label.get(label)
        if not t:
            continue

        mean_threat = float(t["mean_threat"])
        mean_speed_norm = float(m["mean_speed_norm"])

        # prosty wskaźnik "ruchomego zagrożenia"
        moving_threat_index = mean_threat * mean_speed_norm

        combined.append(
            (
                label,
                {
                    "mean_threat": mean_threat,
                    "dominant_level": str(t["dominant_level"]),
                    "count": float(t["count"]),
                    "pairs": float(m["pairs"]),
                    "mean_speed_norm": mean_speed_norm,
                    "max_speed_norm": float(m["max_speed_norm"]),
                    "moving_threat_index": moving_threat_index,
                },
            )
        )

    if not combined:
        print("[SDD][WARN] Brak klas z jednoczesnymi metrykami zagrożenia i ruchu.")
        return

    combined_sorted = sorted(
        combined, key=lambda kv: kv[1]["moving_threat_index"], reverse=True
    )

    print()
    print(
        "Label; count; dom_level; mean_threat; pairs; "
        "mean_speed_norm; max_speed_norm; moving_threat_index"
    )
    print(
        "-----; -----; ---------; -----------; -----; "
        "--------------; -------------; -------------------"
    )

    for label, s in combined_sorted:
        print(
            f"{label}; {int(s['count'])}; {s['dominant_level']}; "
            f"{s['mean_threat']:11.3f}; {int(s['pairs'])}; "
            f"{s['mean_speed_norm']:14.5f}; {s['max_speed_norm']:11.5f}; "
            f"{s['moving_threat_index']:19.5f}"
        )

    print()
    print("[SDD] Analiza ruchomych zagrożeń zakończona.")


if __name__ == "__main__":
    main()
