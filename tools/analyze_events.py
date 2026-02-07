from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple analysis of SDD *_events.csv summary file."
    )
    parser.add_argument(
        "events_csv",
        nargs="?",
        default="detections_events.csv",
        help="Ścieżka do pliku *_events.csv (domyślnie detections_events.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.events_csv

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"[SDD][ERROR] Nie znaleziono pliku zdarzeń: {path}")
        return

    if not rows:
        print(f"[SDD][WARN] Plik {path} jest pusty.")
        return

    print(f"[SDD][INFO] Wczytano {len(rows)} wierszy podsumowania z {path}.")
    print()
    print("Label; count; first_ts; last_ts; min_score; max_score; mean_score")
    print("-----; -----; --------; --------; ---------; ---------; ----------")

    # Sort by count descending
    rows_sorted = sorted(
        rows,
        key=lambda r: int(r.get("count", 0)),
        reverse=True,
    )

    for r in rows_sorted:
        label = r.get("label", "?")
        count = int(r.get("count", 0))
        first_ts = float(r.get("first_timestamp", 0.0))
        last_ts = float(r.get("last_timestamp", 0.0))
        min_score = float(r.get("min_score", 0.0))
        max_score = float(r.get("max_score", 0.0))
        mean_score = float(r.get("mean_score", 0.0))

        print(
            f"{label}; {count}; {first_ts:8.3f}; {last_ts:8.3f}; "
            f"{min_score:9.3f}; {max_score:9.3f}; {mean_score:10.3f}"
        )

    print()
    print("[SDD] Analiza zakończona.")


if __name__ == "__main__":
    main()
