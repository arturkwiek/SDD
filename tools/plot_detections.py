from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize detection results from detections.csv"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="detections.csv",
        help="Ścieżka do pliku detections.csv (domyślnie detections.csv)",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=1.0,
        help="Rozmiar binu czasowego w sekundach dla wykresu liczby detekcji w czasie (domyślnie 1s)",
    )
    return parser.parse_args()


def load_detections(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_counts_per_label(detections: List[Dict[str, str]]) -> Counter:
    labels = [d["label"] for d in detections]
    return Counter(labels)


def plot_counts_per_label(counts: Counter) -> None:
    if not counts:
        print("[SDD][WARN] Brak detekcji do wizualizacji (counts per label).")
        return

    labels_sorted, values = zip(*sorted(counts.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(8, 4))
    plt.bar(labels_sorted, values)
    plt.xlabel("Label")
    plt.ylabel("Liczba detekcji")
    plt.title("Liczba detekcji na klasę")
    plt.xticks(rotation=90)
    plt.tight_layout()


def plot_counts_over_time(detections: List[Dict[str, str]], bin_size: float) -> None:
    if not detections:
        print("[SDD][WARN] Brak detekcji do wizualizacji (counts over time).")
        return

    # binujemy po timestamp
    bins: Dict[int, int] = defaultdict(int)
    for d in detections:
        ts = float(d["timestamp"])
        bin_idx = int(ts // bin_size)
        bins[bin_idx] += 1

    xs = sorted(bins.keys())
    ys = [bins[i] for i in xs]
    times = [x * bin_size for x in xs]

    plt.figure(figsize=(8, 4))
    plt.plot(times, ys, marker="o")
    plt.xlabel("Czas [s]")
    plt.ylabel("Liczba detekcji w binie")
    plt.title(f"Liczba detekcji w czasie (bin={bin_size}s)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()


def main() -> None:
    args = parse_args()
    path = args.csv_path

    try:
        detections = load_detections(path)
    except FileNotFoundError:
        print(f"[SDD][ERROR] Nie znaleziono pliku: {path}")
        return

    if not detections:
        print(f"[SDD][WARN] Plik {path} jest pusty.")
        return

    print(f"[SDD][INFO] Wczytano {len(detections)} detekcji z {path}.")

    counts = compute_counts_per_label(detections)

    # zapis czytelnej listy klas do pliku tekstowego
    if path.lower().endswith(".csv"):
        base = path[:-4]
        txt_path = f"{base}_label_counts.txt"
    else:
        txt_path = f"{path}_label_counts.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("label;count\n")
        for label, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{label};{cnt}\n")

    print(f"[SDD][INFO] Zapisano listę klas do: {txt_path}")

    plot_counts_per_label(counts)
    plot_counts_over_time(detections, args.bin_size)

    print("[SDD] Wyświetlam wykresy (zamknij okna, aby zakończyć).")
    plt.show()


if __name__ == "__main__":
    main()
