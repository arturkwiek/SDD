from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from sdd.threat_utils import classify_label_safety


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize aggregated threat metrics from *_events.csv grouped "
            "by dominant threat level (low/medium/high)."
        )
    )
    parser.add_argument(
        "events_csv",
        nargs="?",
        default="detections_events.csv",
        help="Ścieżka do pliku *_events.csv (domyślnie detections_events.csv)",
    )
    parser.add_argument(
        "--metric",
        choices=["mean_threat", "max_threat", "count"],
        default="mean_threat",
        help=(
            "Metryka do sortowania w ramach grupy: "
            "mean_threat (domyślnie), max_threat lub count."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Maksymalna liczba etykiet na wykres w każdej grupie (domyślnie 10)",
    )
    return parser.parse_args()


def load_events(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_by_threat_level(
    rows: List[Dict[str, str]], metric: str
) -> Dict[str, List[Tuple[str, float, int]]]:
    """Zwraca słownik: level -> lista (label, metric_value, count).

    metric in {"mean_threat", "max_threat", "count"}.
    """

    groups: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)

    for r in rows:
        level = (r.get("dominant_threat_level") or "?").lower()
        label = r.get("label", "?")
        try:
            count = int(r.get("count", 0))
        except ValueError:
            count = 0

        try:
            mean_threat = float(r.get("mean_threat_score", 0.0))
        except ValueError:
            mean_threat = 0.0

        try:
            max_threat = float(r.get("max_threat_score", 0.0))
        except ValueError:
            max_threat = 0.0

        if metric == "mean_threat":
            value = mean_threat
        elif metric == "max_threat":
            value = max_threat
        else:  # "count"
            value = float(count)

        groups[level].append((label, value, count))

    return groups
def plot_groups(
    groups: Dict[str, List[Tuple[str, float, int]]], metric: str, top_n: int
) -> None:
    # Interesują nas głównie te trzy poziomy w ustalonej kolejności.
    levels_order = ["high", "medium", "low"]
    present_levels = [lvl for lvl in levels_order if lvl in groups and groups[lvl]]

    if not present_levels:
        print("[SDD][WARN] Brak danych do wizualizacji w grupach zagrożeń.")
        return

    n_rows = len(present_levels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    metric_label = {
        "mean_threat": "Średni threat_score",
        "max_threat": "Maksymalny threat_score",
        "count": "Liczba detekcji",
    }[metric]

    for ax, level in zip(axes, present_levels):
        data = groups[level]
        # sortujemy malejąco po wartości metryki
        data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
        if top_n > 0:
            data_sorted = data_sorted[:top_n]

        if not data_sorted:
            continue

        labels, values, counts = zip(*data_sorted)

        # Kolorujemy słupki według kategorii bezpieczeństwa etykiety.
        colors = []
        for lbl in labels:
            safety = classify_label_safety(lbl)
            if safety == "danger":
                colors.append("red")
            elif safety == "medium":
                colors.append("gold")
            else:  # "safe"
                colors.append("green")

        ax.bar(labels, values, color=colors)
        ax.set_ylabel(metric_label)
        title_level = level.upper()
        ax.set_title(f"Dominujący poziom zagrożenia: {title_level}")
        # wartości count mogą być pomocne jako tekst nad słupkami
        for i, (x, v, c) in enumerate(zip(labels, values, counts)):
            ax.text(i, v, str(c), ha="center", va="bottom", fontsize=8)

        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=45)

    # Prosta legenda kolorów (wspólna dla wszystkich subplotów)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color="red", label="potencjalne zagrożenie (airborne / krytyczne)"),
        Patch(color="gold", label="pośrednie (pojazdy / infrastruktura)"),
        Patch(color="green", label="raczej bezpieczne / tło"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=8,
    )

    plt.tight_layout()


def main() -> None:
    args = parse_args()
    path = args.events_csv

    try:
        rows = load_events(path)
    except FileNotFoundError:
        print(f"[SDD][ERROR] Nie znaleziono pliku zdarzeń: {path}")
        return

    if not rows:
        print(f"[SDD][WARN] Plik {path} jest pusty.")
        return

    print(f"[SDD][INFO] Wczytano {len(rows)} wierszy podsumowania z {path}.")

    groups = group_by_threat_level(rows, args.metric)

    # Krótka tekstowa zajawka przed wykresem
    for level in ("high", "medium", "low"):
        items = groups.get(level) or []
        total_labels = len(items)
        total_count = sum(c for _, _, c in items)
        print(f"[SDD][INFO] Poziom {level.upper()}: {total_labels} klas, {total_count} detekcji")

    plot_groups(groups, args.metric, args.top_n)

    print("[SDD] Wyświetlam wykres(y) grup zagrożeń (zamknij okna, aby zakończyć).")
    plt.show()


if __name__ == "__main__":
    main()
