import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import re
import math

# ── 0. Config ─────────────────────────────────────────────────────────────────
FILES = {
    "Old preprocessing": "data/DigiHero/2025-12-16_data_symptoms_DigiHero_Bonn_preprocessed.csv",
    "New preprocessing": "data/DigiHero/2026-07-01_data_symptoms_DigiHero_Bonn_preprocessed_combined.csv"
}

# TIMEPOINTS       = ["acute", "4weeks", "12weeks"]
# TIMEPOINT_LABELS = ["0w",      "4w",      "12w"]
TIMEPOINTS       = ["3months", "6months", "12months", "18months", "24months", "30months"]
TIMEPOINT_LABELS = ["3m",      "6m",      "12m",      "18m",      "24m",      "30m"]
TP_ORDER         = {tp: i for i, tp in enumerate(TIMEPOINTS)}

COLORS   = ["#2563EB", "#059669", "#DC2626"]
GRID_COL = "#E5E7EB"
BG       = "#F9FAFB"

# pattern = re.compile(r"^ih_(\w+?)_(.+)$")
pattern = re.compile(r"^pcc_(\w+?)_(.+)$")


# ── 1. Parse & count for a single file, all value types ──────────────────────
def load_counts(path, value_types=(1,)):
    """Zählt für jede (timepoint, symptom)-Kombination, wie oft jeder
    gewünschte Wert (z.B. 1, 0, -1) vorkommt.
    """
    df = pd.read_csv(path)
    parsed = []
    for col in df.columns:
        m = pattern.match(col)
        if m:
            tp, symptom = m.group(1), m.group(2)
            if tp in TP_ORDER:
                parsed.append({"col": col, "timepoint": tp, "symptom": symptom})

    records = []
    for row in parsed:
        for val in value_types:
            count = (df[row["col"]] == val).sum()
            records.append({
                "symptom": row["symptom"],
                "timepoint": row["timepoint"],
                "tp_order": TP_ORDER[row["timepoint"]],
                "value": val,
                "count": count
            })

    return pd.DataFrame(records).sort_values("tp_order")


# ── 2. Generische Plot-Funktion ───────────────────────────────────────────────
def plot_symptom_comparison(series_dict, title, filename, ncols=3):
    """Erzeugt ein Grid von Subplots (einer pro Symptom), das für jede
    Timepoint-Gruppe die Balken aller Einträge in series_dict vergleicht.

    @param series_dict Dict {label: DataFrame}, wobei jedes DataFrame die
        Spalten 'symptom', 'tp_order', 'count' enthält (Output von load_counts,
        ggf. schon nach 'value' gefiltert).
    @param title Titel des gesamten Plots.
    @param filename Dateiname zum Speichern.
    """
    symptoms = sorted(set().union(*[set(df["symptom"]) for df in series_dict.values()]))

    n     = len(symptoms)
    nrows = math.ceil(n / ncols)

    n_series  = len(series_dict)
    bar_width = 0.8 / n_series
    x_base    = np.arange(len(TIMEPOINTS))
    offsets   = (np.arange(n_series) - (n_series - 1) / 2) * bar_width

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 3.8 * nrows),
                             facecolor="white")
    axes = axes.flatten()

    for i, symptom in enumerate(symptoms):
        ax = axes[i]
        ax.set_facecolor(BG)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COL)
        ax.yaxis.grid(True, color=GRID_COL, linewidth=1, zorder=0)
        ax.set_axisbelow(True)

        for j, (label, df) in enumerate(series_dict.items()):
            sub = df[df["symptom"] == symptom].set_index("tp_order")
            y = [sub.loc[tp_idx, "count"] if tp_idx in sub.index else 0
                 for tp_idx in range(len(TIMEPOINTS))]

            ax.bar(x_base + offsets[j], y, width=bar_width,
                   color=COLORS[j], label=label, zorder=3,
                   alpha=0.88, linewidth=0)

        ax.set_xticks(x_base)
        ax.set_xticklabels(TIMEPOINT_LABELS, fontsize=9)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.tick_params(axis="y", labelsize=9, length=0)
        ax.tick_params(axis="x", length=3, color=GRID_COL)

        symptom_title = symptom.replace("_", " ").title()
        ax.set_title(symptom_title, fontsize=11, fontweight="bold", pad=8, color="#111827")
        ax.set_ylabel("Count", fontsize=8, color="#6B7280")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[j]) for j in range(n_series)]
    fig.legend(handles, list(series_dict.keys()),
               loc="lower center", ncol=n_series,
               fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01, color="#111827")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


# ── 3a. Vergleich: Old vs. New preprocessing (nur Symptom == 1) ─────────────
def plot_old_vs_new():
    all_counts = {label: load_counts(path, value_types=(1,))
                  for label, path in FILES.items()}
    plot_symptom_comparison(
        all_counts,
        title="Symptom Cluster Frequency Over Time",
        filename="symptoms_old_vs_new.png"
    )


# ── 3b. Vergleich: Unknown (-1) vs. No symptom (0), pro File separat ─────────
def plot_unknown_vs_zero(file_label):
    """Vergleicht -1 (unknown) vs. 0 (kein Symptom) innerhalb EINES Files."""
    path = FILES[file_label]
    counts = load_counts(path, value_types=(-1, 0))

    series_dict = {
        "Unknown (-1)": counts[counts["value"] == -1],
        "No symptom (0)": counts[counts["value"] == 0],
    }

    plot_symptom_comparison(
        series_dict,
        title=f"Unknown vs. No-Symptom Entries Over Time ({file_label})",
        filename=f"symptoms_unknown_vs_zero_{file_label.replace(' ', '_')}.png"
    )


# ── Aufrufe ───────────────────────────────────────────────────────────────────
# plot_old_vs_new()
# plot_unknown_vs_zero("Old preprocessing")
plot_unknown_vs_zero("New preprocessing")