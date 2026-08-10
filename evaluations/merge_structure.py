 
import os
 
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.cluster import hierarchy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

 
def match_adjacent_models(mat_small, mat_large, method):
    n_small = mat_small.shape[1]
    n_large = mat_large.shape[1]
 
    X_small = mat_small.T   # (n_small, n_features)
    X_large = mat_large.T  # (n_large, n_features)
 
    # Distanzmatrix zwischen ALLEN large-States und ALLEN small-States,
    # einmal berechnet und für alle Kandidaten-Paare wiederverwendet
    dist_large_to_small = cdist(X_large, X_small)  # (n_large, n_small)
 
    all_costs = []
    all_mappings_by_pair = []
 
    for i in range(n_large):
        for j in range(i + 1, n_large):
            remaining = [s for s in range(n_large) if s not in (i, j)]
 
            split_row = method([dist_large_to_small[i], dist_large_to_small[j]], axis=0)
 
            cost_rows = [dist_large_to_small[s] for s in remaining] + [split_row]
            cost_matrix = np.vstack(cost_rows)  # (n_small, n_small)
            rep_keys = remaining + ["split"]
 
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_ind, col_ind].sum()
 
            mapping = {}
            for r, c in zip(row_ind, col_ind):
                key = rep_keys[r]
                small_state = int(c)
                if key == "split":
                    mapping[small_state] = [i, j]
                else:
                    mapping[small_state] = [int(key)]
 
            all_costs.append(total_cost)
            all_mappings_by_pair.append(mapping)
 
    order = np.argsort(all_costs)
    best_idx = order[0]
 
    best_mapping = all_mappings_by_pair[best_idx]
 
    return best_mapping


def build_merge_structure(est_emission_list, method=np.sum):
    est_emission_list = sorted(est_emission_list, key=lambda m: m.shape[1])
 
 
    all_mappings = []
    for k in range(len(est_emission_list) - 1):
        mat_small = est_emission_list[k]
        mat_large = est_emission_list[k + 1]
        n_small = mat_small.shape[1]
        n_large = mat_large.shape[1]
 
        mapping = match_adjacent_models(mat_small, mat_large, method=method)

        all_mappings.append({
            "n_small": n_small, "n_large": n_large, "mapping": mapping
        })
 
    return all_mappings
 
 
def build_lineage(est_emission_list, all_mappings):
    est_sorted = sorted(est_emission_list, key=lambda m: m.shape[1])
    n_small0 = est_sorted[0].shape[1]
 
    lineage = [dict()] * len(est_sorted)
    lineage[0] = {s: s for s in range(n_small0)}
 
    for i, entry in enumerate(all_mappings):
        mapping = entry["mapping"]
        child_lineage = {}
        for s, children in mapping.items():
            origin = lineage[i][s]
            for c in children:
                child_lineage[c] = origin
        lineage[i + 1] = child_lineage
 
    return lineage
 
 
def compute_layout(est_emission_list, all_mappings):
    est_sorted = sorted(est_emission_list, key=lambda m: m.shape[1])
    n_models = len(est_sorted)
 
    positions = [None] * n_models
    positions[0] = {s: s for s in range(est_sorted[0].shape[1])}
 
    for i, entry in enumerate(all_mappings):
        mapping = entry["mapping"]
        small_positions = positions[i]
        ordered_small = sorted(small_positions.keys(), key=lambda s: small_positions[s])
 
        large_positions = {}
        y = 0
        for s in ordered_small:
            children = mapping.get(s, [])
            for c in children:
                large_positions[c] = y
                y += 1
        positions[i + 1] = large_positions
 
    return positions
 
 
def plot_merge_structure(est_emission_list, all_mappings, filename="merge_structure.png",
                          node_w=0.62, node_h=0.34, y_gap=0.56, figsize=(14, 7.5)):
    est_sorted = sorted(est_emission_list, key=lambda m: m.shape[1])
    n_states_per_model = [m.shape[1] for m in est_sorted]
    positions = compute_layout(est_emission_list, all_mappings)
    positions = [{s: y * y_gap for s, y in pos.items()} for pos in positions]
 
    all_values = np.concatenate([m.flatten() for m in est_sorted])
    vmin, vmax = all_values.min(), all_values.max()
    heatmap_cmap = plt.get_cmap("Reds")
 
    max_y = max(v for pos in positions for v in pos.values())
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
 
    n_small0 = n_states_per_model[0]
    lineage_cmap = plt.get_cmap("tab10", n_small0)
    lineage = build_lineage(est_emission_list, all_mappings)
 
    for i, entry in enumerate(all_mappings):
        mapping = entry["mapping"]
        for s, children in mapping.items():
            y_small = positions[i][s]
            is_split = len(children) == 2
            for c in children:
                y_large = positions[i + 1][c]
                color = lineage_cmap(lineage[i][s])
                ax.plot([i + node_w / 2, i + 1 - node_w / 2], [y_small, y_large],
                        color=color, linewidth=2.2 if is_split else 1.4,
                        alpha=0.85, zorder=2,
                        linestyle="--" if is_split else "-")
 
    for i, model in enumerate(est_sorted):
        pos_dict = positions[i]
        for state_idx, y in pos_dict.items():
            # Gedreht: Werte als Zeilenvektor (1, n_features) statt Spalte
            # (n_features, 1) -> die Emissionswerte laufen horizontal im
            # Knoten statt vertikal gestapelt zu sein.
            values = model[:, state_idx].reshape(1, -1)
            extent = [i - node_w / 2, i + node_w / 2, y - node_h / 2, y + node_h / 2]
            ax.imshow(values, extent=extent, cmap=heatmap_cmap,
                      vmin=vmin, vmax=vmax, aspect="auto", zorder=3, origin="upper")
            border_color = lineage_cmap(lineage[i][state_idx])
            rect = mpatches.Rectangle((extent[0], extent[2]), node_w, node_h,
                                       fill=False, edgecolor=border_color, linewidth=1.8, zorder=4)
            ax.add_patch(rect)
            ax.text(i, y - node_h / 2 - 0.09, f"S{state_idx}",
                    ha="center", va="top", fontsize=7, color="#374151", zorder=4)
 
    ax.set_xticks(range(len(est_sorted)))
    ax.set_xticklabels([f"{k} States" for k in n_states_per_model], fontsize=10)
    ax.set_xlim(-0.5, len(est_sorted) - 0.5)
    ax.set_ylim(max_y + node_h, -node_h)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.tick_params(axis="x", length=0)
 
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=1.4, linestyle="-", label="1:1 Match"),
        Line2D([0], [0], color="gray", linewidth=2.2, linestyle="--", label="Split (2 States)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False)
 
    sm = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.015)
    cbar.set_label("Emission probability", fontsize=9)
 
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

 
if __name__ == "__main__":
    sim_no = "age_sex_comorbidity_covs_fups"
    sim_dir = "outputs/2026_07_20/"
 
    with h5py.File(f"{sim_dir}/est_emission_mats_{sim_no}.h5", "r") as f:
        est_emission_mat_list = [f[f'em_states_{i+1}'][:] for i in range(2, 9)]

    merge_structure_sum = build_merge_structure(est_emission_mat_list, method=np.sum)
    merge_structure_mean = build_merge_structure(est_emission_mat_list, method=np.mean)
 
    plot_merge_structure(est_emission_mat_list, merge_structure_sum, filename=f"{sim_dir}/merge_structure_sum.png")
    plot_merge_structure(est_emission_mat_list, merge_structure_mean, filename=f"{sim_dir}/merge_structure_mean.png")

