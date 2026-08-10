import h5py
import numpy as np
import matplotlib.pyplot as plt

symptoms = ["abdomen_pain", 
    "arrhythmia",
    "diarrhea",
    "fatigue",
    "headache",
    "joint_muscle_pain",
    "nausea",
    "pms",
    "shortness_of_breath",
    "sleep_disorder",
    "vertigo"
]
symptoms = [symptom.replace("_", " ").title() for symptom in symptoms]

def _prepare_other(sim_no):
    with h5py.File(f"outputs/2026_07_20/est_emission_mats_{sim_no}.h5", "r") as f:
        est_emission_mat_list = [f[f'em_states_{i+1}'][:] for i in range(2, 8)]

    est_emission_mat_list = [np.moveaxis(h, [0, 1], [1, 0]) for h in est_emission_mat_list]

    # pad rows within this group and mask NaNs
    max_rows = max(h.shape[0] for h in est_emission_mat_list)
    display_mats = []
    for h in est_emission_mat_list:
        if h.shape[0] < max_rows:
            padded = np.full((max_rows, h.shape[1]), np.nan)
            padded[: h.shape[0], :] = h
            display_mats.append(np.ma.masked_invalid(padded))
        else:
            display_mats.append(np.ma.masked_invalid(h))

    return display_mats

def load_and_plot_emissions_combined(name="est_emission_matrices_combined.png"):
    other_mats = _prepare_other(sim_no="age_sex_comorbidity_covs_fups")

    n2 = len(other_mats)
    ncols = n2

    # common color scale across all heatmaps
    all_mats = other_mats
    vmin = min(np.nanmin(m.filled(np.nan)) for m in all_mats)
    vmax = max(np.nanmax(m.filled(np.nan)) for m in all_mats)

    # colormap with masked values shown as light gray
    cmap = plt.cm.get_cmap('Reds').copy()
    cmap.set_bad('lightgray')

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True, sharey=True)

    # plot second row (other)
    for i in range(ncols):
        ax = axes[i]
        if i < n2:
            heatmap = other_mats[i]
            im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_yticks(np.arange(heatmap.shape[0]), labels=["State " + str(j) for j in range(1, heatmap.shape[0] + 1)])
            ax.set_xticks(np.arange(heatmap.shape[1]), labels=symptoms, rotation=45, ha='right')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(f'{i+3} States')
        else:
            ax.axis('off')

    # single colorbar for all
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.8)
    fig.savefig("outputs/2026_07_20/est_emission_matrices_combined.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    load_and_plot_emissions_combined()