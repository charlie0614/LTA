import h5py
import numpy as np
import matplotlib.pyplot as plt

def _prepare_roy():
    with h5py.File("estimates/est_emissions.h5", 'r') as f:
        est_emission_mat_list = [f[f'est_emission_mat_list_{i+1}'][:] for i in range(5)]

    symptoms = [
        "Cough", "Headache", "Memory Loss", "Dyspnoea", "Myalgia",
        "Arthralgia", "Fatigue", "Ageusia", "Anosmia"
    ]
    symptom_order = [7, 8, 5, 4, 0, 3, 6, 1, 2]
    symptoms = [symptoms[i] for i in symptom_order]

    est_emission_mat_list = [np.moveaxis(h, [0, 1], [1, 0]) for h in est_emission_mat_list]

    reorder_em_list = [
        np.array([2, 3, 1, 0]),
        np.array([2, 0, 3, 1, 4]),
        np.array([4, 1, 5, 2, 0, 3]),
        np.array([1, 5, 2, 4, 0, 6, 3]),
        np.array([6, 4, 1, 0, 5, 3, 2, 7])
    ]

    est_emission_mat_list = [est_emission_mat_list[i][reorder_em_list[i], :] for i in range(len(est_emission_mat_list))]
    est_emission_mat_list = [est_emission_mat_list[i][:, np.array(symptom_order)] for i in range(len(est_emission_mat_list))]

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

    return display_mats, symptoms

def _prepare_other(sim_no):
    with h5py.File(f"estimates/est_emission_mats_{sim_no}.h5", "r") as f:
        est_emission_mat_list = [f[f'em_states_{i+1}'][:] for i in range(3, 8)]

    symptoms = [
        "Ageusia/Anosmia",
        "Arthralgia/Myalgia",
        "Cough",
        "Dyspnoea",
        "Fatigue",
        "Headache",
        "Memory Loss"
    ]

    est_emission_mat_list = [np.moveaxis(h, [0, 1], [1, 0]) for h in est_emission_mat_list]

    reorder_em_list = [
        np.array([3, 0, 2, 1]),
        np.array([1, 4, 0, 3, 2]),
        np.array([2, 1, 3, 0, 5, 4]),
        np.array([3, 5, 2, 0, 6, 4, 1]),
        np.array([3, 4, 2, 5, 0, 1, 6, 7])
    ]

    est_emission_mat_list = [est_emission_mat_list[i][reorder_em_list[i], :] for i in range(len(est_emission_mat_list))]

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

    return display_mats, symptoms

def load_and_plot_emissions_combined(name="est_emission_matrices_combined.png"):
    roy_mats, roy_symptoms = _prepare_roy()
    other_mats, other_symptoms = _prepare_other(sim_no="age_wave")

    n1 = len(roy_mats)
    n2 = len(other_mats)
    ncols = max(n1, n2)

    # common color scale across all heatmaps
    all_mats = roy_mats + other_mats
    vmin = min(np.nanmin(m.filled(np.nan)) for m in all_mats)
    vmax = max(np.nanmax(m.filled(np.nan)) for m in all_mats)

    # colormap with masked values shown as light gray
    cmap = plt.cm.get_cmap('Reds').copy()
    cmap.set_bad('lightgray')

    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8), constrained_layout=True, sharey=True)
    # ensure axes is 2D array
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    # plot first row (Roy)
    for i in range(ncols):
        ax = axes[0, i]
        if i < n1:
            heatmap = roy_mats[i]
            im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_yticks(np.arange(heatmap.shape[0]), labels=["State " + str(j) for j in range(1, heatmap.shape[0] + 1)])
            ax.set_xticks(np.arange(heatmap.shape[1]), labels=roy_symptoms, rotation=45, ha='right')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(f'{i+4} States')
        else:
            ax.axis('off')
    axes[0, 0].set_ylabel('Roy', fontweight='bold', fontsize='x-large')

    # plot second row (other)
    for i in range(ncols):
        ax = axes[1, i]
        if i < n2:
            heatmap = other_mats[i]
            im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_yticks(np.arange(heatmap.shape[0]), labels=["State " + str(j) for j in range(1, heatmap.shape[0] + 1)])
            ax.set_xticks(np.arange(heatmap.shape[1]), labels=other_symptoms, rotation=45, ha='right')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(f'{i+4} States')
        else:
            ax.axis('off')
    axes[1, 0].set_ylabel('Mine', fontweight='bold', fontsize='x-large')

    # single colorbar for all
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.8)
    fig.savefig("est_emission_matrices_combined.png", bbox_inches='tight', dpi=300)

def load_and_plot_emissions_compare_covs(name="est_emission_matrices_with_age_wave.png"):
    age_mats, other_symptoms = _prepare_other(sim_no="age")
    age_wave_mats, _ = _prepare_other(sim_no="age_wave")

    n1 = len(age_wave_mats)
    n2 = len(age_mats)
    ncols = max(n1, n2)

    # common color scale across all heatmaps
    all_mats = age_wave_mats + age_mats
    vmin = min(np.nanmin(m.filled(np.nan)) for m in all_mats)
    vmax = max(np.nanmax(m.filled(np.nan)) for m in all_mats)

    # colormap with masked values shown as light gray
    cmap = plt.cm.get_cmap('Reds').copy()
    cmap.set_bad('lightgray')

    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8), constrained_layout=True, sharey=True)
    # ensure axes is 2D array
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    # plot first row (Roy)
    for i in range(ncols):
        ax = axes[0, i]
        if i < n1:
            heatmap = age_mats[i]
            im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_yticks(np.arange(heatmap.shape[0]), labels=["State " + str(j) for j in range(1, heatmap.shape[0] + 1)])
            ax.set_xticks(np.arange(heatmap.shape[1]), labels=other_symptoms, rotation=45, ha='right')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(f'{i+4} States')
        else:
            ax.axis('off')
    axes[0, 0].set_ylabel('Age', fontweight='bold', fontsize='x-large')

    # plot second row (other)
    for i in range(ncols):
        ax = axes[1, i]
        if i < n2:
            heatmap = age_wave_mats[i]
            im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_yticks(np.arange(heatmap.shape[0]), labels=["State " + str(j) for j in range(1, heatmap.shape[0] + 1)])
            ax.set_xticks(np.arange(heatmap.shape[1]), labels=other_symptoms, rotation=45, ha='right')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(f'{i+4} States')
        else:
            ax.axis('off')
    axes[1, 0].set_ylabel('Age and Covid Wave', fontweight='bold', fontsize='x-large')

    # single colorbar for all
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.8)
    fig.savefig(name, bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    # load_and_plot_emissions_combined()
    load_and_plot_emissions_compare_covs()