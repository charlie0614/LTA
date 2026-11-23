import h5py
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# symptoms = ["abdomen_pain",
#             "arrhythmia",
#             "diarrhea",
#             "fatigue",
#             "headache",
#             "joint_muscle_pain",
#             "nausea",
#             "pms",
#             "shortness_of_breath",
#             "sleep_disorder",
#             "vertigo"
#             ]

symptoms = [
    "abdomen_pain",
    "arrhythmia",
    "back_pain",
    "chest_pain",
    "circulatory_problems",
    "concentration_problems",
    "diarrhea",
    "fainting_spell",
    "fatigue",
    "headache",
    "joint_muscle_pain",
    "light_sensitivity",
    "low_performance",
    "memory_impairment",
    "muscle_twitching",
    "muscle_weakness",
    "nausea",
    "pain_intercourse",
    "pms",
    "rapid_exhaustion",
    "sensory_disturbances",
    "shortness_of_breath",
    "sleep_disorder",
    "vertigo"]

symptoms = [s.replace("_", " ").title() for s in symptoms]
row_labels = symptoms
# row_labels = [f"{s} (≥ little)" for s in symptoms] + \
#     [f"{s} (strong)" for s in symptoms]


def build_state_feature_matrix(est_emission_list):
    state_vectors = []
    labels = []

    for model_id, model in enumerate(est_emission_list):
        for state_id in range(model.shape[2]):
            state_vectors.append(model[:, :, state_id])
            labels.append(f"M{model_id+3}_S{state_id}")

    X = np.array(state_vectors)
    return X, labels


def pairwise_wasserstein_distance(X):
    """
    Compute pairwise distances between state emission distributions.

    Parameters:
    X : array-like, shape (n_samples, n_categories, n_symptoms)
        Emission distributions for each state, category, and symptom.

    Returns:
    D : ndarray, shape (n_samples * (n_samples - 1) // 2,)
        Condensed pairwise distances, summed over symptoms.
    """
    from scipy.stats import wasserstein_distance

    X = np.asarray(X)
    n_samples, n_categories, n_symptoms = X.shape
    categories = np.arange(n_categories)
    D = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            for symptom in range(n_symptoms):
                D[i, j] += wasserstein_distance(
                    categories,
                    categories,
                    u_weights=X[i, :, symptom],
                    v_weights=X[j, :, symptom],
                )
            D[j, i] = D[i, j]

    return squareform(D)


def compute_hierarch_clustering(corr_pairwdist,
                                methods=['single', 'complete', 'average',
                                         'weighted', 'centroid', 'median',
                                         'ward']):
    # NOTE: if changing method, pay attention to linkage methods;
    #       'centroid', 'median', and 'ward' are correctly defined only if
    #       Euclidean pairwise metric is used in distance matrix that we used as input.

    max_coph_corr_coeff = -1
    scores = dict()
    if not isinstance(methods, list):
        methods = [methods]

    for method in methods:
        cluster_hierarch = hierarchy.linkage(corr_pairwdist, method=method)
        coph_corr_coeff, coph_dist_mat = hierarchy.cophenet(
            cluster_hierarch, corr_pairwdist)
        scores[method] = coph_corr_coeff
        if coph_corr_coeff > max_coph_corr_coeff:
            max_coph_corr_coeff = coph_corr_coeff
            max_method = method
            max_coph_dist_mat = coph_dist_mat

    cluster_hierarch = hierarchy.linkage(corr_pairwdist, method=max_method)

    print(
        "Cophenetic correlation distance for method " + max_method + ": " +
        str(max_coph_corr_coeff))

    return cluster_hierarch, max_coph_dist_mat


sim_no = "age_sex_comorbidity_covs_fups_ordinal_ordinal"
with h5py.File(f"outputs/2026_07_20/est_emission_mats_{sim_no}.h5", "r") as f:
    est_emission_mat_list = [f[f'em_states_{i+1}'][:] for i in range(2, 10)]

state_feature_matrix, state_labels = build_state_feature_matrix(
    est_emission_mat_list)

# pdist liefert die condensed pairwise distance matrix, die
# hierarchy.linkage / hierarchy.cophenet erwarten
pairwdist = pairwise_wasserstein_distance(
    state_feature_matrix)

cluster_hierarch, coph_dist_mat = compute_hierarch_clustering(
    pairwdist, methods='complete')


# skip level 0 (= 1 - p_mild - p_strong)
levels = [1, 2]
level_names = ["Mildly affected", "Strongly affected"]
level_colors = ["#E08A00", "#B71C1C"]
n_levels = len(levels)

fig = plt.figure(figsize=(12, 15))
gs = fig.add_gridspec(
    1 + n_levels, 2,
    height_ratios=[1.2] + [1.7] * n_levels,
    width_ratios=[1, 0.03],
    hspace=0.1,
    wspace=0.02
)

ax_dendro = fig.add_subplot(gs[0, 0])
ax_spacer = fig.add_subplot(gs[0, 1])
ax_spacer.axis('off')

dendro = hierarchy.dendrogram(
    cluster_hierarch,
    labels=state_labels,
    leaf_rotation=90,
    ax=ax_dendro,
    color_threshold=0
)

ax_dendro.set_xticks([])

leaf_order = dendro['leaves']
labels_ordered = [state_labels[i] for i in leaf_order]

for i, k in enumerate(levels):
    ax_heatmap = fig.add_subplot(gs[1 + i, 0])
    ax_cbar = fig.add_subplot(gs[1 + i, 1])

    X_ordered = state_feature_matrix[leaf_order][:, k, :]
    cmap = LinearSegmentedColormap.from_list("", ["white", level_colors[i]])
    im = ax_heatmap.imshow(
        X_ordered.T,
        aspect='auto',
        cmap=cmap,
        vmin=0, vmax=1
    )

    ax_heatmap.set_yticks(range(state_feature_matrix.shape[2]))
    ax_heatmap.set_yticklabels(row_labels)
    ax_heatmap.set_title(level_names[i], loc='left', fontsize=11,
                         color=level_colors[i], fontweight='bold')

    if i == n_levels - 1:
        ax_heatmap.set_xticks(range(len(labels_ordered)))
        ax_heatmap.set_xticklabels(labels_ordered, rotation=90)
    else:
        ax_heatmap.set_xticks([])

    fig.colorbar(im, cax=ax_cbar)
fig.tight_layout()

plt.savefig("outputs/2026_07_20/clustering_ordinal.png",
            bbox_inches='tight', dpi=300)
