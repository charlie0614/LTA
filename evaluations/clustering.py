import h5py
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
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

def build_state_feature_matrix(est_emission_list):
    state_vectors = []
    labels = []

    for model_id, model in enumerate(est_emission_list):
        for state_id in range(model.shape[1]):
            state_vectors.append(model[:, state_id])
            labels.append(f"M{model_id+3}_S{state_id}")

    X = np.array(state_vectors)
    return X, labels


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


sim_no = "age_sex_comorbidity_covs_fups"
with h5py.File(f"outputs/2026_07_20/est_emission_mats_{sim_no}.h5", "r") as f:
    est_emission_mat_list = [f[f'em_states_{i+1}'][:] for i in range(2, 8)]

state_feature_matrix, state_labels = build_state_feature_matrix(est_emission_mat_list)

# pdist liefert die condensed pairwise distance matrix, die
# hierarchy.linkage / hierarchy.cophenet erwarten
pairwdist = pdist(state_feature_matrix, metric='euclidean')

cluster_hierarch, coph_dist_mat = compute_hierarch_clustering(pairwdist)

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[3, 1],
    width_ratios=[1, 0.03],
    hspace=0.05,
    wspace=0.02
)

ax_dendro = fig.add_subplot(gs[0, 0])
ax_heatmap = fig.add_subplot(gs[1, 0])
ax_cbar = fig.add_subplot(gs[1, 1])
ax_spacer = fig.add_subplot(gs[0, 1])
ax_spacer.axis('off')

dendro = hierarchy.dendrogram(
    cluster_hierarch,
    labels=state_labels,
    leaf_rotation=90,
    ax=ax_dendro
)
ax_dendro.set_xticks([])

leaf_order = dendro['leaves']
X_ordered = state_feature_matrix[leaf_order]
labels_ordered = [state_labels[i] for i in leaf_order]

cmap = plt.get_cmap('Reds').copy()
im = ax_heatmap.imshow(
    X_ordered.T,
    aspect='auto',
    cmap=cmap
)

ax_heatmap.set_xticks(range(len(labels_ordered)))
ax_heatmap.set_xticklabels(labels_ordered, rotation=90)
ax_heatmap.set_yticks(range(state_feature_matrix.shape[1]))
ax_heatmap.set_yticklabels(symptoms)

fig.colorbar(im, cax=ax_cbar)

plt.savefig("outputs/2026_07_20/clustering.png", bbox_inches='tight', dpi=300)