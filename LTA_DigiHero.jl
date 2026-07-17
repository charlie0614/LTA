include("src/LTA.jl")
include("digihero_params.jl")
using .LTA
using CSV
using DataFrames
using JLD2
using PlotlyJS
using Plots

######### PREPROCESSING ##########

sim_dir = "data/DigiHero/"
n_states = 3
sim_no = string("2026_07_01/", n_states, "states/age_sex_comorbidity_covs_datafusion")
combined_all = CSV.read(sim_dir * "2026-07-01_data_symptoms_DigiHero_Bonn_preprocessed.csv", DataFrame, missingstring=["", "NA"]);

println("Running simulation with ", sim_no)

symptoms = vcat(symptoms_list, symptoms_list_fup)

function add_dummy_symptom_cols(combined_all)
    for timepoint in ["9months", "15months", "21months", "27months"]
        # Add missing timepoint symptom columns
        for symptom in symptoms
            combined_all[!, "pcc_"*timepoint*"_sym_"*symptom] .= -1
        end
    end

    # Rename columns
    return combined_all
end

pattern = Regex(join(vcat(symptoms, covariates_list_new_agegroups), "|"))
preprocessed_df = select(combined_all, names(combined_all, pattern))
preprocessed_df = preprocessed_df[:, 3:end]
preprocessed_df = add_dummy_symptom_cols(preprocessed_df)

preprocessed_df = filter(row -> !all(ismissing, row[Not(covariates_list_new_agegroups)]), preprocessed_df)

sim_output = LTA.create_sim_mod_data(preprocessed_df; covs=covariates_list_new_agegroups, bins=symptoms, conts=String[], visits=String[], labels=["pcc_" * string(timepoints)*"months_sym_" for timepoints in collect(0:3:30)], n_states=n_states, covariate_tup=(initial=covariates_list_new_agegroups, trans=covariates_list_new_agegroups, em=[]), sim_no=sim_no, comments="")
# save("outputs/" * sim_no * "/simulation/sim_output.jld2", "sim_output", sim_output)

# ############ ESTIMATION #############

# estimation_output = LTA.run_estimation(sim_output; est_seed=1,
#     # meth = "GradientDescent",
#     meth="BFGS",
#     n_starts=2,
#     iterations=5)

# save("outputs/" * sim_no * "/results/estimates/est_output.jld2", "estimation_output", estimation_output)

estimation_output = load("outputs/" * sim_no * "/results/estimates/est_output.jld2", "estimation_output")

mkpath("outputs/" * sim_no * "/results/plots")

sim_output, sim_params, sim_hyper = LTA.unpack(estimation_output)
covariate_idx = LTA.convert_covariate_2_df_indices(
    sim_params.covariate_mat_headers,
    sim_hyper.covariate_tup
)

best_model_params = LTA.get_model_params(estimation_output, type="best_fitted")

fig_em = PlotlyJS.make_subplots(
    rows=1, cols=1
)
em_states = LTA.get_bernoulli_probs(best_model_params.emissions.beta_bernoulli, sim_params.covariate_mat[1, :][covariate_idx[:em]])
true_heat_trace = LTA.plot_single_bernoulli_probs(em_states; symptom_labels=symptoms)
add_trace!(fig_em, true_heat_trace, row=1, col=1)
LTA.savefig_auto(fig_em, "outputs/" * sim_no * "/results/plots/emission_matrix.png", width=600, height=400)

titles = reshape(["Emission matrix", "Transition matrix", "Initial States"], 1, 3)
fig_em_trans = PlotlyJS.make_subplots(
    rows=1, cols=3, column_widths=[0.4, 0.4, 0.2], subplot_titles=titles
)
em_states = LTA.get_bernoulli_probs(best_model_params.emissions.beta_bernoulli, sim_params.covariate_mat[1, :][covariate_idx[:em]])
true_heat_trace = LTA.plot_single_bernoulli_probs(em_states; symptom_labels=symptoms)
add_trace!(fig_em_trans, true_heat_trace, row=1, col=1)
transition_probs = LTA.get_rho_beta_transition_mat(
    best_model_params.beta_transition, best_model_params.rho_trans, sim_params.covariate_mat[1, :])
    transition_probs = LTA.get_rho_beta_transition_mat(
    best_model_params.beta_transition, best_model_params.rho_trans, sim_params.covariate_mat[1, :])
true_heat_trace = LTA.plot_single_transition_mat(transition_probs)
add_trace!(fig_em_trans, true_heat_trace, row=1, col=2)#, name="Transition matrix")
initial_states = LTA.get_rho_beta_initial_states(
    best_model_params.beta_initial,
    best_model_params.rho_initial,
    sim_params.covariate_mat[1, :][covariate_idx[:initial]])
true_heat_trace = LTA.plot_single_initial(initial_states)
add_trace!(fig_em_trans, true_heat_trace, row=1, col=3)#, name="Initial States")
relayout!(fig_em_trans,
    xaxis_showticklabels=true,
    xaxis2_showticklabels=true,
    xaxis3_showticklabels=false
)
LTA.savefig_auto(fig_em_trans, "outputs/" * sim_no * "/results/plots/heatmaps.png", width=1500, height=400)

obs = sim_output.observations.bernoulli_observations

fig1 = LTA.compare_estimation_2_data_fup(estimation_output; T=11, symptom_names=symptoms, type="best_fitted", observation_type="bernoulli")
fig1_with_CI = LTA.compare_estimation_2_data_with_CIs_fup(estimation_output; T=11, symptom_names=symptoms, type="best_fitted", observation_type="bernoulli")
LTA.savefig_auto(fig1, "outputs/" * sim_no * "/results/plots/compare_est_2_data.png")
LTA.savefig_auto(fig1_with_CI, "outputs/" * sim_no * "/results/plots/compare_est_2_data_with_CIs.png")
fig2 = LTA.compare_bernoulli_heatmaps(estimation_output)
LTA.savefig_auto(fig2, "outputs/" * sim_no * "/results/plots/compare_bernoulli_heatmaps.png")
fig3 = LTA.compare_rhos(estimation_output)
LTA.savefig_auto(fig3, "outputs/" * sim_no * "/results/plots/compare_rhos.png")

waterfall = LTA.waterfall_plot(estimation_output)
LTA.savefig_auto(waterfall, "outputs/" * sim_no * "/results/plots/waterfall.png")

range = LTA.extract_rtrans_hist(best_model_params.rho_trans, sim_params.covariate_mat)
range = LTA.extract_rtrans_range(best_model_params.rho_trans, sim_params.covariate_mat)

println(range)
fig4 = LTA.plot_transition_matrices(best_model_params.beta_transition, range)
LTA.savefig_auto(fig4, "outputs/" * sim_no * "/results/plots/transition_matrices_covs_test.png")

fig5 = LTA.plot_initial_probs(best_model_params.beta_initial, range)
LTA.savefig_auto(fig5, "outputs/" * sim_no * "/results/plots/initial_probs_covs.png")


# Ensemble-based uncertainty quantification 
fig = LTA.ensemble_uq_fup(estimation_output; T=11, symptom_names=symptoms, observation_type="bernoulli", num_multistarts=2)
LTA.savefig_auto(fig, "outputs/" * sim_no * "/results/plots/ensemble_uq.png")

fig = LTA.estimated_states_over_time(estimation_output; T=11, type="best_fitted")
Plots.savefig(fig, "outputs/" * sim_no * "/results/plots/estimated_states_over_time.png")