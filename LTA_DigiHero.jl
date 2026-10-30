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
n_states = 5
sim_no = string("2026_07_20/", n_states, "states/age_sex_comorbidity_covs_fups_all_symptoms_all_steps")
combined_all = CSV.read(sim_dir * "2026-07-20_data_symptoms_DigiHero_Bonn_preprocessed.csv", DataFrame, missingstring=["", "NA"]);

println("Running simulation with ", sim_no)

pattern = Regex(join(vcat(symptoms_list_binary, symptoms_list_ordinal, covariates_list_new_agegroups), "|"))
preprocessed_df = select(combined_all, names(combined_all, pattern))
preprocessed_df = preprocessed_df[:, 3:end]

preprocessed_df = filter(row -> !all(ismissing, row[Not(covariates_list_new_agegroups)]), preprocessed_df)

sim_output = LTA.create_sim_mod_data(preprocessed_df; covs=covariates_list_new_agegroups, bins=String[], ordinals=symptoms_list_ordinal, conts=String[], visits=String[], labels=["pcc_" * string(timepoints) * "months_sym_" for timepoints in collect(6:6:30)], n_states=n_states, covariate_tup=(initial=covariates_list_new_agegroups, trans=covariates_list_new_agegroups, em=[]), sim_no=sim_no, comments="")
save("outputs/" * sim_no * "/simulation/sim_output.jld2", "sim_output", sim_output)

############ ESTIMATION #############

estimation_output = LTA.run_estimation(sim_output; est_seed=1,
    # meth = "GradientDescent",
    meth="BFGS",
    n_starts=20,
    iterations=500)

save("outputs/" * sim_no * "/results/estimates/est_output.jld2", "estimation_output", estimation_output)

# estimation_output = load("outputs/" * sim_no * "/results/estimates/est_output.jld2", "estimation_output")
println("Iterations limit reached: ", count(estimation_output.iteration_limit_reached), "/", length(estimation_output.iteration_limit_reached))

mkpath("outputs/" * sim_no * "/results/plots")

sim_output, sim_params, sim_hyper = LTA.unpack(estimation_output)
covariate_idx = LTA.convert_covariate_2_df_indices(
    sim_params.covariate_mat_headers,
    sim_hyper.covariate_tup
)

best_model_params = LTA.get_model_params(estimation_output, type="best_fitted")

fig_ord = LTA.plot_ordinal_heatmaps(best_model_params;
    ordinal_labels=symptoms_list_ordinal,
    category_labels=["0: not present", "1: little", "2: strong"])
LTA.savefig_auto(fig_ord, "outputs/" * sim_no * "/results/plots/emission_matrix_ordinal.png")

fig_ord_time = LTA.compare_estimation_2_data_fup(estimation_output; T=5, observation_type="ordinal",
    symptom_names=symptoms_list_ordinal, category_labels=["0: not present", "1: little", "2: strong"])
LTA.savefig_auto(fig_ord_time, "outputs/" * sim_no * "/results/plots/compare_est_2_data_ordinal.png")

waterfall = LTA.waterfall_plot(estimation_output)
LTA.savefig_auto(waterfall, "outputs/" * sim_no * "/results/plots/waterfall.png")

range = LTA.extract_rtrans_hist(best_model_params.rho_trans, sim_params.covariate_mat)
range = LTA.extract_rtrans_range(best_model_params.rho_trans, sim_params.covariate_mat)

fig4 = LTA.plot_transition_matrices(best_model_params.beta_transition, range)
LTA.savefig_auto(fig4, "outputs/" * sim_no * "/results/plots/transition_matrices_covs_test.png")

fig5 = LTA.plot_initial_probs(best_model_params.beta_initial, range)
LTA.savefig_auto(fig5, "outputs/" * sim_no * "/results/plots/initial_probs_covs.png")

# Ensemble-based uncertainty quantification 
fig = LTA.ensemble_uq_monthly_return_traces(estimation_output; T=5, symptom_names=symptoms_list_ordinal, observation_type="ordinal")
LTA.savefig_auto(fig, "outputs/" * sim_no * "/results/plots/ensemble_uq_ordinal.png")

fig = LTA.estimated_states_over_time(estimation_output; T=5, type="best_fitted", n_states=n_states)
Plots.savefig(fig, "outputs/" * sim_no * "/results/plots/estimated_states_over_time.png")