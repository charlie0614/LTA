
using HDF5
include("../src/LTA.jl")
using .LTA
using CSV
using DataFrames
using JLD2
using PlotlyJS
using Plots

sim_dir = "outputs/2026_07_20/"
out_file = sim_dir * "est_emission_mats_age_sex_comorbidity_covs_fups.h5"

h5open(out_file, "w") do f
    for n_states in 3:9
        sim_no = string(n_states, "states/age_sex_comorbidity_covs_fups/")
        estimation_output = load(sim_dir * sim_no * "results/estimates/est_output.jld2", "estimation_output")

        sim_output, sim_params, sim_hyper = LTA.unpack(estimation_output)
        covariate_idx = LTA.convert_covariate_2_df_indices(
            sim_params.covariate_mat_headers,
            sim_hyper.covariate_tup
        )
        best_model_params = LTA.get_model_params(estimation_output, type="best_fitted")

        em_states = LTA.get_bernoulli_probs(
            best_model_params.emissions.beta_bernoulli,
            sim_params.covariate_mat[1, :][covariate_idx[:em]]
        )
        write(f,  "em_states_" * string(n_states), em_states)

    end
end

println("Saved emission matrices to ", out_file)
