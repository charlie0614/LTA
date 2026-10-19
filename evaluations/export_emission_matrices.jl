
using HDF5
include("../src/LTA.jl")
using .LTA
using CSV
using DataFrames
using JLD2
using PlotlyJS
using Plots

function export_best_emission_matrices(sim_dir::String, out_file::String)
    h5open(out_file, "w") do f
        for n_states in 3:10
            sim_no = string(n_states, "states/age_sex_comorbidity_covs_fups_all_symptoms/")
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
            write(f, "em_states_" * string(n_states), em_states)

        end
    end

    println("Saved emission matrices to ", out_file)
end

function export_emission_matrices_multistarts(sim_dir::String, out_file::String, n_states::Int)
    h5open(sim_dir * string(n_states) * "states/" * out_file, "w") do f
        for multistart in 1:20
            sim_no = string(n_states, "states/age_sex_comorbidity_covs_fups_all_symptoms/")
            estimation_output = load(sim_dir * sim_no * "/results/estimates/est_output.jld2", "estimation_output")

            sim_output, sim_params, sim_hyper = LTA.unpack(estimation_output)
            covariate_idx = LTA.convert_covariate_2_df_indices(
                sim_params.covariate_mat_headers,
                sim_hyper.covariate_tup
            )
            best_model_params = LTA.get_model_params(estimation_output, type=multistart)

            em_states = LTA.get_bernoulli_probs(
                best_model_params.emissions.beta_bernoulli,
                sim_params.covariate_mat[1, :][covariate_idx[:em]]
            )
            write(f, "em_states_multistart_" * string(multistart), em_states)

        end
    end

    println("Saved emission matrices to ", out_file)
end

sim_dir = "outputs/2026_07_20/"
out_file = sim_dir * "est_emission_mats_age_sex_comorbidity_covs_fups_all_symptoms.h5"
export_best_emission_matrices(sim_dir, out_file)

for n_symptoms in 3:10
    export_emission_matrices_multistarts(sim_dir, "est_emission_mats_multistarts_age_sex_comorbidity_covs_fups_all_symptoms.h5", n_symptoms)
end
