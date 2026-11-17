
using HDF5
include("../src/LTA.jl")
using .LTA
using CSV
using DataFrames
using JLD2
using PlotlyJS
using Plots

function export_best_emission_matrices_ordinal(sim_dir::String, out_file::String)
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

            em = best_model_params.emissions
            feats = Matrix{Float64}[]
            if haskey(em, :beta_bernoulli)   # any symptoms you kept binary
                push!(feats, LTA.get_bernoulli_probs(em.beta_bernoulli,
                                                    sim_params.covariate_mat[1, :][covariate_idx[:em]]))
            end
            if haskey(em, :beta_ordinal)
                p = LTA.get_ordinal_probs(em.beta_ordinal)                      # S × L × J
                p_geq = reverse(cumsum(reverse(p, dims=3), dims=3), dims=3)     # P(X ≥ k-1)
                p_geq = p_geq[:, :, 2:end]                                      # drop P(≥0)=1 → S × L × (J-1)
                push!(feats, reshape(p_geq, size(p, 1), :))                     # S × (L·(J-1))
            end
            em_states = reduce(hcat, feats)

            write(f, "em_states_" * string(n_states), em_states)

        end
    end

    println("Saved emission matrices to ", out_file)
end

function export_best_emission_matrices_binary(sim_dir::String, out_file::String)
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

            em = best_model_params.emissions
            feats = Matrix{Float64}[]
            if haskey(em, :beta_bernoulli)   # any symptoms you kept binary
                push!(feats, LTA.get_bernoulli_probs(em.beta_bernoulli,
                                                    sim_params.covariate_mat[1, :][covariate_idx[:em]]))
            end
            if haskey(em, :beta_ordinal)
                p = LTA.get_ordinal_probs(em.beta_ordinal)   # S × L × J
                push!(feats, 1 .- p[:, :, 1])                # P(X ≥ 1) = 1 − P(X = 0) → S × L
            end
            em_states = reduce(hcat, feats)

            write(f, "em_states_" * string(n_states), em_states)

        end
    end

    println("Saved emission matrices to ", out_file)
end

sim_dir = "outputs/2026_07_20/"
export_best_emission_matrices_binary(sim_dir, sim_dir * "est_emission_mats_age_sex_comorbidity_covs_fups_ordinal_binary.h5")
export_best_emission_matrices_ordinal(sim_dir, sim_dir * "est_emission_mats_age_sex_comorbidity_covs_fups_ordinal_ordinal.h5")
