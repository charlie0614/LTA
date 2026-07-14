
using HDF5


sim_dir = "estimates/"
out_file = sim_dir * "est_transition_mats_age_sex_covs_infectionphase.h5"

h5open(out_file, "w") do f
    for n_states in 3:7
        sim_no = string(n_states, "2025_12_16/age_sex_covs_infectionphase/")
        estimation_output = load(sim_dir * sim_no * "results/estimates/est_output.jld2", "estimation_output")

        sim_output, sim_params, sim_hyper = LTA.unpack(estimation_output)
        covariate_idx = LTA.convert_covariate_2_df_indices(
            sim_params.covariate_mat_headers,
            sim_hyper.covariate_tup
        )
        best_model_params = LTA.get_model_params(estimation_output, type="best_fitted")

        trans_states = LTA.get_bernoulli_probs(
            best_model_params.beta_transition,
            sim_params.covariate_mat[1, :][covariate_idx[:em]]
        )
        println("Estimated transition probabilities: ", trans_states)
                write(f,  "trans_states_" * string(n_states), trans_states)

    end
end

println("Saved transition matrices to ", out_file)
