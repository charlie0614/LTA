# K-fold cross validation for choosing the number of states.
#
# Needs the preprocessed data with a "fold" column:
#   python preprocessing_combined.py --folds 5
# For every number of states and every fold k, the model is fitted on all folds != k
# and the negative log-likelihood is evaluated on fold k with the fitted parameters.
# Every patient is held out exactly once, so the held-out NLL summed over the folds
# is comparable across models (lower is better).
#
# Run one job per number of states (or per number of states and fold):
#   julia evaluations/cross_validation.jl 5        # 5 states, all folds
#   julia evaluations/cross_validation.jl 5 2      # 5 states, fold 2 only
#   julia evaluations/cross_validation.jl          # everything, sequentially
# Each fit writes its own row to <out_dir>/fold_results/. When all jobs are done,
# evaluations/cv_model_selection.jl combines them and makes the model selection plots.

include("../src/LTA.jl")
include("../digihero_params.jl")
using .LTA
using CSV
using DataFrames
using JLD2
using Statistics
using PlotlyJS
using Plots

######### SETTINGS ##########

data_file = "data/DigiHero/2026-07-20_data_symptoms_DigiHero_Bonn_preprocessed_ordinal.csv"
out_dir = "outputs/2026_07_20/cross_validation/"
n_state_list = length(ARGS) >= 1 ? [parse(Int, ARGS[1])] : 3:10
n_starts = 10          # per fold; fewer than the full fit to keep the run time manageable
iterations = 500
est_seed = 1

covs = covariates_list_new_agegroups
time_labels = ["pcc_" * string(t) * "months_sym_" for t in 6:6:30]

######### DATA ##########

combined_all = CSV.read(data_file, DataFrame, missingstring=["", "NA"])
@assert "fold" in names(combined_all) "No fold column: run `python preprocessing_combined.py --folds 5` first"
folds = sort(unique(combined_all.fold))
fold_list = length(ARGS) >= 2 ? [parse(Int, ARGS[2])] : folds

# same column selection and row filter as LTA_DigiHero.jl
pattern = Regex(join(vcat(symptoms_list_ordinal, covs), "|"))
function prepare(df::DataFrame)
    d = select(df, names(df, pattern))
    d = d[:, 3:end]
    return filter(row -> !all(ismissing, row[Not(covs)]), d)
end

make_sim(d::DataFrame, n_states::Int, sim_no::String) = LTA.create_sim_mod_data(d;
    covs=covs, bins=String[], ordinals=symptoms_list_ordinal, conts=String[], visits=String[],
    labels=time_labels, n_states=n_states,
    covariate_tup=(initial=covs, trans=covs, em=[]),
    sim_no=sim_no, comments="")

n_categories(sim_output) = LTA.unpack(sim_output)[2].n_obs_tup.ordinal_categories
n_patients(sim_output) = LTA.unpack(sim_output)[2].N

# J is read off the data; every fold has to agree with the full data, otherwise
# the parameter vector of the training fit does not match the test data
J_full = n_categories(make_sim(prepare(combined_all), first(n_state_list), "cv_check"))

######### EVALUATION PLOTS PER FIT ##########

make_fit_plots = true   # the plots of LTA_DigiHero.jl for every (n_states, fold) fit
category_labels = ["0: not present", "1: little", "2: strong"]

function plot_fit(estimation_output, plot_dir::String, n_states::Int)
    mkpath(plot_dir)
    _, sim_params, _ = LTA.unpack(estimation_output)
    best_model_params = LTA.get_model_params(estimation_output, type="best_fitted")

    fig_ord = LTA.plot_ordinal_heatmaps(best_model_params;
        ordinal_labels=symptoms_list_ordinal, category_labels=category_labels)
    LTA.savefig_auto(fig_ord, plot_dir * "emission_matrix_ordinal.png")

    fig_ord_time = LTA.compare_estimation_2_data_fup(estimation_output; T=5, observation_type="ordinal",
        symptom_names=symptoms_list_ordinal, category_labels=category_labels)
    LTA.savefig_auto(fig_ord_time, plot_dir * "compare_est_2_data_ordinal.png")

    LTA.savefig_auto(LTA.waterfall_plot(estimation_output), plot_dir * "waterfall.png")

    rtrans_range = LTA.extract_rtrans_range(best_model_params.rho_trans, sim_params.covariate_mat)
    LTA.savefig_auto(LTA.plot_transition_matrices(best_model_params.beta_transition, rtrans_range),
        plot_dir * "transition_matrices_covs.png")
    LTA.savefig_auto(LTA.plot_initial_probs(best_model_params.beta_initial, rtrans_range),
        plot_dir * "initial_probs_covs.png")

    fig_uq = LTA.ensemble_uq_monthly_return_traces(estimation_output; T=5,
        symptom_names=symptoms_list_ordinal, observation_type="ordinal")
    LTA.savefig_auto(fig_uq, plot_dir * "ensemble_uq_ordinal.png")

    fig_states = LTA.estimated_states_over_time(estimation_output; T=5, type="best_fitted", n_states=n_states)
    Plots.savefig(fig_states, plot_dir * "estimated_states_over_time.png")
end

######### CROSS VALIDATION ##########

result_dir = out_dir * "fold_results/"
mkpath(result_dir)

for n_states in n_state_list, k in fold_list
    sim_no = string("cross_validation/", n_states, "states/fold", k)
    train_sim = make_sim(prepare(combined_all[combined_all.fold .!= k, :]), n_states, sim_no)
    test_sim = make_sim(prepare(combined_all[combined_all.fold .== k, :]), n_states, sim_no)
    @assert n_categories(train_sim) == n_categories(test_sim) == J_full "Fold $k does not contain all $J_full categories"

    # reuse finished fits, so an interrupted run can be restarted
    est_file = out_dir * string(n_states, "states_fold", k, ".jld2")
    if isfile(est_file)
        estimation_output = load(est_file, "estimation_output")
    else
        println("Fitting ", n_states, " states, fold ", k)
        estimation_output = LTA.run_estimation(train_sim; est_seed=est_seed, meth="BFGS",
            n_starts=n_starts, iterations=iterations)
        save(est_file, "estimation_output", estimation_output)
    end
    make_fit_plots && plot_fit(estimation_output, out_dir * string(n_states, "states/fold", k, "/plots/"), n_states)

    best_params = LTA.get_model_params(estimation_output, type="best_fitted")
    params_vec = LTA.model_params_2_param_vec(best_params)

    # one file per fit, so parallel jobs never write to the same file
    row = DataFrame(n_states=n_states, fold=k, n_folds=length(folds),
        n_train=n_patients(train_sim), n_test=n_patients(test_sim),
        nll_train=minimum(estimation_output.fitted_log_lkl_list),
        nll_test=LTA.get_lkl_val(params_vec, test_sim))   # lambda = 0: plain negative log-likelihood
    CSV.write(result_dir * string(n_states, "states_fold", k, ".csv"), row)
end
