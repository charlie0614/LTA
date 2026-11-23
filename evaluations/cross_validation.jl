# K-fold cross validation for choosing the number of states.
#
# Needs the preprocessed data with a "fold" column:
#   python preprocessing_combined.py --folds 5
# For every number of states and every fold k, the model is fitted on all folds != k
# and the negative log-likelihood is evaluated on fold k with the fitted parameters.
# Every patient is held out exactly once, so the held-out NLL summed over the folds
# is comparable across models (lower is better).

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
n_state_list = 3:10
n_starts = 10          # per fold; fewer than the full fit to keep the run time manageable
iterations = 500
est_seed = 1

covs = covariates_list_new_agegroups
time_labels = ["pcc_" * string(t) * "months_sym_" for t in 6:6:30]

######### DATA ##########

combined_all = CSV.read(data_file, DataFrame, missingstring=["", "NA"])
@assert "fold" in names(combined_all) "No fold column: run `python preprocessing_combined.py --folds 5` first"
folds = sort(unique(combined_all.fold))

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

mkpath(out_dir)
results = DataFrame(n_states=Int[], fold=Int[], n_train=Int[], n_test=Int[],
    nll_train=Float64[], nll_test=Float64[])

for n_states in n_state_list, k in folds
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

    push!(results, (n_states, k, n_patients(train_sim), n_patients(test_sim),
        minimum(estimation_output.fitted_log_lkl_list),
        LTA.get_lkl_val(params_vec, test_sim)))   # lambda = 0: plain negative log-likelihood
    CSV.write(out_dir * "cv_folds.csv", results)
end

######### SUMMARY ##########

cv_summary = combine(groupby(results, :n_states),
    :nll_test => sum => :nll_test_total,
    [:nll_test, :n_test] => ((l, n) -> sum(l) / sum(n)) => :nll_test_per_patient,
    [:nll_test, :n_test] => ((l, n) -> std(l ./ n)) => :sd_across_folds)
CSV.write(out_dir * "cv_summary.csv", cv_summary)
println(cv_summary)

######### MODEL SELECTION PLOTS ##########

xaxis = PlotlyJS.attr(title="Number of states", tickmode="linear", dtick=1)

# total held-out NLL, same as test_nll.png from the prediction check
p_total = PlotlyJS.plot(PlotlyJS.bar(x=cv_summary.n_states, y=cv_summary.nll_test_total),
    PlotlyJS.Layout(title="$(length(folds))-fold cross validation", xaxis=xaxis,
        yaxis=PlotlyJS.attr(title="Held-out NLL (sum over folds)"), width=800, height=500))
PlotlyJS.savefig(p_total, out_dir * "cv_nll_total.png", width=800, height=500)

# per patient: mean ± sd across folds, single folds as faint lines, training NLL for the overfitting gap
results.nll_test_pp = results.nll_test ./ results.n_test
results.nll_train_pp = results.nll_train ./ results.n_train
traces = PlotlyJS.GenericTrace[]
for k in folds
    r = sort(results[results.fold .== k, :], :n_states)
    push!(traces, PlotlyJS.scatter(x=r.n_states, y=r.nll_test_pp, mode="lines+markers",
        line=PlotlyJS.attr(color="lightgray"), marker=PlotlyJS.attr(color="lightgray"),
        name="fold $k", showlegend=false))
end
train_summary = combine(groupby(results, :n_states), :nll_train_pp => mean => :nll_train_pp)
push!(traces, PlotlyJS.scatter(x=train_summary.n_states, y=train_summary.nll_train_pp,
    mode="lines+markers", line=PlotlyJS.attr(dash="dash"), name="training"))
push!(traces, PlotlyJS.scatter(x=cv_summary.n_states, y=cv_summary.nll_test_per_patient,
    error_y=PlotlyJS.attr(type="data", array=cv_summary.sd_across_folds, visible=true),
    mode="lines+markers", name="held-out (mean ± sd over folds)"))
p_pp = PlotlyJS.plot(traces, PlotlyJS.Layout(title="$(length(folds))-fold cross validation",
    xaxis=xaxis, yaxis=PlotlyJS.attr(title="NLL per patient"), width=900, height=500))
PlotlyJS.savefig(p_pp, out_dir * "cv_nll_per_patient.png", width=900, height=500)
