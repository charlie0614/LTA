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

    best_params = LTA.get_model_params(estimation_output, type="best_fitted")
    params_vec = LTA.model_params_2_param_vec(best_params)

    push!(results, (n_states, k, n_patients(train_sim), n_patients(test_sim),
        minimum(estimation_output.fitted_log_lkl_list),
        LTA.get_lkl_val(params_vec, test_sim)))   # lambda = 0: plain negative log-likelihood
    CSV.write(out_dir * "cv_folds.csv", results)
end

######### SUMMARY ##########

summary = combine(groupby(results, :n_states),
    :nll_test => sum => :nll_test_total,
    [:nll_test, :n_test] => ((l, n) -> sum(l) / sum(n)) => :nll_test_per_patient,
    [:nll_test, :n_test] => ((l, n) -> std(l ./ n)) => :sd_across_folds)
CSV.write(out_dir * "cv_summary.csv", summary)
println(summary)
