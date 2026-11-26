# Does the ordinal information reduce the uncertainty of the transition matrices?
#
# For every number of states, the ensemble UQ of the binary and of the ordinal fit is computed
# (ensemble_uq_transition_stats) and the widths of the ensemble intervals are compared.
# The states of the two models are matched via P(X ≥ 1) (Wasserstein distance with the
# ordinal categories collapsed to 0 / ≥1), the covariate combinations via their covariate values.
# Both fits need the same symptoms (in the same order) and covariates.

include("../src/LTA.jl")
using .LTA
using CSV
using DataFrames
using JLD2
using Statistics
using PlotlyJS

######### SETTINGS ##########

sim_dir = "outputs/2026_07_20/"
n_state_list = 3:9
runs = (binary="age_sex_comorbidity_covs_fups", ordinal="age_sex_comorbidity_covs_fups_ordinal")
select_function = LTA.select_params_with_missingness   # or LTA.select_params
out_dir = sim_dir * "transition_uncertainty/"
mkpath(out_dir)

######### ENSEMBLE UQ PER FIT ##########

# the selection of the ensemble is the expensive part: it is saved next to the fit and reused
function transition_stats(n_states, run)
    est_dir = string(sim_dir, n_states, "states/", run, "/results/estimates/")
    est_output = load(est_dir * "est_output.jld2", "estimation_output")
    kept_file = est_dir * "kept_params_" * string(nameof(select_function)) * ".jld2"
    if isfile(kept_file)
        kept = load(kept_file, "kept_params")
    else
        kept = LTA.get_ensemble_params(est_output; T=5, select_function=select_function)
        save(kept_file, "kept_params", kept)
    end
    stats = LTA.ensemble_uq_transition_stats(est_output; kept_params=kept)
    return stats, LTA.get_model_params(est_output, type="best_fitted")
end

######### COMPARISON ##########

transitions = DataFrame()
state_matching = DataFrame()
for n_states in n_state_list
    stats_bin, best_bin = transition_stats(n_states, runs.binary)
    stats_ord, best_ord = transition_stats(n_states, runs.ordinal)

    # match the binary states to the ordinal states via P(X ≥ 1)
    p_bin = LTA.state_category_probs(best_bin)[1]                        # S × L × 2
    P_ord = LTA.state_category_probs(best_ord)[1]                        # S × L × J
    p_ord = cat(P_ord[:, :, 1], 1 .- P_ord[:, :, 1], dims=3)             # collapsed to 0 / ≥1
    cost = [LTA.wasserstein_distance(p_ord[i, :, :], p_bin[j, :, :]) for i in 1:n_states, j in 1:n_states]
    perm = LTA.match_states(cost)   # perm[i] = binary state matched to ordinal state i
    append!(state_matching, DataFrame(n_states=n_states, ordinal_state=1:n_states, binary_state=perm,
        distance=[cost[i, perm[i]] for i in 1:n_states]))

    # same covariate combination in both fits
    titles = LTA.covariate_profile_titles(stats_ord.covariates)
    bin_rows = collect.(eachrow(stats_bin.covariates))
    for (c, ord_row) in enumerate(eachrow(stats_ord.covariates))
        c_bin = findfirst(==(collect(ord_row)), bin_rows)
        isnothing(c_bin) && continue
        for i in 1:n_states, j in 1:n_states
            push!(transitions, (n_states=n_states, covariates=titles[c], from=i, to=j,
                    best_ordinal=stats_ord.best[c][i, j], best_binary=stats_bin.best[c_bin][perm[i], perm[j]],
                    width_ordinal=stats_ord.width[c][i, j], width_binary=stats_bin.width[c_bin][perm[i], perm[j]],
                    n_kept_ordinal=stats_ord.n_kept, n_kept_binary=stats_bin.n_kept))
        end
    end
end
CSV.write(out_dir * "transitions.csv", transitions)        # states numbered as in the ordinal fit
CSV.write(out_dir * "state_matching.csv", state_matching)

# per number of states: mean interval width and how often the ordinal interval is narrower
uq_summary = combine(groupby(transitions, :n_states),
    :width_binary => mean => :mean_width_binary,
    :width_ordinal => mean => :mean_width_ordinal,
    [:width_ordinal, :width_binary] => ((o, b) -> mean(o) / mean(b)) => :ratio_ordinal_binary,
    [:width_ordinal, :width_binary] => ((o, b) -> mean(o .< b)) => :share_ordinal_narrower,
    :n_kept_binary => first => :n_kept_binary,
    :n_kept_ordinal => first => :n_kept_ordinal)
CSV.write(out_dir * "summary.csv", uq_summary)
println(uq_summary)

######### PLOT ##########

p = PlotlyJS.plot([
        PlotlyJS.scatter(x=uq_summary.n_states, y=uq_summary.mean_width_binary, mode="lines+markers", name="binary"),
        PlotlyJS.scatter(x=uq_summary.n_states, y=uq_summary.mean_width_ordinal, mode="lines+markers", name="ordinal")],
    PlotlyJS.Layout(title="Uncertainty of the transition probabilities",
        xaxis=PlotlyJS.attr(title="Number of states", tickmode="linear", dtick=1),
        yaxis=PlotlyJS.attr(title="Mean width of the 95% ensemble interval"), width=800, height=500))
PlotlyJS.savefig(p, out_dir * "mean_interval_width.png", width=800, height=500)
