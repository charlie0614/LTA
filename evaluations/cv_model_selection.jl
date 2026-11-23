# Model selection from the K-fold cross validation.
#
# Run after all jobs of evaluations/cross_validation.jl are done. Reads the per-fit
# results from <out_dir>/fold_results/ and writes cv_folds.csv, cv_summary.csv,
# cv_nll_total.png and cv_nll_per_patient.png to <out_dir>.

using CSV
using DataFrames
using Statistics
using PlotlyJS

out_dir = "outputs/2026_07_20/cross_validation/"   # same as in cross_validation.jl

######### COLLECT RESULTS ##########

result_files = filter(f -> endswith(f, ".csv"), readdir(out_dir * "fold_results/", join=true))
results = sort(reduce(vcat, CSV.read.(result_files, DataFrame)), [:n_states, :fold])
CSV.write(out_dir * "cv_folds.csv", results)

# the summed held-out NLL is only comparable when every patient was held out once,
# so number of states with missing folds (job not finished or failed) are dropped
n_folds = only(unique(results.n_folds))
complete = combine(groupby(results, :n_states), :fold => (f -> length(unique(f)) == n_folds) => :complete)
for s in complete.n_states[.!complete.complete]
    @warn "$s states: only folds $(sort(results.fold[results.n_states .== s])) of $n_folds done, left out"
end
results = results[in(complete.n_states[complete.complete]).(results.n_states), :]

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
    PlotlyJS.Layout(title="$n_folds-fold cross validation", xaxis=xaxis,
        yaxis=PlotlyJS.attr(title="Held-out NLL (sum over folds)"), width=800, height=500))
PlotlyJS.savefig(p_total, out_dir * "cv_nll_total.png", width=800, height=500)

# per patient: mean ± sd across folds, single folds as faint lines, training NLL for the overfitting gap
results.nll_test_pp = results.nll_test ./ results.n_test
results.nll_train_pp = results.nll_train ./ results.n_train
traces = PlotlyJS.GenericTrace[]
for k in sort(unique(results.fold))
    r = results[results.fold .== k, :]
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
p_pp = PlotlyJS.plot(traces, PlotlyJS.Layout(title="$n_folds-fold cross validation",
    xaxis=xaxis, yaxis=PlotlyJS.attr(title="NLL per patient"), width=900, height=500))
PlotlyJS.savefig(p_pp, out_dir * "cv_nll_per_patient.png", width=900, height=500)
