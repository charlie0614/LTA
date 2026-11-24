# Parameter recovery: take the best fit of a run as the true parameters, simulate new data
# for the same patients (covariates), fit the model to it and compare with the truth.
#
# States are only identified up to relabelling, so the estimated states are matched to the
# true states first (Wasserstein distance of the emissions, as in clustering_ordinal.py).
# beta_initial, beta_transition and rho depend on the reference state and rho is only
# identified up to scale and sign, so these are compared as probabilities.

include("../src/LTA.jl")
include("../digihero_params.jl")
using .LTA
using CSV
using DataFrames
using JLD2
using Statistics
using PlotlyJS

######### SETTINGS ##########

n_states = 5
sim_no = string("2026_07_20/", n_states, "states/age_sex_comorbidity_covs_fups_ordinal")   # run whose best fit is the truth
simulation_seed = 1
category_labels = ["0: not present", "1: little", "2: strong"]

rec_dir = "outputs/" * sim_no * "/parameter_recovery/seed" * string(simulation_seed)
mkpath(rec_dir * "/simulation")
mkpath(rec_dir * "/results/estimates")
mkpath(rec_dir * "/results/plots")

######### SIMULATION ##########

real_est = load("outputs/" * sim_no * "/results/estimates/est_output.jld2", "estimation_output")
real_sim, sim_params, sim_hyper = LTA.unpack(real_est)
true_params = LTA.get_model_params(real_est, type="best_fitted")

true_sim_params = LTA.SimParams(sim_hyper=sim_hyper,
    covariate_mat_headers=sim_params.covariate_mat_headers,
    covariate_mat=sim_params.covariate_mat,
    model_params=true_params)
sim_output = LTA.run_simulation(true_sim_params; simulation_seed=simulation_seed,
    T=size(real_sim.observations.ordinal_observations, 2))
save(rec_dir * "/simulation/sim_output.jld2", "sim_output", sim_output)

######### ESTIMATION ##########

estimation_output = LTA.run_estimation(sim_output; est_seed=1, meth="BFGS", n_starts=10, iterations=500)
save(rec_dir * "/results/estimates/est_output.jld2", "estimation_output", estimation_output)
est_params = LTA.get_model_params(estimation_output, type="best_fitted")

# the best fitted NLL should be <= the NLL at the true parameters
println("NLL at true parameters: ", estimation_output.true_log_lkl,
    "   best fitted NLL: ", minimum(estimation_output.fitted_log_lkl_list))

######### MATCH STATES ##########

P_true = LTA.get_ordinal_probs(true_params.emissions.beta_ordinal)   # S × L × J
P_est = LTA.get_ordinal_probs(est_params.emissions.beta_ordinal)

# Wasserstein distance between two states, summed over symptoms (as in clustering_ordinal.py).
# p, q: L × J category probabilities. For categories 0..J-1 the Wasserstein distance of one
# symptom is sum_k |P(X ≥ k) - Q(X ≥ k)|, k = 1..J-1.
function wasserstein_distance(p, q)
    p_geq = reverse(cumsum(reverse(p, dims=2), dims=2), dims=2)[:, 2:end]
    q_geq = reverse(cumsum(reverse(q, dims=2), dims=2), dims=2)[:, 2:end]
    return sum(abs.(p_geq .- q_geq))
end
cost = [wasserstein_distance(P_true[i, :, :], P_est[j, :, :]) for i in 1:n_states, j in 1:n_states]

# perm[i] = estimated state matched to true state i (one-to-one): closest pairs first,
# then swap pairs as long as that lowers the total distance
perm = zeros(Int, n_states)
for ij in sort(vec(CartesianIndices(cost)), by=ij -> cost[ij])
    i, j = Tuple(ij)
    perm[i] == 0 && !(j in perm) && (perm[i] = j)
end
improved = true
while improved
    global improved = false
    for a in 1:n_states-1, b in a+1:n_states
        if cost[a, perm[b]] + cost[b, perm[a]] < cost[a, perm[a]] + cost[b, perm[b]]
            perm[a], perm[b] = perm[b], perm[a]
            global improved = true
        end
    end
end
println("true state i ↔ estimated state perm[i]: ", perm)
println("matched distances: ", [cost[i, perm[i]] for i in 1:n_states])

######### COMPARISON ##########

X = sim_params.covariate_mat
cidx = LTA.convert_covariate_2_df_indices(sim_params.covariate_mat_headers, sim_hyper.covariate_tup)

# probabilities averaged over the patients' covariates
mean_initial(mp) = mean(LTA.get_rho_beta_initial_states(mp.beta_initial, mp.rho_initial, X[n, cidx.initial]) for n in axes(X, 1))
mean_transition(mp) = mean(LTA.get_rho_beta_transition_mat(mp.beta_transition, mp.rho_trans, X[n, cidx.trans]) for n in axes(X, 1))

ord_true = LTA.convert_ordinal_2_true(true_params.emissions.beta_ordinal)
ord_est = LTA.convert_ordinal_2_true(est_params.emissions.beta_ordinal)

comparison = [
    ("initial probability", mean_initial(true_params), mean_initial(est_params)[perm]),
    ("transition probability", mean_transition(true_params), mean_transition(est_params)[perm, perm]),
    ("emission probability", P_true, P_est[perm, :, :]),
    ("eta (state effect)", ord_true.beta, ord_est.beta[perm, :, :]),
    ("tau (cutpoint)", ord_true.alpha[2:end, :], ord_est.alpha[2:end, :]),   # tau_1 = 0 is fixed
]
rows = reduce(vcat, [DataFrame(group=g, true_value=vec(t), estimate=vec(e)) for (g, t, e) in comparison])
CSV.write(rec_dir * "/results/recovery_parameters.csv", rows)

errors = combine(groupby(rows, :group),
    [:true_value, :estimate] => ((t, e) -> mean(abs.(e .- t))) => :mean_abs_error,
    [:true_value, :estimate] => ((t, e) -> maximum(abs.(e .- t))) => :max_abs_error)
CSV.write(rec_dir * "/results/recovery_errors.csv", errors)
println(errors)

######### PLOTS ##########

# true vs estimated, points on the diagonal = recovered
function plot_true_vs_est(groups, file)
    sub = rows[in(groups).(rows.group), :]
    lo, hi = extrema(vcat(sub.true_value, sub.estimate))
    traces = [PlotlyJS.scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=PlotlyJS.attr(color="gray", dash="dash"), name="y = x")]
    for g in groups
        r = sub[sub.group .== g, :]
        push!(traces, PlotlyJS.scatter(x=r.true_value, y=r.estimate, mode="markers", name=g))
    end
    p = PlotlyJS.plot(traces, PlotlyJS.Layout(xaxis=PlotlyJS.attr(title="true"),
        yaxis=PlotlyJS.attr(title="estimated", scaleanchor="x"), width=800, height=700))
    PlotlyJS.savefig(p, rec_dir * "/results/plots/" * file, width=800, height=700)
end
plot_true_vs_est(["initial probability", "transition probability", "emission probability"], "recovery_probabilities.png")
plot_true_vs_est(["eta (state effect)", "tau (cutpoint)"], "recovery_ordinal.png")

# emission heatmaps, estimated states in the order of the true states
est_params_matched = (; est_params..., emissions=(beta_ordinal=(
    beta=est_params.emissions.beta_ordinal.beta[perm, :, :], alpha=est_params.emissions.beta_ordinal.alpha),))
for (name, mp) in [("true", true_params), ("estimated", est_params_matched)]
    fig = LTA.plot_ordinal_heatmaps(mp; ordinal_labels=symptoms_list_ordinal, category_labels=category_labels)
    LTA.savefig_auto(fig, rec_dir * "/results/plots/emission_matrix_ordinal_" * name * ".png")
end
