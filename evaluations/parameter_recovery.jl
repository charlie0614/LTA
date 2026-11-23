# Parameter recovery: can the estimation find the parameters that generated the data?
#
# 1. take the best fitted parameters of a run (sim_no) as the "true" parameters
# 2. simulate new symptom trajectories for the real patients (their covariates, T, and
#    their missingness pattern), so the simulated data look like the real data
# 3. fit the model to the simulated data the same way as the real data
# 4. compare true and estimated parameters
#
# States are only identified up to relabelling, so the estimated states are first
# matched to the true states via their emission probabilities. beta_initial,
# beta_transition and rho are not compared directly: they depend on the reference state,
# and rho is only identified up to scale and sign (rho * c with the slopes / c gives the
# same model). The comparison is therefore done on the probability scale (initial and
# transition probabilities per patient, emission probabilities), plus the ordinal
# state effects eta and cutpoints tau, which are identified.
#
# Everything is saved next to the original run:
#   outputs/<sim_no>/parameter_recovery/rep<r>/simulation/sim_output.jld2
#   outputs/<sim_no>/parameter_recovery/rep<r>/results/estimates/est_output.jld2
#   outputs/<sim_no>/parameter_recovery/rep<r>/results/plots/...
#
# One replicate per job (the replicate number is the simulation seed):
#   julia evaluations/parameter_recovery.jl 1
#   julia evaluations/parameter_recovery.jl 2   ...

include("../src/LTA.jl")
include("../digihero_params.jl")
using .LTA
using Accessors
using CSV
using DataFrames
using JLD2
using Statistics
using PlotlyJS

######### SETTINGS ##########

n_states = 5
sim_no = string("2026_07_20/", n_states, "states/age_sex_comorbidity_covs_fups_ordinal")   # run whose best fit is the truth
rep = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1   # replicate = simulation seed
n_starts = 10
iterations = 500
est_seed = 1
category_labels = ["0: not present", "1: little", "2: strong"]

rec_sim_no = string(sim_no, "/parameter_recovery/rep", rep)
rec_dir = "outputs/" * rec_sim_no
mkpath(rec_dir * "/simulation")
mkpath(rec_dir * "/results/estimates")
mkpath(rec_dir * "/results/plots")
println("Parameter recovery for ", sim_no, ", replicate ", rep)

######### TRUE PARAMETERS AND SIMULATION ##########

real_est = load("outputs/" * sim_no * "/results/estimates/est_output.jld2", "estimation_output")
real_sim, sim_params, sim_hyper = LTA.unpack(real_est)
true_params = LTA.get_model_params(real_est, type="best_fitted")

real_obs = real_sim.observations.ordinal_observations      # N × T × L, -1 = missing
T = size(real_obs, 2)

# same patients and covariates, but the true parameters are the best fit
true_sim_params = LTA.SimParams(sim_hyper=(@set sim_hyper.sim_no = rec_sim_no),
    covariate_mat_headers=sim_params.covariate_mat_headers,
    covariate_mat=sim_params.covariate_mat,
    model_params=true_params)
sim = LTA.run_simulation(true_sim_params; simulation_seed=rep, T=T, percentage_observations_missing=0.0)

# copy the missingness pattern of the real data
sim_obs = copy(sim.observations.ordinal_observations)
sim_obs[real_obs .== -1] .= -1
sim_output = LTA.SimulationOutput(sim_params=true_sim_params, simulation_seed=rep,
    states=sim.states, observations=(ordinal_observations=sim_obs,))
save(rec_dir * "/simulation/sim_output.jld2", "sim_output", sim_output)

# the number of categories is fixed by sim_hyper; warn if a category was never simulated
for l in axes(sim_obs, 3)
    obs_l = sim_obs[:, :, l][sim_obs[:, :, l] .>= 0]
    missing_cats = setdiff(0:(sim_hyper.n_obs_tup.ordinal_categories-1), unique(obs_l))
    isempty(missing_cats) || @warn "$(symptoms_list_ordinal[l]): category $(missing_cats) never simulated"
end

######### ESTIMATION ##########

est_file = rec_dir * "/results/estimates/est_output.jld2"
if isfile(est_file)
    estimation_output = load(est_file, "estimation_output")
else
    estimation_output = LTA.run_estimation(sim_output; est_seed=est_seed, meth="BFGS",
        n_starts=n_starts, iterations=iterations)
    save(est_file, "estimation_output", estimation_output)
end
est_params = LTA.get_model_params(estimation_output, type="best_fitted")

# the fitted NLL should be <= the NLL at the true parameters (it is the maximum likelihood);
# if it is clearly larger, the optimiser did not find the global optimum
nll_true = estimation_output.true_log_lkl
nll_fit = minimum(estimation_output.fitted_log_lkl_list)
println("NLL at true parameters: ", nll_true, "   best fitted NLL: ", nll_fit,
    "   starts within 1 of best: ", count(estimation_output.fitted_log_lkl_list .< nll_fit + 1), "/", n_starts)

######### MATCH STATES ##########

P_true = LTA.get_ordinal_probs(true_params.emissions.beta_ordinal)   # S × L × J
P_est = LTA.get_ordinal_probs(est_params.emissions.beta_ordinal)

# distance between two states = Wasserstein distance per symptom summed over symptoms,
# as in clustering_ordinal.py. For categories 0..J-1 this is sum_k |P(X ≥ k) - P'(X ≥ k)|,
# i.e. the shift in severity in units of categories (= difference in expected severity
# if one state is more severe in every category)
p_geq(P) = reverse(cumsum(reverse(P, dims=3), dims=3), dims=3)[:, :, 2:end]   # S × L × (J-1)
wasserstein_cost(P_true, P_est) =
    [sum(abs, p_geq(P_true)[i, :, :] .- p_geq(P_est)[j, :, :]) for i in axes(P_true, 1), j in axes(P_est, 1)]

# perm[i] = estimated state that corresponds to true state i
# (one-to-one: greedy assignment on the distance, improved by pairwise swaps)
function match_states(cost)
    S = size(cost, 1)
    perm = zeros(Int, S)
    for ij in sort(vec(CartesianIndices(cost)), by=ij -> cost[ij])
        i, j = Tuple(ij)
        perm[i] == 0 && !(j in perm) && (perm[i] = j)
    end
    improved = true
    while improved
        improved = false
        for a in 1:S-1, b in a+1:S
            if cost[a, perm[b]] + cost[b, perm[a]] < cost[a, perm[a]] + cost[b, perm[b]] - 1e-12
                perm[a], perm[b] = perm[b], perm[a]
                improved = true
            end
        end
    end
    return perm
end
cost = wasserstein_cost(P_true, P_est)
perm = match_states(cost)
println("true state i ↔ estimated state perm[i]: ", perm)

# matched distance vs distance to the closest other true state: a recovered state is much
# closer to its partner than the true states are to each other
cost_true = wasserstein_cost(P_true, P_true)
state_match = DataFrame(true_state=1:size(cost, 1), estimated_state=perm,
    distance=[cost[i, perm[i]] for i in axes(cost, 1)],
    nearest_other_true_state=[minimum(cost_true[i, setdiff(axes(cost, 1), i)]) for i in axes(cost, 1)])
CSV.write(rec_dir * "/results/state_matching.csv", state_match)
println(state_match)

######### COMPARISON ##########

X = sim_params.covariate_mat
cidx = LTA.convert_covariate_2_df_indices(sim_params.covariate_mat_headers, sim_hyper.covariate_tup)
N = size(X, 1)

initial_probs(mp) = reduce(hcat, [LTA.get_rho_beta_initial_states(mp.beta_initial, mp.rho_initial, X[i, cidx.initial]) for i in 1:N])  # S × N
transition_probs(mp) = cat([LTA.get_rho_beta_transition_mat(mp.beta_transition, mp.rho_trans, X[i, cidx.trans]) for i in 1:N]..., dims=3)  # S × S × N

init_true, init_est = initial_probs(true_params), initial_probs(est_params)[perm, :]
trans_true, trans_est = transition_probs(true_params), transition_probs(est_params)[perm, perm, :]

ord_true = LTA.convert_ordinal_2_true(true_params.emissions.beta_ordinal)
ord_est = LTA.convert_ordinal_2_true(est_params.emissions.beta_ordinal)
eta_true, eta_est = ord_true.beta[:, 1, :], ord_est.beta[perm, 1, :]    # S × L
tau_true, tau_est = ord_true.alpha[2:end, :], ord_est.alpha[2:end, :]   # tau_1 = 0 is fixed

S, L, J = size(P_true)
rows = DataFrame(group=String[], parameter=String[], true_value=Float64[], estimate=Float64[])
for i in 1:S
    push!(rows, ("initial probability (mean over patients)", "S$i", mean(init_true[i, :]), mean(init_est[i, :])))
end
for i in 1:S, j in 1:S
    push!(rows, ("transition probability (mean over patients)", "S$i→S$j", mean(trans_true[i, j, :]), mean(trans_est[i, j, :])))
end
for i in 1:S, l in 1:L, k in 1:J
    push!(rows, ("emission probability", "S$i $(symptoms_list_ordinal[l]) cat $(k-1)", P_true[i, l, k], P_est[perm[i], l, k]))
end
for i in 1:S, l in 1:L
    push!(rows, ("eta (state effect)", "S$i $(symptoms_list_ordinal[l])", eta_true[i, l], eta_est[i, l]))
end
for k in 1:size(tau_true, 1), l in 1:L
    push!(rows, ("tau (cutpoint)", "tau$(k+1) $(symptoms_list_ordinal[l])", tau_true[k, l], tau_est[k, l]))
end
rows.error = rows.estimate .- rows.true_value
CSV.write(rec_dir * "/results/recovery_parameters.csv", rows)

# error summary; for initial and transition probabilities also per patient (covariate effects)
errors = combine(groupby(rows, :group),
    :error => (e -> mean(abs.(e))) => :mean_abs_error,
    :error => (e -> maximum(abs.(e))) => :max_abs_error)
push!(errors, ("initial probability (per patient)", mean(abs.(init_est .- init_true)), maximum(abs.(init_est .- init_true))))
push!(errors, ("transition probability (per patient)", mean(abs.(trans_est .- trans_true)), maximum(abs.(trans_est .- trans_true))))
errors.rep .= rep
errors.nll_true .= nll_true
errors.nll_fit .= nll_fit
CSV.write(rec_dir * "/results/recovery_errors.csv", errors)
println(errors)

######### PLOTS ##########

plot_dir = rec_dir * "/results/plots/"

# true vs estimated, points on the diagonal = recovered
function scatter_true_vs_est(groups, title, file)
    sub = rows[in(groups).(rows.group), :]
    lo, hi = extrema(vcat(sub.true_value, sub.estimate))
    traces = [PlotlyJS.scatter(x=[lo, hi], y=[lo, hi], mode="lines",
        line=PlotlyJS.attr(color="gray", dash="dash"), name="y = x")]
    for g in groups
        r = sub[sub.group .== g, :]
        push!(traces, PlotlyJS.scatter(x=r.true_value, y=r.estimate, mode="markers",
            text=r.parameter, name=g, marker=PlotlyJS.attr(size=6, opacity=0.7)))
    end
    p = PlotlyJS.plot(traces, PlotlyJS.Layout(title=title,
        xaxis=PlotlyJS.attr(title="true"), yaxis=PlotlyJS.attr(title="estimated", scaleanchor="x"),
        width=800, height=700))
    PlotlyJS.savefig(p, plot_dir * file, width=800, height=700)
end
scatter_true_vs_est(["initial probability (mean over patients)", "transition probability (mean over patients)", "emission probability"],
    "Recovery of probabilities ($n_states states, replicate $rep)", "recovery_probabilities.png")
scatter_true_vs_est(["eta (state effect)", "tau (cutpoint)"],
    "Recovery of ordinal parameters ($n_states states, replicate $rep)", "recovery_ordinal.png")

# emission heatmaps side by side; the estimated states are put in the order of the true states
est_params_matched = (; est_params..., emissions=(beta_ordinal=(
    beta=est_params.emissions.beta_ordinal.beta[perm, :, :],
    alpha=est_params.emissions.beta_ordinal.alpha),))
for (name, mp) in [("true", true_params), ("estimated", est_params_matched)]
    fig = LTA.plot_ordinal_heatmaps(mp; ordinal_labels=symptoms_list_ordinal, category_labels=category_labels)
    LTA.savefig_auto(fig, plot_dir * "emission_matrix_ordinal_$(name).png")
end
