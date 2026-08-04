using PlotlyJS
using Statistics
using StatsBase
using Plots

cscale = [(0, "white"), (1, "red")]

function get_model_params(
    est_output::EstimationOutput;
    type = "simulation"
)

    sim_output, sim_params, sim_hyper = unpack(est_output)

    if type == "simulation"
        return est_output.sim_output.sim_params.model_params
    elseif type == "best_fitted"

        argmin_idx = argmin(est_output.fitted_log_lkl_list)
        model_params = est_output.fitted_model_params[argmin_idx]
        return model_params
    else
        model_params = est_output.fitted_model_params[type]
        return model_params
    end

end

# initial probs
function plot_single_initial(
    vec::Vector{Float64};
    # symptom_list = symptom_dict
)

    n_states = length(vec)
    mat = reshape(vec, (n_states, 1))
    plt =
        PlotlyJS.heatmap(
			x = "",
			y = reverse("State " .* string.(collect(1:n_states))),
			z = mat[end:-1:1, :],
			zmin = 0.0, zmax = 1.0,
			colorscale = cscale,
        )

    return plt
end

function compare_initial_heatmaps(
    est_output::EstimationOutput;
    patient_idx = 1,
    save_dir = nothing
)
    sim_output, sim_params, sim_hyper = unpack(est_output)

    covariate_idx = convert_covariate_2_df_indices(
        sim_params.covariate_mat_headers,
        sim_hyper.covariate_tup
    )
    x_vec = sim_params.covariate_mat[patient_idx, :]
    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

    subplot_titles = ["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))" for (val, idx) in enumerate(lkl_idx)]
    subplot_titles = vcat("Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))", subplot_titles)
    s = length(est_output.fitted_log_lkl_list) + 1
    fig = PlotlyJS.make_subplots(
		rows = 1, cols = s,
		subplot_titles = reshape(subplot_titles, 1, s)
    )

    # ------------------------------------------------------------------
    # true model
    model_params = get_model_params(
        est_output;
        type = "simulation"
    )
    initial_states = get_rho_beta_initial_states(
        model_params.beta_initial,
        model_params.rho_initial,
        x_vec[covariate_idx[:initial]])
    true_heat_trace = plot_single_initial(initial_states)
	add_trace!(fig, true_heat_trace, row = 1, col = 1)

    # loop through all multistarts
    for (val, idx) in enumerate(lkl_idx)
        model_params = get_model_params(
            est_output;
            type = idx
        )
        initial_states = get_rho_beta_initial_states(
            model_params.beta_initial,
            model_params.rho_initial,
            x_vec[covariate_idx[:initial]])
        heat_trace = plot_single_initial(initial_states)
		add_trace!(fig, heat_trace, row = 1, col = val + 1)
    end

	relayout!(fig, width = 700 * s, height = 500)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

    return fig
end

# transitiions
function plot_single_transition_mat(
    mat::Matrix{Float64},
)

    n_states = size(mat, 1)

    text_data = [string(round(mat[i, j], digits=2)) for i in 1:size(mat, 1), j in 1:size(mat, 2)]

    plt =
        PlotlyJS.heatmap(
			x = "To State " .* string.(collect(1:n_states)),
			y = reverse("From State " .* string.(collect(1:n_states))),
			z = mat[end:-1:1, :],
			zmin = 0.0, zmax = 1.0,

			colorscale = cscale,
        )

    return plt
end

function compare_trans_heatmaps(
    est_output::EstimationOutput;
    patient_idx = 1,
    save_dir = nothing
)
    sim_output, sim_params, sim_hyper = unpack(est_output)

    covariate_idx = convert_covariate_2_df_indices(
        sim_params.covariate_mat_headers,
        sim_hyper.covariate_tup
    )
    x_vec = sim_params.covariate_mat[patient_idx, :]
    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

    subplot_titles = ["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))" for (val, idx) in enumerate(lkl_idx)]
    subplot_titles = vcat("Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))", subplot_titles)
    s = length(est_output.fitted_log_lkl_list) + 1
    fig = PlotlyJS.make_subplots(
		rows = 7, cols = ceil(Int, s/7),
		subplot_titles = reshape(subplot_titles, 1, s)
    )

    # ------------------------------------------------------------------
    # true model
    model_params = get_model_params(
        est_output;
        type = "simulation"
    )
    transition_probs = get_rho_beta_transition_mat(
        model_params.beta_transition, model_params.rho_trans, x_vec[covariate_idx[:trans]])
    true_heat_trace = plot_single_transition_mat(transition_probs)
	add_trace!(fig, true_heat_trace, row = 1, col = 1)

    # loop through all multistarts
    for (val, idx) in enumerate(lkl_idx)
        model_params = get_model_params(
            est_output;
            type = idx
        )
        transition_probs = get_rho_beta_transition_mat(
            model_params.beta_transition, model_params.rho_trans, x_vec[covariate_idx[:trans]])
        heat_trace = plot_single_transition_mat(transition_probs)
		add_trace!(fig, heat_trace, row = floor(Int, val/ceil(Int, s/7)) + 1, col = val % ceil(Int, s/7)+1)
    end

	relayout!(fig, width = 800 * s, height = 250)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

    return fig
end

# emissions
function plot_single_bernoulli_probs(
    mat::Matrix{Float64};
	symptom_labels = nothing
)

    if isnothing(symptom_labels)
        symptom_labels = "To Symptom " .* string.(collect(1:size(mat, 2)))
    end

    plt =
        PlotlyJS.heatmap(
			x = symptom_labels,
			y = reverse("From State " .* string.(collect(1:size(mat, 1)))),
			z = mat[end:-1:1, :],
			zmin = 0.0, zmax = 1.0,
			colorscale = cscale,
        )

    return plt
end

function compare_bernoulli_heatmaps(
    est_output::EstimationOutput;
    patient_idx = 1,
    save_dir = nothing
)
    sim_output, sim_params, sim_hyper = unpack(est_output)

    covariate_idx = convert_covariate_2_df_indices(
        sim_params.covariate_mat_headers,
        sim_hyper.covariate_tup
    )
    x_vec = sim_params.covariate_mat[patient_idx, :]
    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

    subplot_titles = ["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))" for (val, idx) in enumerate(lkl_idx)]
    subplot_titles = vcat("Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))", subplot_titles)
    s = length(est_output.fitted_log_lkl_list) + 1
    fig = PlotlyJS.make_subplots(
		rows = 1, cols = s,
		subplot_titles = reshape(subplot_titles, 1, s)
    )

    # ------------------------------------------------------------------
    # true model
    model_params = get_model_params(
        est_output;
        type = "simulation"
    )
    em_states = get_bernoulli_probs(model_params.emissions.beta_bernoulli, x_vec[covariate_idx[:em]])
    true_heat_trace = plot_single_bernoulli_probs(em_states)
	add_trace!(fig, true_heat_trace, row = 1, col = 1)

    # loop through all multistarts
    for (val, idx) in enumerate(lkl_idx)
        model_params = get_model_params(
            est_output;
            type = idx
        )
        em_states = get_bernoulli_probs(model_params.emissions.beta_bernoulli, x_vec[covariate_idx[:em]])
        heat_trace = plot_single_bernoulli_probs(em_states)
		add_trace!(fig, heat_trace, row = 1, col = val + 1)
    end

	relayout!(fig, width = 700 * s, height = 500)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

    return fig
end

# gaussian emissions
function plot_single_gaussian_pdf(;
	μ::Float64, σ::Float64, x_range::Vector{Float64}, name::String, line_color = "red"
)

    # Define the normal distribution
    dist = Normal(μ, σ)
    # Compute the PDF values over the x_range
    y_values = pdf.(dist, x_range)

    trace = PlotlyJS.scatter(
		x = x_range, y = y_values, 
		mode = "lines", name = name, 
		line_color = line_color, fill="tozeroy", showlegend = false)

    return trace
end

function compare_gaussian_pdf(
    est_output::EstimationOutput;
	xlims::Vector{Float64} = [0.0, 1.0],
	ylims::Vector{Float64} = [0.0, 10.0],
    save_dir = nothing
)
    sim_output, sim_params, sim_hyper = unpack(est_output)

    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

    subplot_titles = [repeat(["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))"], sim_hyper.n_obs_tup.gaussian) for (val, idx) in enumerate(lkl_idx)]
    subplot_titles = vcat(repeat(["Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))"], sim_hyper.n_obs_tup.gaussian), subplot_titles...)
    s = (length(est_output.fitted_log_lkl_list) + 1) * sim_hyper.n_obs_tup.gaussian
    fig = PlotlyJS.make_subplots(
		rows = sim_hyper.n_states, cols = s,
		subplot_titles = reshape(subplot_titles, 1, s),
		shared_yaxes = "rows",
		shared_xaxes = "columns"
    )


    # ------------------------------------------------------------------
    # true model
    model_params = get_model_params(
        est_output;
        type = "simulation"
    )
    true_gaussian_emission = convert_gauss_2_true(model_params.emissions.beta_gaussian)
	x_range = collect(range(xlims[1], xlims[2]; length = 500))
    for i in 1:sim_hyper.n_states
        for j in 1:sim_hyper.n_obs_tup.gaussian
            # plot the true gaussian pdf
            true_heat_trace = plot_single_gaussian_pdf(;
				μ = true_gaussian_emission.means[i, j], 
				σ = true_gaussian_emission.stds[i, j], 
				x_range = x_range, 
				name = "State " * string(i)
            )
			add_trace!(fig, true_heat_trace, row = i, col = j)
        end
    end

    # loop through all multistarts
    for (val, idx) in enumerate(lkl_idx)
        model_params = get_model_params(
            est_output;
            type = idx
        )
        true_gaussian_emission = convert_gauss_2_true(model_params.emissions.beta_gaussian)
		x_range = collect(range(xlims[1], xlims[2]; length = 500))
        for i in 1:sim_hyper.n_states
            for j in 1:sim_hyper.n_obs_tup.gaussian
                # plot the true gaussian pdf
                true_heat_trace = plot_single_gaussian_pdf(;
					μ = true_gaussian_emission.means[i, j], 
					σ = true_gaussian_emission.stds[i, j], 
					x_range = x_range, 
					name = "State " * string(i)
                )
				add_trace!(fig, true_heat_trace, row = i, col = val*sim_hyper.n_obs_tup.gaussian + j)
            end
        end
    end

    # relayout!(fig; Symbol("yaxis"*"_range") => ylims)
    # for axis_idx in 1:(sim_hyper.n_states * sim_hyper.n_obs_tup.gaussian)
    #     relayout!(fig; Symbol("yaxis$(axis_idx)"*"_range") => ylims)
    # end

	relayout!(fig; Symbol("yaxis"*"_range")=> ylims)
	for axis_idx in 1:(sim_hyper.n_states * (length(lkl_idx) + 1) * sim_hyper.n_obs_tup.gaussian * 2)
        relayout!(fig; Symbol("yaxis$axis_idx"*"_range")=> ylims)
    end


	relayout!(fig, width = 600 * s, height = 500)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

    return fig
end

# rho
function rho_trace(
    variables::Vector{String},
    estimates::Vector{Float64};
	se_vals::Vector{Float64} = zeros(length(estimates))
)

    # handle p_ values
    pv = p_values(estimates, se_vals ./ 1.96)

    # Define color intensities based on p-values
    function get_color(value, p_value)
        # If not significant or the value is too close to zero, return gray.
        if p_value > 0.05 || abs(value) ≤ 0.1
            return "gray"
        end

        # Compute an intensity factor based on the p-value.
        # For p_value in [0, 0.05], we map it to an intensity in [1, 0]:
        #   p_value == 0   --> intensity = 1 (most significant → darkest)
        #   p_value == 0.05 --> intensity = 0 (least significant → lightest)
        intensity = 1 - (p_value / 0.05)

        # Define thresholds for the intensity factor.
        # You can adjust these thresholds to fine-tune the color breakpoints.
        if value > 0.1  # Positive values: use red shades.
            if intensity ≥ 0.8
                return "darkred"
            elseif intensity ≥ 0.5
                return "red"
            else
                return "lightcoral"
            end
        elseif value < -0.1  # Negative values: use green shades.
            if intensity ≥ 0.8
                return "darkgreen"
            elseif intensity ≥ 0.5
                return "green"
            else
                return "lightgreen"
            end
        else
            # For values between -0.1 and 0.1, return gray.
            return "gray"
        end
    end

    colors = [get_color(estimates[i], pv[i]) for i in 1:length(estimates)]

    trace = PlotlyJS.scatter(
		x = estimates,
		y = variables,
		mode = "markers+text",
		marker = attr(
			size = 14,
			color = colors # Conditional coloring
        ),
		error_x = attr(
			type = "data",
			symmetric = true,  # Use asymmetric error bars if needed
			array = se_vals,  # Upper error (distance from estimate to upper bound)
			color = colors
            # arrayminus = estimates .- lower_bound  # Lower error (distance from estimate to lower bound)
        )
    )

    return trace

end

function set_all_xranges!(fig, lo, hi)
    n_axes = count(k -> startswith(String(k), "xaxis"), keys(fig.plot.layout))

    for i in 1:n_axes
        if i == 1
            PlotlyJS.relayout!(
                fig;
				Symbol("xaxis") => attr(range = (lo, hi)),
            )
        else
            PlotlyJS.relayout!(
                fig;
				Symbol("xaxis$i") => attr(range = (lo, hi)),
            )
        end
    end

    return fig
end

function compare_rhos(
    est_output::EstimationOutput;
    rho_type = "initial",
    save_dir = nothing
)
    sim_output, sim_params, sim_hyper = unpack(est_output)

    lkl_idx = sortperm(est_output.fitted_log_lkl_list)

    subplot_titles = ["Estimated $idx<br>-Log-lkl: $(round(Int, est_output.fitted_log_lkl_list[idx]))" for (val, idx) in enumerate(lkl_idx)]
    subplot_titles = vcat("Simulation<br>-Log-lkl:$(round(Int, est_output.true_log_lkl))", subplot_titles)
    s = length(est_output.fitted_log_lkl_list) + 1
    fig = PlotlyJS.make_subplots(
		rows = 1, cols = s,
		subplot_titles = reshape(subplot_titles, 1, s),
		shared_yaxes = "rows",
		shared_xaxes = "columns"
    )

    # ------------------------------------------------------------------
    # true model
    model_params = get_model_params(
        est_output;
        type = "simulation"
    )
    true_heat_trace = rho_trace(
        sim_hyper.covariate_tup[rho_type == "initial" ? :initial : :trans],
        model_params[rho_type == "initial" ? :rho_initial : :rho_trans]
    )
	add_trace!(fig, true_heat_trace, row = 1, col = 1)

    # loop through all multistarts
    for (val, idx) in enumerate(lkl_idx)
        model_params = get_model_params(
            est_output;
            type = idx
        )
        heat_trace = rho_trace(
            sim_hyper.covariate_tup[rho_type == "initial" ? :initial : :trans],
            model_params[rho_type == "initial" ? :rho_initial : :rho_trans]
        )
		add_trace!(fig, heat_trace, row = 1, col = val + 1)
    end
    set_all_xranges!(fig, -2, 2)

	relayout!(fig, width = 700 * s, height = 500)

    if !isnothing(save_dir)
        open(save_dir, "w") do io
            PlotlyBase.to_html(io, fig.plot)
        end
    end

    return fig
end

# rho 
function p_values(estimates, std_errs)
    # Compute the test statistics (z-values) element-wise
    t_values = estimates ./ std_errs
    # Compute two-tailed p-values for each z-value
    return 2 .* (1 .- cdf.(Normal(0, 1), abs.(t_values)))
end

# model evaluations
# function run_simulation_from_estimation(
# 	est_output::EstimationOutput;
# 	simulation_seed = 1,
# 	T = 5,
# 	type = "best_fitted",

# )

# 	# unpack
# 	sim_output, sim_params, sim_hyper = unpack(est_output)

# 	# take in a sim_param struct and run the simulation
# 	Random.seed!(simulation_seed)	

# 	# simulate!
# 	covariate_df = DataFrame(sim_params.covariate_mat, sim_params.covariate_mat_headers)

# 	model_params = get_model_params(
#         est_output;
#         type = type
#     )
# 	states, observations = simulate(;
# 		model_params = model_params,
# 		covariate_df = covariate_df,
# 		covariate_tup = sim_params.sim_hyper.covariate_tup,
# 		T = T
# 	)

# 		# assign all vars to struct now
# 	sim_output = SimulationOutput(
# 		sim_params = sim_params,
# 		simulation_seed = simulation_seed, 

# 		states = states,
# 		observations = observations
# 	)

# 	return sim_output

# end

# # new
# _symptom_time(mat) = size(mat, 1) ≤ size(mat, 2) ? mat : permutedims(mat)

# function gaussian_means_by_symptom_time(obs)
#     # Supports either Vector{Matrix} (each: individuals × time) per symptom,
#     # or a 3D Array (individuals × time × symptoms)
#     if obs isa AbstractVector{<:AbstractMatrix}
#         S = length(obs)
#         T = size(obs[1], 2)
#         M = Array{Float64}(undef, S, T)
#         @inbounds for s in 1:S, t in 1:T
#             M[s, t] = mean(obs[s][:, t])
#         end
#         return M
#     elseif ndims(obs) == 3
#         N, T, S = size(obs)
#         M = Array{Float64}(undef, S, T)
#         @inbounds for s in 1:S, t in 1:T
#             M[s, t] = mean(@view obs[:, t, s])
#         end
#         return M
#     else
#         error("Unsupported gaussian_observations container: $(typeof(obs))")
#     end
# end

# function _subplot_titles(S, base)
#     [string(base, " (symptom ", s, ")") for s in 1:S]
# end

# # --- plotting --------------------------------------------------------------

# function create_bernoulli_proportion_plot2(estimated_props, true_props; symptom_names::Union{Nothing,Vector}=nothing)
#     est = _symptom_time(estimated_props)
#     tru = _symptom_time(true_props)
#     S, T = size(est)
#     titles = isnothing(symptom_names) ? _subplot_titles(S, "Bernoulli proportion") :
#              [string("Bernoulli proportion — ", n) for n in symptom_names]

#     plt = make_subplots(rows=S, cols=1; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=titles)

#     xs = collect(1:T)
#     for s in 1:S
#         add_trace!(plt, PlotlyJS.scatter(x=xs, y=vec(tru[s, :]), mode="lines+markers", name=(s==1 ? "True" : "True (s$s)")); row=s, col=1)
#         add_trace!(plt, PlotlyJS.scatter(x=xs, y=vec(est[s, :]), mode="lines+markers", name=(s==1 ? "Estimated" : "Estimated (s$s)")); row=s, col=1)
#         relayout!(plt, Dict("yaxis$(s)_title_text" => "prop", "xaxis$(s)_title_text" => (s==S ? "time" : "")))
#     end
#     relayout!(plt, title="Observed vs Estimated Bernoulli Proportions", legend_title_text="Series")
#     return plt
# end

# function create_gaussian_mean_plot(estimated_gauss, true_gauss; symptom_names::Union{Nothing,Vector}=nothing)
#     estM = _gaussian_means_by_symptom_time(estimated_gauss)  # S × T means
#     truM = _gaussian_means_by_symptom_time(true_gauss)       # S × T means
#     S, T = size(estM)
#     titles = isnothing(symptom_names) ? _subplot_titles(S, "Gaussian mean") :
#              [string("Gaussian mean — ", n) for n in symptom_names]

#     plt = make_subplots(rows=S, cols=1; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=titles)

#     xs = collect(1:T)
#     for s in 1:S
#         add_trace!(plt, PlotlyJS.scatter(x=xs, y=vec(truM[s, :]), mode="lines+markers", name=(s==1 ? "True mean" : "True mean (s$s)")); row=s, col=1)
#         add_trace!(plt, PlotlyJS.scatter(x=xs, y=vec(estM[s, :]), mode="lines+markers", name=(s==1 ? "Estimated mean" : "Estimated mean (s$s)")); row=s, col=1)
#         relayout!(plt, Dict("yaxis$(s)_title_text" => "mean", "xaxis$(s)_title_text" => (s==S ? "time" : "")))
#     end
#     relayout!(plt, title="Observed vs Estimated Gaussian Means", legend_title_text="Series")
#     return plt
# end

# # --- main ------------------------------------------------------------------

# function compare_estimation_2_data(
#     est_output::EstimationOutput;
#     simulation_seed = 1,
#     T = 5,
#     type = "best_fitted",
#     observation_type = "bernoulli",
#     symptom_names::Union{Nothing,Vector}=nothing
# )
#     estimated_sim_output = run_simulation_from_estimation(
#         est_output;
#         simulation_seed = simulation_seed,
#         T = T,
#         type = type
#     )

#     if observation_type == "bernoulli"
#         # compare Bernoulli observations via proportions
#         estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
#         true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)

#         fig = create_bernoulli_proportion_plot2(estimated_props, true_props; symptom_names = symptom_names)

#     elseif observation_type == "gaussian"
#         # compare Gaussian observations via per-time means (one subplot per symptom)
#         est_gauss  = estimated_sim_output.observations.gaussian_observations
#         true_gauss = est_output.sim_output.observations.gaussian_observations

#         fig = create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names=symptom_names)

#     else
#         error("Unknown observation type: $observation_type")
#     end

#     display(fig)
#     return estimated_sim_output, fig
# end

function run_simulation_from_estimation(
    est_output::EstimationOutput;
    simulation_seed::Integer = 1,
    T::Integer = 5,
    type::AbstractString = "best_fitted",
)
    # unpack (keep names explicit for clarity)
    sim_output, sim_params, sim_hyper = unpack(est_output)

    # seed for reproducibility
    Random.seed!(simulation_seed)

    # covariates to DataFrame (assumes headers align with columns)
    covariate_df = DataFrame(sim_params.covariate_mat, sim_params.covariate_mat_headers)

    # pull model params from estimation
    model_params = get_model_params(est_output; type = type)

    # simulate!
    states, observations = simulate(
        ; model_params = model_params,
          covariate_df = covariate_df,
          covariate_tup = sim_params.sim_hyper.covariate_tup, # uses sim_params' hyper
          T = T
    )

    # package results
    return SimulationOutput(
        sim_params = sim_params,
        simulation_seed = simulation_seed,
        states = states,
        observations = observations,
    )
end

# --- helpers -----------------------------------------------------------------

# Ensure matrices are S × T (symptom × time)
_symptom_time(mat::AbstractMatrix) = size(mat, 1) ≤ size(mat, 2) ? mat : permutedims(mat)

# Compute per-time means for Gaussian observations across individuals.
# Accepts either Vector{Matrix} with (N × T) per symptom, or a 3D Array (N × T × S).
function gaussian_means_by_symptom_time(obs)
    if obs isa AbstractVector{<:AbstractMatrix}
        S = length(obs)
        @assert S > 0 "Empty gaussian observation vector"
        N, T = size(obs[1])
        @assert all(size(M) == (N, T) for M in obs) "All symptom matrices must be N×T"
        M = Array{Float64}(undef, S, T)
        @inbounds for s in 1:S, t in 1:T
            M[s, t] = mean(obs[s][:, t])
        end
        return M
    elseif ndims(obs) == 3
        N, T, S = size(obs)
        M = Array{Float64}(undef, S, T)
        @inbounds for s in 1:S, t in 1:T
            M[s, t] = mean(@view obs[:, t, s])
        end
        return M
    else
        error("Unsupported gaussian_observations container: $(typeof(obs))")
    end
end

_subplot_titles(S::Integer, base::AbstractString) =
    [string(base, " (symptom ", s, ")") for s in 1:S]

# --- plotting ---------------------------------------------------------------
function get_binary_proportions(obs; skip_missing::Bool = true)
    if obs isa AbstractVector{<:AbstractMatrix}
        obs = [replace(M, -1 => missing) for M in obs]
    elseif ndims(obs) == 3
        obs = replace(obs, -1 => missing)
    end
    proportion(v) = skip_missing ? mean(skipmissing(v)) : mean(v)

    if obs isa AbstractVector{<:AbstractMatrix}
        S = length(obs)
        @assert S > 0 "Empty bernoulli observation vector"
        N, T = size(obs[1])
        @assert all(size(M) == (N, T) for M in obs) "All symptom matrices must be N×T"

        P = Array{Float64}(undef, S, T)
        @inbounds for s in 1:S, t in 1:T
            P[s, t] = proportion(@view obs[s][:, t])
        end
        return P

    elseif ndims(obs) == 3
        N, T, S = size(obs)
        P = Array{Float64}(undef, S, T)
        @inbounds for s in 1:S, t in 1:T
            P[s, t] = proportion(@view obs[:, t, s])
        end
        return P

    else
        error("Unsupported bernoulli_observations container: $(typeof(obs))")
    end
end

function create_bernoulli_proportion_plot2(
    estimated_props::AbstractMatrix,
    true_props::AbstractMatrix;
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    est = estimated_props #_symptom_time(estimated_props)
    tru = true_props # _symptom_time(true_props)
    @assert size(est) == size(tru) "Estimated and true proportion arrays must have same size"
    S, T = size(est)

    titles = isnothing(symptom_names) ? _subplot_titles(S, "Bernoulli proportion") :
             symptom_names#[string("Bernoulli proportion — ", n) for n in symptom_names]
    print(length([string(t) for t in titles]),  " S ", S)
    titles = reshape([string(t) for t in titles], S, 1)
    
    nrows = cld(S, 3)
    plt = make_subplots(rows=nrows, cols=3; shared_xaxes=true, vertical_spacing=0.02, horizontal_spacing=0.02, subplot_titles=titles)
    xs = collect(1:T)

    for s in 1:S
        # Compute row and column for 3x3 subplot grid
        row = div(s - 1, 3) + 1
        col = mod(s - 1, 3) + 1

        add_trace!(plt, PlotlyJS.scatter(
                x=xs, y=vec(tru[s, :]),
                mode="lines+markers",
                name=(s == 1 ? "True" : "True ($(titles[s]))"),
                showlegend=(s == 1 ? true : false),
                line=attr(color="blue")
            ); row=row, col=col)
        # add_trace!(plt, PlotlyJS.scatter(
        #         x=[xs[2], xs[4]],
        #         y=[vec(tru[s, :])[2], vec(tru[s, :])[4]],
        #         mode="lines+markers",
        #         showlegend=false,
        #         line=attr(color="blue", dash="dash")
        #     ); row=row, col=col)
        add_trace!(plt, PlotlyJS.scatter(x=xs, y=vec(est[s, :]), mode="lines+markers", name=(s == 1 ? "Estimated" : "Estimated ($(titles[s]))"), showlegend=(s == 1 ? true : false), line=attr(color="red")); row=row, col=col)
        relayout!(plt;
            Symbol("yaxis$(s)") => attr(title="Percentage Frequency"),
            Symbol("xaxis$(s)") => attr(title=(s == S ? "time" : ""))  # only bottom row gets x-title
        )
    end

    relayout!(plt, title="Observed vs Estimated Bernoulli Proportions",
        legend_title_text="Series", height=300 + 200 * nrows, width=2000)
    return plt
end

function create_bernoulli_proportion_plot_fup(
    estimated_props::AbstractMatrix,
    true_props::AbstractMatrix;
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    est = estimated_props #_symptom_time(estimated_props)
    tru = true_props # _symptom_time(true_props)
    @assert size(est) == size(tru) "Estimated and true proportion arrays must have same size"
    S, T = size(est)

    titles = isnothing(symptom_names) ? _subplot_titles(S, "Bernoulli proportion") :
             symptom_names#[string("Bernoulli proportion — ", n) for n in symptom_names]
    titles = reshape([string(t) for t in titles], S, 1)

    nrows = cld(S, 3)
    plt = make_subplots(rows=nrows, cols=3; shared_xaxes=true, vertical_spacing=0.02, horizontal_spacing=0.02, subplot_titles=titles)
    xs = collect(1:T)

    for s in 1:S
        # Compute row and column for 3x3 subplot grid
        row = div(s - 1, 3) + 1
        col = mod(s - 1, 3) + 1

        add_trace!(plt, PlotlyJS.scatter(
                x=xs, y=vec(tru[s, :]),
                mode="lines+markers",
                name=(s == 1 ? "True" : "True ($(titles[s]))"),
                showlegend=(s == 1 ? true : false),
                line=attr(color="blue")
            ); row=row, col=col)
        # for i in 1:floor(Int, T/2 -1)
        #     add_trace!(plt, PlotlyJS.scatter(
        #             x=[xs[2*i], xs[2*i+2]],
        #             y=[vec(tru[s, :])[2*i], vec(tru[s, :])[2*i+2]],
        #             mode="lines+markers",
        #             showlegend=false,
        #             line=attr(color="blue", dash="dash")
        #         ); row=row, col=col)
        # end
        add_trace!(plt, PlotlyJS.scatter(x=xs, y=vec(est[s, :]), mode="lines+markers", name=(s == 1 ? "Estimated" : "Estimated ($(titles[s]))"), showlegend=(s == 1 ? true : false), line=attr(color="red")); row=row, col=col)
        relayout!(plt;
            Symbol("yaxis$(s)") => attr(title="Percentage Frequency"),
            Symbol("xaxis$(s)") => attr(title=(s == S ? "time" : ""))  # only bottom row gets x-title
        )
    end

    relayout!(plt, title="Observed vs Estimated Bernoulli Proportions",
        legend_title_text="Series", height=300 + 200 * nrows, width=2000)
    return plt
end


function create_bernoulli_proportion_plot_monthly(
    estimated_props::AbstractMatrix,
    true_props::AbstractMatrix;
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    est = estimated_props #_symptom_time(estimated_props)
    tru = true_props # _symptom_time(true_props)
    @assert size(est) == size(tru) "Estimated and true proportion arrays must have same size"
    S, T = size(est)

    titles = isnothing(symptom_names) ? _subplot_titles(S, "Bernoulli proportion") :
             symptom_names#[string("Bernoulli proportion — ", n) for n in symptom_names]
    titles = reshape([string(t) for t in titles], S, 1)

    nrows = cld(S, 3)
    plt = make_subplots(rows=nrows, cols=3; shared_xaxes=true, vertical_spacing=0.02, horizontal_spacing=0.02, subplot_titles=titles)
    xs = collect(1:T)

    for s in 1:S
        # Compute row and column for 3x3 subplot grid
        row = div(s - 1, 3) + 1
        col = mod(s - 1, 3) + 1

        add_trace!(plt, PlotlyJS.scatter(
                x=xs, y=vec(tru[s, :]),
                mode="lines+markers",
                name=(s == 1 ? "True" : "True ($(titles[s]))"),
                showlegend=(s == 1 ? true : false),
                line=attr(color="blue")
            ); row=row, col=col)
        add_trace!(plt, PlotlyJS.scatter(x=xs, y=vec(est[s, :]), mode="lines+markers", name=(s == 1 ? "Estimated" : "Estimated ($(titles[s]))"), showlegend=(s == 1 ? true : false), line=attr(color="red")); row=row, col=col)
        relayout!(plt;
            Symbol("yaxis$(s)") => attr(title="Percentage Frequency"),
            Symbol("xaxis$(s)") => attr(title=(s == S ? "time" : ""))  # only bottom row gets x-title
        )
    end

    relayout!(plt, title="Observed vs Estimated Bernoulli Proportions",
        legend_title_text="Series", height=300 + 200 * nrows, width=2000)
    return plt
end

function create_gaussian_mean_plot(
    estimated_gauss,
    true_gauss;
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    estM = gaussian_means_by_symptom_time(estimated_gauss)  # S × T
    truM = gaussian_means_by_symptom_time(true_gauss)       # S × T
    @assert size(estM) == size(truM) "Estimated and true Gaussian mean arrays must have same size"
    S, T = size(estM)

    titles = isnothing(symptom_names) ? _subplot_titles(S, "Gaussian mean") :
             [string("Gaussian mean — ", n) for n in symptom_names]
    titles = reshape([string(t) for t in titles], S, 1)

    plt = make_subplots(rows=S, cols=1; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=titles)
    xs = collect(1:T)

    for s in 1:S
        add_trace!(plt, PlotlyJS.scatter(x=xs, y=vec(truM[s, :]), mode="lines+markers", name=(s==1 ? "True mean" : "True mean (s$s)")); row=s, col=1)
        add_trace!(plt, PlotlyJS.scatter(x=xs, y=vec(estM[s, :]), mode="lines+markers", name=(s==1 ? "Estimated mean" : "Estimated mean (s$s)")); row=s, col=1)
        relayout!(plt;
			Symbol("yaxis$(s)") => attr(title = "Mean"),
			Symbol("xaxis$(s)") => attr(title = (s == S ? "time" : ""))  # only bottom row gets x-title
        )
    end

    relayout!(plt, title="Observed vs Estimated Gaussian Means",
        legend_title_text="Series")
    return plt
end

# --- main -------------------------------------------------------------------

function compare_estimation_2_data(
    est_output::EstimationOutput;
    simulation_seed::Integer = 1,
    T::Integer = 5,
    type::AbstractString = "best_fitted",
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    estimated_sim_output = run_simulation_from_estimation(
        est_output;
        simulation_seed = simulation_seed,
        T = T,
        type = type
    )

    fig = if observation_type == "bernoulli"
        # proportions over time (S × T expected after _symptom_time)
        estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
        true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
        create_bernoulli_proportion_plot2(estimated_props, true_props; symptom_names = symptom_names)

    elseif observation_type == "gaussian"
        # per-time means (one subplot per symptom)
        est_gauss  = estimated_sim_output.observations.gaussian_observations
        true_gauss = est_output.sim_output.observations.gaussian_observations
        create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

    else
        error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
    end

    return fig
end


function compare_estimation_2_data_fup(
    est_output::EstimationOutput;
    simulation_seed::Integer = 1,
    T::Integer = 5,
    type::AbstractString = "best_fitted",
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    estimated_sim_output = run_simulation_from_estimation(
        est_output;
        simulation_seed = simulation_seed,
        T = T,
        type = type
    )

    fig = if observation_type == "bernoulli"
        # proportions over time (S × T expected after _symptom_time)
        estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
        true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
        create_bernoulli_proportion_plot_fup(estimated_props, true_props; symptom_names = symptom_names)

    elseif observation_type == "gaussian"
        # per-time means (one subplot per symptom)
        est_gauss  = estimated_sim_output.observations.gaussian_observations
        true_gauss = est_output.sim_output.observations.gaussian_observations
        create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

    else
        error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
    end

    return fig
end

function summarize_sim_plots(sim_plots; q=0.05)
    S = length(sim_plots)

    T = length(sim_plots[1][1])  # assume consistent

    median_props = zeros(S, T)
    lower_props  = zeros(S, T)
    upper_props  = zeros(S, T)

    for s in 1:S
        runs = sim_plots[s]  # Vector of vectors

        for t in 1:T
            vals = [run[t] for run in runs]

            median_props[s, t] = median(vals)
            lower_props[s, t]  = quantile(vals, q/2)
            upper_props[s, t]  = quantile(vals, 1 - q/2)
        end
    end

    return median_props, lower_props, upper_props
end


function compare_estimation_2_data_with_CIs(
    est_output::EstimationOutput;
    T::Integer = 5,
    type::AbstractString = "best_fitted",
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    # plt = make_subplots(rows=4, cols=3; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=symptom_names)
    sim_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    true_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    for simulation_seed in 1:10
        estimated_sim_output = run_simulation_from_estimation(
            est_output;
            simulation_seed = simulation_seed,
            T = T,
            type = type
        )

        fig = if observation_type == "bernoulli"
            # proportions over time (S × T expected after _symptom_time)
            estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
            true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
            create_bernoulli_proportion_plot2(estimated_props, true_props; symptom_names = symptom_names)

        elseif observation_type == "gaussian"
            # per-time means (one subplot per symptom)
            est_gauss  = estimated_sim_output.observations.gaussian_observations
            true_gauss = est_output.sim_output.observations.gaussian_observations
            create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

        else
            error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
        end
        for symptom in 1:length(symptom_names)
            push!(sim_plots[symptom], fig.plot.data[3*symptom][:y])
            if simulation_seed == 1
                push!(true_plots[symptom], fig.plot.data[3*symptom - 2][:y])
            end
        end
    end
    true_props_mat = reduce(vcat, [tp[1]' for tp in true_plots])

    median, lower, upper = summarize_sim_plots(sim_plots; q=0.05)

    fig = plot_median_and_CI(median, lower, upper, true_props_mat; symptom_names = symptom_names)

    return fig

end

function compare_estimation_2_data_with_CIs_fup(
    est_output::EstimationOutput;
    T::Integer = 5,
    type::AbstractString = "best_fitted",
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    # plt = make_subplots(rows=4, cols=3; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=symptom_names)
    sim_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    true_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    for simulation_seed in 1:10
        estimated_sim_output = run_simulation_from_estimation(
            est_output;
            simulation_seed = simulation_seed,
            T = T,
            type = type
        )

        fig = if observation_type == "bernoulli"
            # proportions over time (S × T expected after _symptom_time)
            estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
            true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
            create_bernoulli_proportion_plot_fup(estimated_props, true_props; symptom_names = symptom_names)

        elseif observation_type == "gaussian"
            # per-time means (one subplot per symptom)
            est_gauss  = estimated_sim_output.observations.gaussian_observations
            true_gauss = est_output.sim_output.observations.gaussian_observations
            create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

        else
            error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
        end
        for symptom in 1:length(symptom_names)
            push!(sim_plots[symptom], fig.plot.data[6*symptom][:y])
            if simulation_seed == 1
                push!(true_plots[symptom], fig.plot.data[6*symptom - 5][:y])
            end
        end
    end
    true_props_mat = reduce(vcat, [tp[1]' for tp in true_plots])

    median, lower, upper = summarize_sim_plots(sim_plots; q=0.05)

    fig = plot_median_and_CI(median, lower, upper, true_props_mat; symptom_names = symptom_names)

    return fig

end

function compare_estimation_2_data_with_CIs_monthly(
    est_output::EstimationOutput;
    T::Integer = 5,
    type::AbstractString = "best_fitted",
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing
)
    sim_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    true_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    n_symptoms = length(symptom_names)
    for simulation_seed in 1:10
        estimated_sim_output = run_simulation_from_estimation(
            est_output;
            simulation_seed = simulation_seed,
            T = T,
            type = type
        )

        fig = if observation_type == "bernoulli"
            # proportions over time (S × T expected after _symptom_time)
            estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
            true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
            create_bernoulli_proportion_plot_monthly(estimated_props, true_props; symptom_names = symptom_names)

        elseif observation_type == "gaussian"
            # per-time means (one subplot per symptom)
            est_gauss  = estimated_sim_output.observations.gaussian_observations
            true_gauss = est_output.sim_output.observations.gaussian_observations
            create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

        else
            error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
        end
        for symptom in 1:n_symptoms
            push!(sim_plots[symptom], fig.plot.data[2*symptom][:y])
            if simulation_seed == 1
                push!(true_plots[symptom], fig.plot.data[2*symptom - 1][:y])
            end
        end
    end
    
    true_props_mat = reduce(vcat, [tp[1]' for tp in true_plots])

    median, lower, upper = summarize_sim_plots(sim_plots; q=0.05)

    fig = plot_median_and_CI(median, lower, upper, true_props_mat; symptom_names = symptom_names)

    return fig

end

function plot_median_and_CI(
    median_props::AbstractMatrix,
    lower_props::AbstractMatrix,
    upper_props::AbstractMatrix,
    true_props::AbstractMatrix;
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing,
    quantile::Float64 = 0.05
)
    @assert size(median_props) == size(true_props)
    @assert size(lower_props) == size(true_props)
    @assert size(upper_props) == size(true_props)

    S, T = size(true_props)

    titles = isnothing(symptom_names) ?
        ["Bernoulli proportion $s" for s in 1:S] :
        symptom_names

    titles = reshape(string.(titles), S, 1)

    nrows = cld(S, 3)
    plt = make_subplots(
        rows=nrows,
        cols=3;
        shared_xaxes=true,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
        subplot_titles=titles
    )

    xs = collect(1:T)

    for s in 1:S
        row = div(s - 1, 3) + 1
        col = mod(s - 1, 3) + 1

        # upper CI
        add_trace!(plt, PlotlyJS.scatter(
            x=xs,
            y=vec(upper_props[s, :]),
            mode="lines",
            line=attr(width=0),
            showlegend=false,
            name="Upper CI"
        ); row=row, col=col)

        # lower CI + shaded fill
        add_trace!(plt, PlotlyJS.scatter(
            x=xs,
            y=vec(lower_props[s, :]),
            mode="lines",
            line=attr(width=0),
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.2)",
            name=(s == 1 ? string((1-quantile)*100) * "% CI" : ""),
            showlegend=(s == 1)
        ); row=row, col=col)

        # median estimate
        add_trace!(plt, PlotlyJS.scatter(
            x=xs,
            y=vec(median_props[s, :]),
            mode="lines+markers",
            line=attr(color="red"),
            name=(s == 1 ? "Median estimate" : ""),
            showlegend=(s == 1)
        ); row=row, col=col)

        # true values
        add_trace!(plt, PlotlyJS.scatter(
            x=xs,
            y=vec(true_props[s, :]),
            mode="lines+markers",
            line=attr(color="blue"),
            name=(s == 1 ? "True" : ""),
            showlegend=(s == 1)
        ); row=row, col=col)

        relayout!(plt;
            Symbol("yaxis$(s)") => attr(title="Percentage Frequency"),
            Symbol("xaxis$(s)") => attr(title=(row == nrows ? "time" : ""))
        )
    end

    relayout!(plt,
        title="Observed vs Estimated Bernoulli Proportions",
        legend_title_text="Series",
        height=300 + 200 * nrows,
        width=2000
    )

    return plt
end

function plot_transition_matrices(beta_transition::Array{<:Real}, range::AbstractRange{Float64})
    
    n = length(range)  # Anzahl der Steps
    
    # z.B. cols fix, rows berechnen
    cols = 3
    rows = ceil(Int, n / cols)
    
    fig_trans_covs = PlotlyJS.make_subplots(rows=rows, cols=cols, horizontal_spacing=0.06, vertical_spacing=0.06)

    for (i, sig) in enumerate(range)
        trans_mat = get_transition_mat(beta_transition, [1.0, sig])
        trace = plot_single_transition_mat(trans_mat)
        row = div(i - 1, 3) + 1
        col = mod(i - 1, 3) + 1
        add_trace!(fig_trans_covs, trace, row=row, col=col)
    end
    relayout!(fig_trans_covs, title="Initial State Probabilities for Different Covariate Values", height=300 + 200 * rows, width=2000)

    return fig_trans_covs
end

function plot_initial_probs(beta_initial::Array{<:Real}, range::AbstractRange{Float64})
    n = length(range)  # Anzahl der Steps
    
    # z.B. cols fix, rows berechnen
    cols = 3
    rows = ceil(Int, n / cols)
    
    fig_initials_covs = PlotlyJS.make_subplots(
        rows=rows, cols=cols, horizontal_spacing=0.1, vertical_spacing=0.1
    )

    for (i, sig) in enumerate(range)
        initial_states = multinom_reg(beta_initial, [1.0, sig])
        trace = plot_single_initial(initial_states)
        row = div(i - 1, 3) + 1
        col = mod(i - 1, 3) + 1
        add_trace!(fig_initials_covs, trace, row=row, col=col)
    end
    relayout!(fig_initials_covs, title="Initial State Probabilities for Different Covariate Values", height=300 + 200 * rows, width=1000)

    return fig_initials_covs
end

function extract_rtrans_range(rho_vec, x_vec)
    aggr_vals = Float64[]
    for patient in axes(x_vec, 1)
        aggr_val = get_aggr_covars(rho_vec, x_vec[patient, :])
        push!(aggr_vals, aggr_val)
    end
    lo = minimum(aggr_vals)#floor(minimum(aggr_vals))
    hi = maximum(aggr_vals)#ceil(maximum(aggr_vals))
    step = (hi - lo )/5
    println("Aggregated covariate values range from $lo to $hi")
    return lo:step:hi
end

function extract_rtrans_hist(rho_vec, x_vec)
    aggr_vals = Float64[]
    for patient in axes(x_vec, 1)
        aggr_val = get_aggr_covars(rho_vec, x_vec[patient, :])
        push!(aggr_vals, aggr_val)
    end
    print(countmap(aggr_vals))
    return countmap(aggr_vals)
end

function ensemble_uq_fup(
    est_output::EstimationOutput;
    T::Integer = 5,
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing,
    num_multistarts::Integer = 10
)

    lkl_idx = sortperm(est_output.fitted_log_lkl_list)
    # plt = make_subplots(rows=4, cols=3; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=symptom_names)
    sim_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    true_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    for multistart in 1:num_multistarts
        
        for simulation_seed in 1:10
            estimated_sim_output = run_simulation_from_model_params(
                est_output;
                simulation_seed = simulation_seed,
                T = T,
                model_params=est_output.fitted_model_params[lkl_idx][multistart]
            )

            fig = if observation_type == "bernoulli"
                # proportions over time (S × T expected after _symptom_time)
                estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
                true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
                create_bernoulli_proportion_plot_fup(estimated_props, true_props; symptom_names = symptom_names)

            elseif observation_type == "gaussian"
                # per-time means (one subplot per symptom)
                est_gauss  = estimated_sim_output.observations.gaussian_observations
                true_gauss = est_output.sim_output.observations.gaussian_observations
                create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

            else
                error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
            end
            for symptom in 1:length(symptom_names)
                push!(sim_plots[symptom], fig.plot.data[6*symptom][:y])
                if simulation_seed == 1
                    push!(true_plots[symptom], fig.plot.data[6*symptom - 5][:y])
                end
            end
        end
    end
    true_props_mat = reduce(vcat, [tp[1]' for tp in true_plots])

    median, lower, upper = summarize_sim_plots(sim_plots; q=0.05)

    fig = plot_median_and_CI(median, lower, upper, true_props_mat; symptom_names = symptom_names)

    return fig

end

function ensemble_uq_monthly(
    est_output::EstimationOutput;
    T::Integer = 5,
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing,
    num_multistarts::Integer = 10,
    quantile::Float64 = 0.05
)

    lkl_idx = sortperm(est_output.fitted_log_lkl_list)
    # plt = make_subplots(rows=4, cols=3; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=symptom_names)
    sim_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    true_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    for multistart in 1:num_multistarts
        
        for simulation_seed in 1:10
            estimated_sim_output = run_simulation_from_model_params(
                est_output;
                simulation_seed = simulation_seed,
                T = T,
                model_params=est_output.fitted_model_params[lkl_idx][multistart]
            )

            fig = if observation_type == "bernoulli"
                # proportions over time (S × T expected after _symptom_time)
                estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
                true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
                create_bernoulli_proportion_plot_monthly(estimated_props, true_props; symptom_names = symptom_names)

            elseif observation_type == "gaussian"
                # per-time means (one subplot per symptom)
                est_gauss  = estimated_sim_output.observations.gaussian_observations
                true_gauss = est_output.sim_output.observations.gaussian_observations
                create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

            else
                error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
            end
            for symptom in 1:length(symptom_names)
                push!(sim_plots[symptom], fig.plot.data[2*symptom][:y])
                if simulation_seed == 1
                    push!(true_plots[symptom], fig.plot.data[2*symptom - 1][:y])
                end
            end
        end
    end
    true_props_mat = reduce(vcat, [tp[1]' for tp in true_plots])

    median, lower, upper = summarize_sim_plots(sim_plots; q=quantile)

    fig = plot_median_and_CI(median, lower, upper, true_props_mat; symptom_names = symptom_names, quantile=quantile)

    return fig

end

function ensemble_uq_monthly_startingparams(
    est_output::EstimationOutput;
    T::Integer = 5,
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing,
    num_multistarts::Integer = 10,
    quantile::Float64 = 0.05
)

    lkl_idx = sortperm(est_output.fitted_log_lkl_list)
    # plt = make_subplots(rows=4, cols=3; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=symptom_names)
    sim_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    true_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    for multistart in 1:num_multistarts
        
        for simulation_seed in 1:10
            estimated_sim_output = run_simulation_from_model_params(
                est_output;
                simulation_seed = simulation_seed,
                T = T,
                model_params=est_output.starting_model_params[lkl_idx][multistart]
            )

            fig = if observation_type == "bernoulli"
                # proportions over time (S × T expected after _symptom_time)
                estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
                true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
                create_bernoulli_proportion_plot_monthly(estimated_props, true_props; symptom_names = symptom_names)

            elseif observation_type == "gaussian"
                # per-time means (one subplot per symptom)
                est_gauss  = estimated_sim_output.observations.gaussian_observations
                true_gauss = est_output.sim_output.observations.gaussian_observations
                create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

            else
                error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
            end
            for symptom in 1:length(symptom_names)
                push!(sim_plots[symptom], fig.plot.data[2*symptom][:y])
                if simulation_seed == 1
                    push!(true_plots[symptom], fig.plot.data[2*symptom - 1][:y])
                end
            end
        end
    end
    true_props_mat = reduce(vcat, [tp[1]' for tp in true_plots])

    median, lower, upper = summarize_sim_plots(sim_plots; q=quantile)

    fig = plot_median_and_CI(median, lower, upper, true_props_mat; symptom_names = symptom_names, quantile=quantile)

    return fig

end

function ensemble_uq(
    est_output::EstimationOutput;
    T::Integer = 5,
    observation_type::AbstractString = "bernoulli",
    symptom_names::Union{Nothing,Vector{<:AbstractString}} = nothing,
    num_multistarts::Integer = 10
)

    lkl_idx = sortperm(est_output.fitted_log_lkl_list)
    # plt = make_subplots(rows=4, cols=3; shared_xaxes=true, vertical_spacing=0.06, subplot_titles=symptom_names)
    sim_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    true_plots = [Vector{Any}() for _ in 1:length(symptom_names)]
    for multistart in 1:num_multistarts
        
        for simulation_seed in 1:10
            estimated_sim_output = run_simulation_from_model_params(
                est_output;
                simulation_seed = simulation_seed,
                T = T,
                model_params=est_output.fitted_model_params[lkl_idx][multistart]
            )

            fig = if observation_type == "bernoulli"
                # proportions over time (S × T expected after _symptom_time)
                estimated_props = get_binary_proportions(estimated_sim_output.observations.bernoulli_observations)
                true_props      = get_binary_proportions(est_output.sim_output.observations.bernoulli_observations)
                create_bernoulli_proportion_plot2(estimated_props, true_props; symptom_names = symptom_names)

            elseif observation_type == "gaussian"
                # per-time means (one subplot per symptom)
                est_gauss  = estimated_sim_output.observations.gaussian_observations
                true_gauss = est_output.sim_output.observations.gaussian_observations
                create_gaussian_mean_plot(est_gauss, true_gauss; symptom_names = symptom_names)

            else
                error("Unknown observation type: $observation_type (use \"bernoulli\" or \"gaussian\")")
            end
            for symptom in 1:length(symptom_names)
                push!(sim_plots[symptom], fig.plot.data[3*symptom][:y])
                if simulation_seed == 1
                    push!(true_plots[symptom], fig.plot.data[3*symptom - 2][:y])
                end
            end
        end
    end
    true_props_mat = reduce(vcat, [tp[1]' for tp in true_plots])

    median, lower, upper = summarize_sim_plots(sim_plots; q=0.05)

    fig = plot_median_and_CI(median, lower, upper, true_props_mat; symptom_names = symptom_names)

    return fig

end

function run_simulation_from_model_params(
    est_output::EstimationOutput;
    simulation_seed::Integer = 1,
    T::Integer = 5,
    model_params::NamedTuple = [],
)
    # unpack (keep names explicit for clarity)
    sim_output, sim_params, sim_hyper = unpack(est_output)

    # seed for reproducibility
    Random.seed!(simulation_seed)

    # covariates to DataFrame (assumes headers align with columns)
    covariate_df = DataFrame(sim_params.covariate_mat, sim_params.covariate_mat_headers)

    # simulate!
    states, observations = simulate(
        ; model_params = model_params,
          covariate_df = covariate_df,
          covariate_tup = sim_params.sim_hyper.covariate_tup, # uses sim_params' hyper
          T = T
    )

    # package results
    return SimulationOutput(
        sim_params = sim_params,
        simulation_seed = simulation_seed,
        states = states,
        observations = observations,
    )
end

function waterfall_trace(
    lkl_list::Vector{Float64},
    hline::Float64;
    include_sim=false,
    limit_reached=nothing
)
    # Sort the likelihood list for the waterfall
    idx = sortperm(lkl_list)
    sorted_lkl = lkl_list[idx]
    run_list = 1:length(lkl_list)

    if !isnothing(limit_reached)
        colors = [value == 1 ? "red" : "green" for value in limit_reached]
        colors = colors[idx]
    else
        colors = "blue"
    end
    # Create the scatter trace for the likelihoods
    scatter_trace = PlotlyJS.scatter(
        x=run_list,
        y=sorted_lkl,
        mode="markers",
        name="Runs",
        marker=attr(
            size=10,
            color=colors # Conditional coloring
        )
    )

    # If include_sim is true, add a horizontal line trace for the simulation likelihood
    hline_trace = nothing
    if include_sim
        hline_trace = PlotlyJS.scatter(
            x=[minimum(run_list), maximum(run_list)],
            y=[hline, hline],
            mode="lines",
            name="Simulation -Log Lkl",
            line=attr(color="red", dash="dash")
        )
    end

    # Return the traces
    return include_sim ? [scatter_trace, hline_trace] : [scatter_trace]
end

function waterfall_trace(
    est_output::EstimationOutput;
    include_sim=false
)

    traces = waterfall_trace(
        est_output.fitted_log_lkl_list,
        est_output.true_log_lkl;
        include_sim=include_sim,
        limit_reached=est_output.iteration_limit_reached
    )

    return traces
end

function waterfall_plot(
    est_output::EstimationOutput;
    include_sim=false
)

    traces = waterfall_trace(est_output; include_sim=include_sim)

    # Plot the traces
    layout = Layout(
        title="Waterfall Plot",
        xaxis=attr(title="Run No:"),
        yaxis=attr(title="-Log Likelihood")
    )

    plt = PlotlyJS.plot(traces, layout)

    return plt
end


function estimated_states_over_time(
    est_output::EstimationOutput;
    simulation_seed::Integer = 1,
    T::Integer = 5,
    type::AbstractString = "best_fitted",
    n_states::Integer = 3
)
    estimated_sim_output = run_simulation_from_estimation(
        est_output;
        simulation_seed = simulation_seed,
        T = T,
        type = type
    )

    M = zeros(Int, T, n_states)
    
    for timepoint in 1:T
        cm = countmap(estimated_sim_output.states[:, timepoint])
        for j in 1:n_states
            M[timepoint, j] = get(cm, j, 0)
        end
    end
    
    fig = areaplot(1:T, M,
        labels = permutedims(["State " * string(j) for j in 1:n_states]),
        xlabel = "Timepoint",
        legend = :outerright)

    return fig
end

function savefig_auto(fig, path; kwargs...)
    layout = fig isa PlotlyJS.SyncPlot ? fig.plot.layout : fig.layout
    w = get(layout.fields, :width, 700)   # Fallback falls nicht gesetzt
    h = get(layout.fields, :height, 500)
    PlotlyJS.savefig(fig, path; width=w, height=h, kwargs...)
end