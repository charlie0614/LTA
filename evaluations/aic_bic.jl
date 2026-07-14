include("src/LTA.jl")


n_state_list = collect(4:8)
sim_no_list = string.(n_state_list) .* "states_age_wave_covs"

save_dr_list = "estimates/" .* sim_no_list .* "/results/estimates"
est_output_list = [load(save_dr * "/est_output.jld2", "estimation_output") for save_dr in save_dr_list]

lkl_list = [minimum(est_output.fitted_log_lkl_list) for est_output in est_output_list]


function get_AIC(est_output::LTA.EstimationOutput)

    k = length(est_output.fitted_model_params[1])
    aic = 2 * k + minimum(est_output.fitted_log_lkl_list)

    return aic
end

function get_BIC(est_output::LTA.EstimationOutput)

    k = length(est_output.fitted_model_params[1])
    N = est_output.sim_output.sim_params.sim_hyper.N
    bic = k * log(N) + minimum(est_output.fitted_log_lkl_list)

    return bic
end

function plot_aic_bic(
    aic_values::Vector{Float64}, 
    bic_values::Vector{Float64};
    labels =  string.(1:length(aic_values))
    )
    if length(aic_values) != length(bic_values)
        error("The lengths of AIC and BIC values must be the same.")
    end

    # Determine the y-axis limits
    all_values = vcat(aic_values, bic_values)
    min_val = minimum(all_values)
    max_val = maximum(all_values)
    ylims = (min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))  # Add some padding for better visibility

        
    # Create the bar chart
    tr_aic = PlotlyJS.bar(x = labels, y = aic_values, name = "AIC", width = 0.35)
    tr_bic = PlotlyJS.bar(x = labels, y = bic_values, name = "BIC", width = 0.35)

    plt = PlotlyJS.Plot(
        [tr_aic, tr_bic],
        PlotlyJS.Layout(
            barmode = "group",
            title = "AIC and BIC Values for Different Models",
            xaxis = PlotlyJS.attr(title = "Models"),
            yaxis = PlotlyJS.attr(title = "Value", range = [ylims[1], ylims[2]]),
            legend = PlotlyJS.attr(x = 0.95, y = 0.95),
            width = 800,
            height = 500,
            bargroupgap = 0.12
        )
    )

    return plt
end


aic_val_list = get_AIC.(est_output_list)
bic_val_list = get_BIC.(est_output_list)

aic_bic_plt = plot_aic_bic(
    aic_val_list, 
    bic_val_list;
    labels =  ["$i States" for i in n_state_list]
)

PlotlyJS.savefig(aic_bic_plt, "aic_bic_age_wave_covs.png", width=1200, height=800)