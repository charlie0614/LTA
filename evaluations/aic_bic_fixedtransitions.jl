include("../src/LTA.jl")

using JLD2
using PlotlyJS


n_state_list = collect(3:3)
sim_no_list = string.(n_state_list) .* "states/age_sex_comorbidity_covs_fups_all_symptoms_all_steps"

sim_dir = "outputs/2026_07_20/"
save_dr_list = sim_dir .* sim_no_list .* "/results/estimates"
est_output_list = [load(save_dr * "/est_output.jld2", "estimation_output") for save_dr in save_dr_list]


function get_likelihood(est_output::LTA.EstimationOutput)
    sim_output, sim_params, sim_hyper = LTA.unpack(est_output)
    best_model_params = LTA.get_model_params(est_output, type="best_fitted")

    trans_states = LTA.get_rho_beta_transition_mat(
        best_model_params.beta_transition, best_model_params.rho_trans, sim_params.covariate_mat[1, :])

    lkl, n_zeroed = LTA.lkl_func_transmat(model_params=best_model_params, sim_output=sim_output)

    return lkl, n_zeroed

end

function get_AIC(est_output::LTA.EstimationOutput)

    k = length(est_output.fitted_model_params[1])
    aic = 2 * k + 2 * maximum(est_output.fitted_log_lkl_list)

    return aic
end

function get_AIC_fixed(est_output::LTA.EstimationOutput)
    lkl, zeros = get_likelihood(est_output)
    k = length(est_output.fitted_model_params[1])
    aic = 2 * k + 2 * lkl
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
    labels=string.(1:length(aic_values))
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
    tr_aic = PlotlyJS.bar(x=labels, y=aic_values, name="AIC", width=0.35)
    tr_bic = PlotlyJS.bar(x=labels, y=bic_values, name="BIC", width=0.35)

    plt = PlotlyJS.Plot(
        [tr_aic, tr_bic],
        PlotlyJS.Layout(
            barmode="group",
            title="AIC and BIC Values for Different Models",
            xaxis=PlotlyJS.attr(title="Models"),
            yaxis=PlotlyJS.attr(title="Value", range=[ylims[1], ylims[2]]),
            legend=PlotlyJS.attr(x=0.95, y=0.95),
            width=800,
            height=500,
            bargroupgap=0.12
        )
    )

    return plt
end
lkl_list = get_likelihood.(est_output_list)
aic_val_list = get_AIC.(est_output_list)
bic_val_list = get_BIC.(est_output_list)

aic_val_list_new = get_AIC_fixed.(est_output_list)

# aic_bic_plt = plot_aic_bic(
#     aic_val_list,
#     bic_val_list;
#     labels=["$i States" for i in n_state_list]
# )

# PlotlyJS.savefig(aic_bic_plt, sim_dir * "aic_bic_age_sex_comorbidity_covs_fups.png", width=1200, height=800)