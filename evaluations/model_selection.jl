
include("../src/LTA.jl")
include("../digihero_params.jl")
using .LTA
using JLD2
using Statistics
using CSV
using DataFrames
using PlotlyJS

output_dir = "outputs/2026_07_20/"
data_dir = "data/DigiHero/"

combined_all = CSV.read(data_dir * "2026-07-20_data_symptoms_DigiHero_Bonn_preprocessed_ordinal_test.csv", DataFrame, missingstring=["", "NA"])
pattern = Regex(join(vcat(symptoms_list_binary, symptoms_list_ordinal, covariates_list_new_agegroups), "|"))
preprocessed_df = select(combined_all, names(combined_all, pattern))[:, 3:end]
preprocessed_df = filter(row -> !all(ismissing, row[Not(covariates_list_new_agegroups)]), preprocessed_df)

ks = 3:10
nll_test = map(ks) do n_states
    sim_no = string(n_states, "states/age_sex_comorbidity_covs_fups_ordinal_predictioncheck/")
    test_sim_output = LTA.create_sim_mod_data(preprocessed_df; covs=covariates_list_new_agegroups, bins=String[], ordinals=symptoms_list_ordinal, conts=String[], visits=String[], labels=["pcc_" * string(t) * "months_sym_" for t in 6:6:30], n_states=n_states, covariate_tup=(initial=covariates_list_new_agegroups, trans=covariates_list_new_agegroups, em=[]), sim_no=sim_no, comments="")
    estimation_output = load(output_dir * sim_no * "results/estimates/est_output.jld2", "estimation_output")
    LTA.lkl_func(; model_params=LTA.get_model_params(estimation_output, type="best_fitted"), sim_output=test_sim_output)
end

p = PlotlyJS.plot(PlotlyJS.bar(x=collect(ks), y=nll_test), PlotlyJS.Layout(xaxis=PlotlyJS.attr(title="Number of states", tickmode="linear", dtick=1), yaxis=PlotlyJS.attr(title="Test NLL")))
PlotlyJS.savefig(p, "test_nll.png")