# Import Function 
cd()
include("functions_pwbm_w_spouse.jl")
# Packages Needed 
using DataFrames
using Plots
using XLSX
using CSV;
# Solve steady state
par = create_params()
ss  = solve_steady_state(par);
# Baseline Dependency Path #
dep_path = CSV.read("C:/Users/kchanwong/Documents/PWBM/julia_port/dep_rat.csv", DataFrame) |> DataFrame
proj = project_economy(
    ss,
    n_years         = 75,
    start_year      = 2025,
    g_A             = 0.0113,
    g_pop           = 0.05,
    inflation       = 0.024,
    dep_path        = 0.9 * dep_path.dep_rat,
    ss_cola         = "wage",
    trust_fund_init = 2.76e12,
    trust_fund_rate = 0.047
);
plot(proj.year, 100 * proj.ss_cash_flow_nom./proj.taxable_payroll_nom, label = "Baseline", xlabel = "Year", 
ylabel = "% of Taxable Payroll", title = "Projected Outlays",
ylim = (-7, 0))
proj_ADD0_5 = project_economy(
    ss,
    n_years         = 75,
    start_year      = 2025,
    g_A             = 0.0163,
    g_pop           = 0.05,
    inflation       = 0.024,
    dep_path        = 0.9 * dep_path.dep_rat,
    ss_cola         = "wage",
    trust_fund_init = 2.76e12,
    trust_fund_rate = 0.047
);
plot!(proj.year, 100 * proj_ADD0_5.ss_cash_flow_nom./proj_ADD0_5.taxable_payroll_nom, lwd = 3,
label = "Add 0.5% To Wage Growth", col = "darkgreen")
proj_ADD1 = project_economy(
    ss,
    n_years         = 75,
    start_year      = 2025,
    g_A             = 0.0213,
    g_pop           = 0.05,
    inflation       = 0.024,
    dep_path        = 0.9 * dep_path.dep_rat,
    ss_cola         = "wage",
    trust_fund_init = 2.76e12,
    trust_fund_rate = 0.047
);
plot!(proj.year, 100 * proj_ADD1.ss_cash_flow_nom./proj_ADD1.taxable_payroll_nom, lwd = 3,
label = "Add 1% To Wage Growth", col = "darkgreen");
### Export ###
XLSX.openxlsx("C:/Users/kchanwong/Documents/PWBM/julia_port/projections_gdp_increase.xlsx", mode="w") do xf
    for (name, p) in [("Baseline", proj), ("Add0_5pct", proj_ADD0_5), ("Add1pct", proj_ADD1)]
        sheet = XLSX.addsheet!(xf, name)
        # Header row
        cols = [:year, :ss_cash_flow_nom, :taxable_payroll_nom, :ss_outlays_nom,
                :fica_revenue_nom, :ss_cost_rate, :ss_cash_flow_pct, :ss_balance_pct,
                :trust_fund_nom, :avg_ben_per_retiree_nom, :GDP_nominal]
        for (j, col) in enumerate(cols)
            sheet[1, j] = string(col)
        end
        # Data rows
        vals = [getfield(p, col) for col in cols]
        for i in 1:length(p.year)
            for (j, v) in enumerate(vals)
                sheet[i+1, j] = v[i]
            end
        end
    end
end