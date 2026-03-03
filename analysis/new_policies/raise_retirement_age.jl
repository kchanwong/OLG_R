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
dep_path[!, :dep_rat] = ifelse.(dep_path.year .<= 2040, dep_path.dep_rat, dep_path.dep_rat)# Run projection
# Reform Steady State # 
par_reform = create_params()
par_reform[:J_retire] = 70
ss_reform = solve_steady_state(par_reform);
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
### Level 3: FRA Indexed to Life Expectancy (+1 month per 2 years after 2037) ###
year_cal_full = collect(2025:2099)
fra_path = [yr < 2029 ? 67.0 :
            yr < 2037 ? 67.0 + 3.0 * (yr - 2029) / 8 :
            70.0 + (yr - 2037) / 24
            for yr in year_cal_full]

# Each year, dep ratio scales by (J_max - FRA(t)) / (FRA(t) - J_start) relative to FRA=67 baseline
dep_path_le = [0.9 * dep_path.dep_rat[t] *
               ((100 - fra_path[t]) * (67 - 21)) /
               ((100 - 67)          * (fra_path[t] - 21))
               for t in 1:75]
proj_fra_le = project_economy(ss,
    ss_reform        = ss_reform,
    reform_year      = 2029,
    reform_phase_in  = 8,
    n_years          = 75,
    start_year       = 2025,
    g_A              = 0.0113,
    g_pop            = 0.05,
    inflation        = 0.024,
    dep_path         = dep_path_le,
    ss_cola          = "wage",
    trust_fund_init  = 2.76e12,
    trust_fund_rate  = 0.047);
