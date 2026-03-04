# Import Function 
include("C:/Users/kchanwong/Documents/PWBM/julia_port/functions_pwbm_w_spouse.jl")
# Packages Needed 
using DataFrames
using Plots
using XLSX
using CSV;
# Solve steady state
par = create_params()
ss  = solve_steady_state(par)
dep_path = CSV.read("C:/Users/kchanwong/Documents/PWBM/julia_port/dep_rat.csv", DataFrame) |> DataFrame;
# Increase Payroll Tax Rate #
par_reform = create_params()
par_reform[:ss_cap_frac] = 100000
par_reform[:payroll_tax_rate] = 100000000
par_reform[:ss_cap] = 100000000
ss_reform  = solve_steady_state(par_reform);
proj_no_tax_max = project_economy(ss,
    ss_reform        = ss_reform,
    reform_year      = 2029,
    n_years          = 75,
    start_year       = 2025,
    g_A              = 0.0113,
    g_pop            = 0.05,
    inflation        = 0.024,
    dep_path         = dep_path.dep_rat * 0.9,
    ss_cola          = "wage",
    trust_fund_init  = 2.76e12,
    trust_fund_rate  = 0.047);