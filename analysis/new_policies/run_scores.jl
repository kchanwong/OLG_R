using DataFrames
using XLSX
using OrderedCollections

const DIR = "C:/Users/kchanwong/Documents/PWBM/julia_port/ROMINA_IVANE_PWBM_PAPER/new_policies"

function proj_to_df(p)
    DataFrame(
        year                    = collect(p.year),
        ss_cash_flow_nom        = p.ss_cash_flow_nom,
        taxable_payroll_nom     = p.taxable_payroll_nom,
        ss_outlays_nom          = p.ss_outlays_nom,
        fica_revenue_nom        = p.fica_revenue_nom,
        ss_cost_rate            = p.ss_cost_rate,
        ss_cash_flow_pct        = p.ss_cash_flow_pct,
        ss_balance_pct          = p.ss_balance_pct,
        trust_fund_nom          = p.trust_fund_nom,
        avg_ben_per_retiree_nom = p.avg_ben_per_retiree_nom,
        GDP_nominal             = p.GDP_nominal
    )
end

scores = OrderedDict{String, DataFrame}()

# --- 1. increase_payroll ---
include(joinpath(DIR, "increase_payroll.jl"))
scores["increase_payroll"] = proj_to_df(proj_raise_payroll)

# --- baseline (uses ss + dep_path loaded above) ---
proj_baseline = project_economy(ss,
    ss_reform       = ss,
    reform_year     = 2029,
    n_years         = 75,
    start_year      = 2025,
    g_A             = 0.0113,
    g_pop           = 0.05,
    inflation       = 0.024,
    dep_path        = dep_path.dep_rat * 0.9,
    ss_cola         = "wage",
    trust_fund_init = 2.76e12,
    trust_fund_rate = 0.047)
scores["baseline"] = proj_to_df(proj_baseline)

# --- 2. no_tax_max ---
include(joinpath(DIR, "no_tax_max.jl"))
scores["no_tax_max"] = proj_to_df(proj_no_tax_max)

# --- 3. fringe_tax ---
include(joinpath(DIR, "fringe_tax.jl"))
scores["fringe_tax"] = proj_to_df(proj_raise_fringe)

# --- 4. flat_benefits (reform proj is `proj`) ---
include(joinpath(DIR, "flat_benefits.jl"))
scores["flat_benefits"] = proj_to_df(proj)

# --- 5. pia_indexing (last call is unassigned — re-capture here) ---
include(joinpath(DIR, "pia_indexing.jl"))
proj_pia = project_economy_indexed(ss,
    ss_reform           = ss_reform,
    reform_year         = 2029,
    n_years             = 75,
    start_year          = 2025,
    g_A                 = 0.0113,
    g_pop               = 0.05,
    inflation           = 0.024,
    dep_path            = 0.9 * dep_path.dep_rat,
    ss_cola             = "wage",
    trust_fund_init     = 2.76e12,
    trust_fund_rate     = 0.047,
    pia_factor_indexing = true,
    label               = "PIA Factor Indexing")
scores["pia_indexing"] = proj_to_df(proj_pia)

# --- 6. raise_retirement_age (reform is proj_fra_le; proj is baseline) ---
include(joinpath(DIR, "raise_retirement_age.jl"))
scores["raise_retirement_age"] = proj_to_df(proj_fra_le)

# --- 7. stagflation STAG2 (historical 1973-1982 episode, 2032-2041) ---
const STAG_DIR = "C:/Users/kchanwong/Documents/PWBM/julia_port/ROMINA_IVANE_PWBM_PAPER/counterfactuals"
include(joinpath(STAG_DIR, "stagflation.jl"))
scores["stagflation_STAG2"] = proj_to_df(proj_STAG2)

# --- Write to SCORES.xlsx ---
outpath = joinpath(DIR, "SCORES.xlsx")
XLSX.openxlsx(outpath, mode="w") do xf
    for (i, (name, df)) in enumerate(scores)
        sheet = i == 1 ? xf[1] : XLSX.addsheet!(xf, name)
        i == 1 && XLSX.rename!(sheet, name)
        # Header row
        for (c, col) in enumerate(names(df))
            sheet[1, c] = col
        end
        # Data rows
        for (r, row) in enumerate(eachrow(df))
            for (c, val) in enumerate(row)
                sheet[r + 1, c] = val
            end
        end
    end
end

println("\nSCORES.xlsx written → $outpath")
println("Sheets ($(length(scores))): ", join(keys(scores), ", "))
