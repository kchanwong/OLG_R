# ============================================================
# PWBM Social Security OLG Model — Couples Extension
# Rewritten from functions_pwbm.jl (v9, equil-hours AIME)
#
# Adds: joint household problem with (a, z₁, z₂, marital_status)
#   marital_status ∈ {couple, survivor₁, survivor₂}
#   Gender-specific mortality for within-couple transitions
#   Married-filing-jointly tax brackets
#   Swappable benefit functions for policy scoring:
#     - Current law (spousal + survivor)
#     - Capped spousal benefits
#     - Earnings sharing
#     - Caregiver credits
#
# Dependencies:
#   using Pkg
#   Pkg.add(["Distributions", "Printf", "Statistics", "LinearAlgebra"])
# ============================================================

using Distributions
using Printf
using Statistics
using LinearAlgebra
using Base.Threads

# ============================================================
# UTILITY: number formatting
# ============================================================

function format_comma(x::Real)
    s = string(round(Int, x))
    n = length(s)
    result = ""
    for (i, c) in enumerate(s)
        result *= c
        rem_digits = n - i
        if rem_digits > 0 && rem_digits % 3 == 0
            result *= ","
        end
    end
    result
end

# ============================================================
# HELPERS
# ============================================================

function tauchen(n_z::Int, rho::Float64, sigma_eta::Float64, m::Int=3)
    sz = sigma_eta / sqrt(1 - rho^2)
    zm = m * sz
    zg = collect(range(-zm, zm, length=n_z))
    st = zg[2] - zg[1]
    Pi = zeros(n_z, n_z)
    d  = Normal(0.0, 1.0)
    for i in 1:n_z, j in 1:n_z
        if j == 1
            Pi[i,j] = cdf(d, (zg[1] + st/2 - rho*zg[i]) / sigma_eta)
        elseif j == n_z
            Pi[i,j] = 1.0 - cdf(d, (zg[n_z] - st/2 - rho*zg[i]) / sigma_eta)
        else
            Pi[i,j] = cdf(d, (zg[j] + st/2 - rho*zg[i]) / sigma_eta) -
                      cdf(d, (zg[j] - st/2 - rho*zg[i]) / sigma_eta)
        end
    end
    (z_grid=zg, Pi=Pi)
end

function ergodic_dist(Pi::Matrix{Float64})
    n = size(Pi, 1)
    A = Pi' - I(n)
    A[end, :] .= 1.0
    b = zeros(n); b[end] = 1.0
    A \ b
end

function age_efficiency(age::Int, J_start::Int=21)
    x = (age - J_start + 1) / 40.0
    0.8*x - 0.4*x^2
end

function approx_interp(x::Vector{Float64}, y::Vector{Float64}, xout::Float64)
    n = length(x)
    xout <= x[1]  && return y[1]
    xout >= x[n]  && return y[n]
    i = searchsortedlast(x, xout)
    i = clamp(i, 1, n-1)
    w = (xout - x[i]) / (x[i+1] - x[i])
    y[i]*(1-w) + y[i+1]*w
end

# ============================================================
# SSA PERIOD LIFE TABLE (2022) — gender-specific
# q(a) = probability of dying within one year at exact age a.
# Index: age 0 = index 1, age 120 = index 121.
# ============================================================

const SSA_Q_MALE = Float64[
    0.006064, 0.000491, 0.000309, 0.000248, 0.000199,
    0.000167, 0.000143, 0.000126, 0.000121, 0.000121,
    0.000127, 0.000143, 0.000171, 0.000227, 0.000320,
    0.000451, 0.000622, 0.000826, 0.001026, 0.001182,
    0.001301,
    0.001404, 0.001498, 0.001586, 0.001679, 0.001776,
    0.001881, 0.001985, 0.002095, 0.002219, 0.002332,
    0.002445, 0.002562, 0.002653, 0.002716, 0.002791,
    0.002894, 0.002994, 0.003091, 0.003217, 0.003353,
    0.003499, 0.003642, 0.003811, 0.003996, 0.004175,
    0.004388, 0.004666, 0.004973, 0.005305, 0.005666,
    0.006069, 0.006539, 0.007073, 0.007675, 0.008348,
    0.009051, 0.009822, 0.010669, 0.011548, 0.012458,
    0.013403, 0.014450, 0.015571, 0.016737, 0.017897,
    0.019017, 0.020213, 0.021569, 0.023088, 0.024828,
    0.026705, 0.028761, 0.031116, 0.033861, 0.037088,
    0.041126, 0.045241, 0.049793, 0.054768, 0.060660,
    0.067027, 0.073999, 0.081737, 0.090458, 0.100525,
    0.111793, 0.124494, 0.138398, 0.153207, 0.169704,
    0.187963, 0.208395, 0.230808, 0.253914, 0.277402,
    0.300882, 0.324326, 0.347332, 0.369430, 0.391927,
    0.414726, 0.437722, 0.460800, 0.483840, 0.508032,
    0.533434, 0.560105, 0.588111, 0.617516, 0.648392,
    0.680812, 0.714852, 0.750595, 0.788125, 0.827531,
    0.868907, 0.912353, 0.957970, 1.000000, 1.000000,
]

const SSA_Q_FEMALE = Float64[
    0.005119, 0.000398, 0.000240, 0.000198, 0.000160,
    0.000134, 0.000118, 0.000109, 0.000106, 0.000106,
    0.000111, 0.000121, 0.000140, 0.000162, 0.000188,
    0.000224, 0.000276, 0.000337, 0.000395, 0.000450,
    0.000496,
    0.000532, 0.000567, 0.000610, 0.000650, 0.000699,
    0.000743, 0.000796, 0.000855, 0.000924, 0.000988,
    0.001053, 0.001123, 0.001198, 0.001263, 0.001324,
    0.001403, 0.001493, 0.001596, 0.001700, 0.001803,
    0.001905, 0.002009, 0.002116, 0.002223, 0.002352,
    0.002516, 0.002712, 0.002936, 0.003177, 0.003407,
    0.003642, 0.003917, 0.004238, 0.004619, 0.005040,
    0.005493, 0.005987, 0.006509, 0.007067, 0.007658,
    0.008305, 0.008991, 0.009681, 0.010343, 0.011018,
    0.011743, 0.012532, 0.013512, 0.014684, 0.016025,
    0.017468, 0.019195, 0.021195, 0.023452, 0.025980,
    0.029153, 0.032394, 0.035888, 0.039676, 0.044156,
    0.049087, 0.054635, 0.061066, 0.068431, 0.076841,
    0.086205, 0.096851, 0.109019, 0.121867, 0.135805,
    0.151108, 0.168020, 0.186340, 0.206432, 0.228086,
    0.250406, 0.273699, 0.296984, 0.319502, 0.342716,
    0.366532, 0.390844, 0.415531, 0.440463, 0.466891,
    0.494904, 0.524599, 0.556075, 0.589439, 0.624805,
    0.662294, 0.702031, 0.744153, 0.788125, 0.827531,
    0.868907, 0.912353, 0.957970, 1.000000, 1.000000,
]

const SSA_Q_AVG = (SSA_Q_MALE .+ SSA_Q_FEMALE) ./ 2.0

function survival_probs(par::Dict)
    ages = par[:J_start]:par[:J_max]
    s = Float64[1.0 - SSA_Q_AVG[a + 1] for a in ages]
    s[end] = 0.0
    s
end

function survival_probs_male(par::Dict)
    ages = par[:J_start]:par[:J_max]
    s = Float64[1.0 - SSA_Q_MALE[a + 1] for a in ages]
    s[end] = 0.0
    s
end

function survival_probs_female(par::Dict)
    ages = par[:J_start]:par[:J_max]
    s = Float64[1.0 - SSA_Q_FEMALE[a + 1] for a in ages]
    s[end] = 0.0
    s
end

# ============================================================
# PRODUCTION
# ============================================================

production(K, L, alpha) = K^alpha * L^(1-alpha)
mpk(K, L, alpha)        = alpha * (L/K)^(1-alpha)
mpl(K, L, alpha)        = (1-alpha) * (K/L)^alpha

function calibrate_nu2(par::Dict, rho::Float64)
    tp = par[:tau_statutory_corp] * par[:phi_int_corp]
    par[:leverage_ratio_target] * (tp * rho / (1 - tp))^(-1 / (par[:nu1] - 1))
end

# ============================================================
# TAX SYSTEM
# ============================================================

function apply_tax_schedule(income::Float64,
                            brackets::Vector,
                            rates::Vector)
    income <= 0.0 && return 0.0
    tax = 0.0
    for i in eachindex(rates)
        lo = Float64(brackets[i])
        hi = i < length(rates) ? Float64(brackets[i+1]) : Inf
        tax += rates[i] * max(0.0, min(income, hi) - lo)
    end
    tax
end

function taxable_ss_benefits(ss_benefits::Float64,
                              other_income::Float64,
                              par::Dict)
    ss_benefits <= 0.0 && return (total=0.0, oasi=0.0, hi=0.0)
    prov = other_income + 0.5*ss_benefits
    t1   = par[:ss_tax_thresh1]
    t2   = par[:ss_tax_thresh2]
    prov <= t1 && return (total=0.0, oasi=0.0, hi=0.0)
    tier1 = min(0.50*ss_benefits, 0.50*(prov - t1))
    prov <= t2 && return (total=tier1, oasi=tier1, hi=0.0)
    total = min(tier1 + 0.85*(prov - t2), 0.85*ss_benefits)
    (total=total, oasi=tier1, hi=total-tier1)
end

# --- Individual (single / survivor) tax function ---

function tau_HH(y_lab::Float64, y_corp::Float64, y_pass::Float64,
                y_debt::Float64, ss_ben::Float64, consumption::Float64,
                par::Dict)
    mu   = par[:mu_dollar]
    yl_d = y_lab*mu;  yc_d = y_corp*mu
    yp_d = y_pass*mu; yd_d = y_debt*mu; ss_d = ss_ben*mu
    other = par[:theta_lab_ORD]*yl_d + par[:theta_corp_ORD]*yc_d +
            par[:theta_pass_ORD]*yp_d + yd_d
    ss_tx = par[:ss_tax_use_provisional] ?
            taxable_ss_benefits(ss_d, other, par).total :
            par[:theta_ss_ORD]*ss_d
    ord  = other + ss_tx
    pref = par[:theta_corp_PREF]*yc_d
    (apply_tax_schedule(ord,  par[:ord_brackets],  par[:ord_rates]) +
     apply_tax_schedule(pref, par[:pref_brackets], par[:pref_rates]) +
     par[:payroll_rate] * min(yl_d, par[:payroll_cap]) +
     par[:tau_con]*consumption*mu + par[:tau_lumpsum]) / mu
end

function tau_HH_decomposed(y_lab::Float64, y_corp::Float64, y_pass::Float64,
                           y_debt::Float64, ss_ben::Float64, consumption::Float64,
                           par::Dict)
    mu   = par[:mu_dollar]
    yl_d = y_lab*mu;  yc_d = y_corp*mu
    yp_d = y_pass*mu; yd_d = y_debt*mu; ss_d = ss_ben*mu
    non_ss = par[:theta_lab_ORD]*yl_d + par[:theta_corp_ORD]*yc_d +
             par[:theta_pass_ORD]*yp_d + yd_d
    tiers = par[:ss_tax_use_provisional] ?
            taxable_ss_benefits(ss_d, non_ss, par) :
            (oasi=par[:theta_ss_ORD]*ss_d, total=par[:theta_ss_ORD]*ss_d, hi=0.0)
    pref = par[:theta_corp_PREF]*yc_d
    ot0  = apply_tax_schedule(non_ss,               par[:ord_brackets], par[:ord_rates])
    ot1  = apply_tax_schedule(non_ss + tiers.oasi,  par[:ord_brackets], par[:ord_rates])
    ot2  = apply_tax_schedule(non_ss + tiers.total, par[:ord_brackets], par[:ord_rates])
    pt   = apply_tax_schedule(pref, par[:pref_brackets], par[:pref_rates])
    pay  = par[:payroll_rate] * min(yl_d, par[:payroll_cap])
    ttax = (ot2 + pt + pay + par[:tau_con]*consumption*mu + par[:tau_lumpsum]) / mu
    (total_tax=ttax, ss_tax_rev_oasi=(ot1-ot0)/mu, ss_tax_rev_hi=(ot2-ot1)/mu)
end

function mtr_lab(y_lab_dollars::Float64, ss_ben_dollars::Float64, par::Dict)
    eps = max(1.0, abs(y_lab_dollars)*0.001)
    o1  = par[:theta_lab_ORD]*y_lab_dollars
    o2  = par[:theta_lab_ORD]*(y_lab_dollars + eps)
    tx1 = par[:ss_tax_use_provisional] ?
          taxable_ss_benefits(ss_ben_dollars, o1, par).total :
          par[:theta_ss_ORD]*ss_ben_dollars
    tx2 = par[:ss_tax_use_provisional] ?
          taxable_ss_benefits(ss_ben_dollars, o2, par).total : tx1
    t1  = apply_tax_schedule(o1+tx1, par[:ord_brackets], par[:ord_rates])
    t2  = apply_tax_schedule(o2+tx2, par[:ord_brackets], par[:ord_rates])
    (t2-t1)/eps + (y_lab_dollars < par[:payroll_cap] ? par[:payroll_rate] : 0.0)
end

# --- Couple (MFJ) tax functions ---

function tau_HH_couple(y_lab1::Float64, y_lab2::Float64,
                       y_corp::Float64, y_pass::Float64,
                       y_debt::Float64, ss_ben::Float64,
                       consumption::Float64, par::Dict)
    mu    = par[:mu_dollar]
    yl1_d = y_lab1*mu;  yl2_d = y_lab2*mu
    yc_d  = y_corp*mu;  yp_d  = y_pass*mu
    yd_d  = y_debt*mu;  ss_d  = ss_ben*mu
    other = par[:theta_lab_ORD]*(yl1_d + yl2_d) +
            par[:theta_corp_ORD]*yc_d +
            par[:theta_pass_ORD]*yp_d + yd_d
    ss_tx = par[:ss_tax_use_provisional] ?
            taxable_ss_benefits(ss_d, other, par).total :
            par[:theta_ss_ORD]*ss_d
    ord   = other + ss_tx
    pref  = par[:theta_corp_PREF]*yc_d
    itax  = apply_tax_schedule(ord,  par[:ord_brackets_mfj],  par[:ord_rates_mfj]) +
            apply_tax_schedule(pref, par[:pref_brackets_mfj], par[:pref_rates_mfj])
    pay   = par[:payroll_rate] * min(yl1_d, par[:payroll_cap]) +
            par[:payroll_rate] * min(yl2_d, par[:payroll_cap])
    (itax + pay + par[:tau_con]*consumption*mu + par[:tau_lumpsum]) / mu
end

function tau_HH_couple_decomposed(y_lab1::Float64, y_lab2::Float64,
                                  y_corp::Float64, y_pass::Float64,
                                  y_debt::Float64, ss_ben::Float64,
                                  consumption::Float64, par::Dict)
    mu    = par[:mu_dollar]
    yl1_d = y_lab1*mu;  yl2_d = y_lab2*mu
    yc_d  = y_corp*mu;  yp_d  = y_pass*mu
    yd_d  = y_debt*mu;  ss_d  = ss_ben*mu
    non_ss = par[:theta_lab_ORD]*(yl1_d + yl2_d) +
             par[:theta_corp_ORD]*yc_d +
             par[:theta_pass_ORD]*yp_d + yd_d
    tiers = par[:ss_tax_use_provisional] ?
            taxable_ss_benefits(ss_d, non_ss, par) :
            (oasi=par[:theta_ss_ORD]*ss_d, total=par[:theta_ss_ORD]*ss_d, hi=0.0)
    pref = par[:theta_corp_PREF]*yc_d
    ot0  = apply_tax_schedule(non_ss,               par[:ord_brackets_mfj], par[:ord_rates_mfj])
    ot1  = apply_tax_schedule(non_ss + tiers.oasi,  par[:ord_brackets_mfj], par[:ord_rates_mfj])
    ot2  = apply_tax_schedule(non_ss + tiers.total, par[:ord_brackets_mfj], par[:ord_rates_mfj])
    pt   = apply_tax_schedule(pref, par[:pref_brackets_mfj], par[:pref_rates_mfj])
    pay  = par[:payroll_rate]*min(yl1_d, par[:payroll_cap]) +
           par[:payroll_rate]*min(yl2_d, par[:payroll_cap])
    ttax = (ot2 + pt + pay + par[:tau_con]*consumption*mu + par[:tau_lumpsum]) / mu
    (total_tax=ttax, ss_tax_rev_oasi=(ot1-ot0)/mu, ss_tax_rev_hi=(ot2-ot1)/mu)
end

function mtr_lab_couple(own_earn_d::Float64, spouse_earn_d::Float64,
                        ss_ben_d::Float64, par::Dict)
    eps    = max(1.0, abs(own_earn_d)*0.001)
    total1 = par[:theta_lab_ORD]*(own_earn_d + spouse_earn_d)
    total2 = par[:theta_lab_ORD]*(own_earn_d + eps + spouse_earn_d)
    tx1 = par[:ss_tax_use_provisional] ?
          taxable_ss_benefits(ss_ben_d, total1, par).total :
          par[:theta_ss_ORD]*ss_ben_d
    tx2 = par[:ss_tax_use_provisional] ?
          taxable_ss_benefits(ss_ben_d, total2, par).total :
          par[:theta_ss_ORD]*ss_ben_d
    t1 = apply_tax_schedule(total1+tx1, par[:ord_brackets_mfj], par[:ord_rates_mfj])
    t2 = apply_tax_schedule(total2+tx2, par[:ord_brackets_mfj], par[:ord_rates_mfj])
    (t2-t1)/eps + (own_earn_d < par[:payroll_cap] ? par[:payroll_rate] : 0.0)
end

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

function utility(c::Float64, n::Float64, gamma::Float64=0.30, sigma::Float64=2.0)
    c <= 0.0 && return -1e10
    u_c = abs(sigma - 1.0) < 1e-8 ? log(c) : (c^(1-sigma) - 1) / (1-sigma)
    l   = max(1.0 - n, 1e-10)
    u_l = gamma > 0.0 ? gamma * log(l) : 0.0
    u_c + u_l
end

function utility_couple(c::Float64, n1::Float64, n2::Float64,
                        gamma::Float64=0.30, sigma::Float64=2.0)
    c <= 0.0 && return -1e10
    u_c = abs(sigma - 1.0) < 1e-8 ? log(c) : (c^(1-sigma) - 1) / (1-sigma)
    l1  = max(1.0 - n1, 1e-10)
    l2  = max(1.0 - n2, 1e-10)
    u_l = gamma > 0.0 ? gamma * (log(l1) + log(l2)) : 0.0
    u_c + u_l
end

function mu_c(c::Float64, sigma::Float64=2.0)
    c <= 0.0 && return 1e10
    max(c, 1e-10)^(-sigma)
end

function inv_mu(rhs::Float64, sigma::Float64=2.0)
    rhs <= 0.0 && return 1e10
    max(rhs, 1e-30)^(-1.0 / sigma)
end

function optn(c::Float64, wz::Float64, mtr::Float64,
              gamma::Float64=0.30, sigma::Float64=2.0)
    wa = wz*(1.0 - mtr)
    wa < 1e-10 && return 0.0
    denom = wa * max(c, 1e-10)^(-sigma)
    clamp(1.0 - gamma / max(denom, 1e-10), 0.0, 0.99)
end

# ============================================================
# SOCIAL SECURITY BENEFIT FORMULAS
# ============================================================

function compute_PIA(b::Float64, par::Dict)
    b1 = par[:ss_bend1] / par[:mu_dollar]
    b2 = par[:ss_bend2] / par[:mu_dollar]
    first_rr = par[:first_rr]
    second_rr = par[:second_rr]
    third_rr = par[:third_rr]
    b <= b1 && return first_rr*b
    b <= b2 && return first_rr*b1 + second_rr*(b - b1)
    first_rr*b1 + second_rr*(b2 - b1) + third_rr*(b - b2)
end

# --- Current law: spousal (50% of worker PIA) + survivor (100% of deceased PIA) ---

function benefits_current_law(AIME1::Float64, AIME2::Float64,
                              par::Dict, status::Symbol)
    PIA1 = compute_PIA(AIME1, par)
    PIA2 = compute_PIA(AIME2, par)
    if status == :couple
        b1 = PIA1 + max(0.0, 0.5*PIA2 - PIA1)
        b2 = PIA2 + max(0.0, 0.5*PIA1 - PIA2)
        return (b1, b2)
    elseif status == :survivor1  # spouse 1 alive, spouse 2 dead
        return (max(PIA1, PIA2), 0.0)
    elseif status == :survivor2  # spouse 2 alive, spouse 1 dead
        return (0.0, max(PIA2, PIA1))
    end
    (PIA1, PIA2)
end

# --- Capped spousal: spousal benefit cannot exceed PIA at minimum wage ---

function benefits_capped_spousal(AIME1::Float64, AIME2::Float64,
                                 par::Dict, status::Symbol)
    PIA1 = compute_PIA(AIME1, par)
    PIA2 = compute_PIA(AIME2, par)
    cap  = par[:spousal_cap]
    if status == :couple
        sp1 = min(cap, max(0.0, 0.5*PIA2 - PIA1))
        sp2 = min(cap, max(0.0, 0.5*PIA1 - PIA2))
        return (PIA1 + sp1, PIA2 + sp2)
    elseif status == :survivor1
        return (max(PIA1, PIA2), 0.0)
    elseif status == :survivor2
        return (0.0, max(PIA2, PIA1))
    end
    (PIA1, PIA2)
end

# --- Earnings sharing: combine and average earnings, no spousal/survivor top-up ---

function benefits_earnings_sharing(AIME1::Float64, AIME2::Float64,
                                   par::Dict, status::Symbol)
    shared_AIME = (AIME1 + AIME2) / 2.0
    PIA_shared  = compute_PIA(shared_AIME, par)
    if status == :couple
        return (PIA_shared, PIA_shared)
    elseif status == :survivor1
        return (PIA_shared, 0.0)
    elseif status == :survivor2
        return (0.0, PIA_shared)
    end
    (PIA_shared, PIA_shared)
end

# --- Caregiver credits: no spousal benefit, AIME already modified upstream ---

function benefits_caregiver_credits(AIME1::Float64, AIME2::Float64,
                                    par::Dict, status::Symbol)
    PIA1 = compute_PIA(AIME1, par)
    PIA2 = compute_PIA(AIME2, par)
    if status == :couple
        return (PIA1, PIA2)   # no spousal top-up
    elseif status == :survivor1
        return (max(PIA1, PIA2), 0.0)  # keep survivor
    elseif status == :survivor2
        return (0.0, max(PIA2, PIA1))
    end
    (PIA1, PIA2)
end

# ============================================================
# AIME COMPUTATION FOR COUPLES
# ============================================================

function compute_aime_couples(par::Dict, grids::NamedTuple, prices::Dict,
                              zp1::Float64, zp2::Float64,
                              pol_n1_c::Array{Float64,4},
                              pol_n2_c::Array{Float64,4};
                              aime_modifier=nothing)
    nz    = par[:n_z]
    Jr    = par[:J_retire] - par[:J_start] + 1
    zg    = grids.z_grid
    cap_b = par[:ss_cap_frac] * par[:avg_wage_mu]
    g_idx = par[:g_A_index]
    age_60_j = 67 - par[:J_start]
    AIME1_by_z = zeros(nz, nz)
    AIME2_by_z = zeros(nz, nz)

    for iz1 in 1:nz, iz2 in 1:nz
        e1 = zeros(Jr-1)
        e2 = zeros(Jr-1)
        n1_path = zeros(Jr-1)
        n2_path = zeros(Jr-1)
        for jw in 1:(Jr-1)
            zv1 = exp(zg[iz1] + zp1 + grids.age_eff[jw])
            zv2 = exp(zg[iz2] + zp2 + grids.age_eff[jw])
            n1_avg = mean(pol_n1_c[:, iz1, iz2, jw])
            n2_avg = mean(pol_n2_c[:, iz1, iz2, jw])
            n1_path[jw] = n1_avg
            n2_path[jw] = n2_avg
            raw1 = prices[:w] * zv1 * n1_avg * (1.0 + get(par, :aime_nwc_frac, 0.0))
            raw2 = prices[:w] * zv2 * n2_avg  * (1.0 + get(par, :aime_nwc_frac, 0.0))
            if jw <= age_60_j
                yrs = age_60_j - jw
                idx_f = (1.0 + g_idx)^yrs
                e1[jw] = min(raw1 * idx_f, cap_b)
                e2[jw] = min(raw2 * idx_f, cap_b)
            else
                e1[jw] = min(raw1, cap_b)
                e2[jw] = min(raw2, cap_b)
            end
        end

        # Apply AIME modifier (e.g., caregiver credits)
        if !isnothing(aime_modifier)
            e1, e2 = aime_modifier(e1, e2, n1_path, n2_path, par)
        end

        top35 = min(35, Jr-1)
        AIME1_by_z[iz1,iz2] = mean(sort(e1, rev=true)[1:top35])
        AIME2_by_z[iz2,iz1] = mean(sort(e2, rev=true)[1:top35])
    end
    (AIME1=AIME1_by_z, AIME2=AIME2_by_z)
end

# Default caregiver credits modifier
function caregiver_modifier(e1, e2, n1_path, n2_path, par)
    cg_start = get(par, :cg_start_age, 25) - par[:J_start] + 1
    cg_end   = get(par, :cg_end_age, 38)   - par[:J_start] + 1
    cg_level = get(par, :cg_credit_level, 0.5) * par[:avg_wage_mu]
    cg_thresh = get(par, :cg_n_threshold, 0.10)
    cg_max_years = get(par, :cg_max_years, 10)
    years_credited = 0
    for jw in eachindex(e2)
        if jw >= cg_start && jw <= cg_end && years_credited < cg_max_years
            if n2_path[jw] < cg_thresh
                e2[jw] = max(e2[jw], cg_level)
                years_credited += 1
            end
        end
    end
    (e1, e2)
end

# ============================================================
# EGM CORE — SURVIVOR (single-agent, adapted from original)
# ============================================================

function run_egm_survivor(par::Dict, grids::NamedTuple, prices::Dict,
                          zp_own::Float64,
                          AIME_own_by_z::Vector{Float64},
                          AIME_deceased_avg::Float64,
                          surv_gender::Vector{Float64},
                          survivor_id::Symbol,
                          benefit_fn::Function)
    na = par[:n_a]; nz = par[:n_z]; nj = par[:n_ages]
    Jr = par[:J_retire] - par[:J_start] + 1
    ag = grids.a_grid; zg = grids.z_grid
    Pz = grids.Pi_z
    bt = par[:beta]; gm = par[:gamma]; sg = par[:sigma]
    w  = prices[:w]; rp = prices[:pi_corp]
    beq = prices[:beq]

    V   = zeros(na, nz, nj)
    pa  = zeros(na, nz, nj)
    pn  = zeros(na, nz, nj)
    pcc = zeros(na, nz, nj)

    # Terminal condition
    for iz in 1:nz, ia in 1:na
        aime_own = AIME_own_by_z[iz]
        if survivor_id == :survivor1
            b1, _ = benefit_fn(aime_own, AIME_deceased_avg, par, :survivor1)
            ss = b1
        else
            _, b2 = benefit_fn(AIME_deceased_avg, aime_own, par, :survivor2)
            ss = b2
        end
        cash = ag[ia]*(1+rp) + ss + beq
        pcc[ia,iz,nj] = max(1e-10, cash*0.95)
        V[ia,iz,nj]   = utility(pcc[ia,iz,nj], 0.0, gm, sg)
    end

    # Backward induction
    for j in (nj-1):-1:1
        ret = j >= Jr
        sn  = surv_gender[j+1]

        for iz in 1:nz
            zv  = exp(zg[iz] + zp_own + grids.age_eff[j])
            wz  = w * zv
            aime_own = AIME_own_by_z[iz]
            if survivor_id == :survivor1
                b1, _ = benefit_fn(aime_own, AIME_deceased_avg, par, :survivor1)
                ssb = ret ? b1 : 0.0
            else
                _, b2 = benefit_fn(AIME_deceased_avg, aime_own, par, :survivor2)
                ssb = ret ? b2 : 0.0
            end

            # Expected marginal utility tomorrow
            Emu = zeros(na)
            for ia2 in 1:na, iz2 in 1:nz
                Emu[ia2] += Pz[iz,iz2] * mu_c(pcc[ia2,iz2,j+1], sg)
            end

            ce = zeros(na); ne = zeros(na); ae = zeros(na)
            for ia2 in 1:na
                rhs = bt*sn*(1+rp)*Emu[ia2]
                if ret
                    ce[ia2] = inv_mu(rhs, sg)
                    ne[ia2] = 0.0
                else
                    cg = inv_mu(rhs, sg); ng = 0.3
                    for k in 1:6
                        mk = mtr_lab(wz*ng*par[:mu_dollar],
                                     ssb*par[:mu_dollar], par)
                        ng = optn(cg, wz, mk, gm, sg)
                        cg = inv_mu(rhs, sg)
                    end
                    ce[ia2] = cg; ne[ia2] = ng
                end
                tx      = tau_HH(wz*ne[ia2], 0.0, 0.0, 0.0, ssb, ce[ia2], par)
                ae[ia2] = (ag[ia2] + ce[ia2] + tx - wz*ne[ia2] - ssb - beq) / (1+rp)
            end

            vld = [isfinite(ae[i]) && isfinite(ce[i]) &&
                   !isnan(ae[i]) && !isnan(ce[i]) for i in 1:na]

            if sum(vld) < 2
                for ia in 1:na
                    yfb  = ret ? 0.0 : wz*0.3
                    cash = ag[ia]*(1+rp) + yfb + ssb + beq
                    tfb  = tau_HH(yfb, 0.0, 0.0, 0.0, ssb, max(1e-10, cash), par)
                    pcc[ia,iz,j] = max(1e-10, (cash-tfb)*0.5)
                    pa[ia,iz,j]  = max(0.0, cash-tfb-pcc[ia,iz,j])
                    pn[ia,iz,j]  = ret ? 0.0 : 0.3
                    V[ia,iz,j]   = utility(pcc[ia,iz,j], pn[ia,iz,j], gm, sg)
                end
                continue
            end

            av = ae[vld]; cv = ce[vld]; nv = ne[vld]
            ord = sortperm(av)
            av = av[ord]; cv = cv[ord]; nv = nv[ord]
            uniq = vcat(true, [av[i] != av[i-1] for i in 2:length(av)])
            av = av[uniq]; cv = cv[uniq]; nv = nv[uniq]
            amn = av[1]

            for ia in 1:na
                aval = ag[ia]
                if aval <= amn
                    pa[ia,iz,j] = 0.0
                    nc = ret ? 0.0 :
                         optn(0.1, wz,
                              mtr_lab(wz*0.3*par[:mu_dollar],
                                      ssb*par[:mu_dollar], par), gm, sg)
                    yc   = wz*nc
                    cash = aval*(1+rp) + yc + ssb + beq
                    tc   = tau_HH(yc, 0.0, 0.0, 0.0, ssb, max(1e-10, cash), par)
                    cc   = max(1e-10, cash-tc)
                    if !ret
                        for k in 1:5
                            mc = mtr_lab(wz*nc*par[:mu_dollar],
                                         ssb*par[:mu_dollar], par)
                            nc   = optn(cc, wz, mc, gm, sg)
                            yc   = wz*nc
                            cash = aval*(1+rp) + yc + ssb + beq
                            tc   = tau_HH(yc, 0.0, 0.0, 0.0, ssb, max(1e-10, cash), par)
                            cc   = max(1e-10, cash-tc)
                        end
                    end
                    pcc[ia,iz,j] = cc; pn[ia,iz,j] = nc
                else
                    pcc[ia,iz,j] = max(1e-10, approx_interp(av, cv, aval))
                    pn[ia,iz,j]  = ret ? 0.0 :
                                   max(0.0, min(0.99, approx_interp(av, nv, aval)))
                    yi = wz*pn[ia,iz,j]
                    ci = aval*(1+rp) + yi + ssb + beq
                    ti = tau_HH(yi, 0.0, 0.0, 0.0, ssb, pcc[ia,iz,j], par)
                    pa[ia,iz,j] = max(0.0, ci - pcc[ia,iz,j] - ti)
                end
                uv  = utility(pcc[ia,iz,j], pn[ia,iz,j], gm, sg)
                inx = argmin(abs.(ag .- pa[ia,iz,j]))
                ct  = sum(Pz[iz,iz2] * V[inx,iz2,j+1] for iz2 in 1:nz)
                V[ia,iz,j] = uv + sn*bt*ct
            end
        end
    end

    (V=V, pol_a=pa, pol_n=pn, pol_c=pcc)
end

# ============================================================
# EGM CORE — COUPLES
# ============================================================

function run_egm_couple(par::Dict, grids::NamedTuple, prices::Dict,
                        zp1::Float64, zp2::Float64,
                        AIME1_by_z::Matrix{Float64},
                        AIME2_by_z::Matrix{Float64},
                        V_s1::Array{Float64,3},
                        pol_c_s1::Array{Float64,3},
                        V_s2::Array{Float64,3},
                        pol_c_s2::Array{Float64,3},
                        benefit_fn::Function)
    na = par[:n_a]; nz = par[:n_z]; nj = par[:n_ages]
    Jr = par[:J_retire] - par[:J_start] + 1
    ag = grids.a_grid; zg = grids.z_grid
    Pz = grids.Pi_z
    bt = par[:beta]; gm = par[:gamma]; sg = par[:sigma]
    w  = prices[:w]; rp = prices[:pi_corp]
    beq = prices[:beq]
    sm_vec = grids.surv_m
    sf_vec = grids.surv_f

    # 4D arrays: (a, z1, z2, age)
    V_c   = zeros(na, nz, nz, nj)
    pa_c  = zeros(na, nz, nz, nj)
    pn1_c = zeros(na, nz, nz, nj)
    pn2_c = zeros(na, nz, nz, nj)
    pc_c  = zeros(na, nz, nz, nj)

    # Terminal condition
    for iz1 in 1:nz, iz2 in 1:nz, ia in 1:na
        b1, b2 = benefit_fn(AIME1_by_z[iz1,iz2], AIME2_by_z[iz2,iz1], par, :couple)
        cash = ag[ia]*(1+rp) + b1 + b2 + beq
        pc_c[ia,iz1,iz2,nj]  = max(1e-10, cash*0.95)
        V_c[ia,iz1,iz2,nj]   = utility_couple(pc_c[ia,iz1,iz2,nj], 0.0, 0.0, gm, sg)
    end

    # Backward induction
    for j in (nj-1):-1:1
        ret = j >= Jr
        sm  = sm_vec[j+1]
        sf  = sf_vec[j+1]

        # ---- Vectorized Emu: precompute marginal utility matrices once per age ----
        # Couple: MU_cc[ia, iz1p, iz2p] = u'(c) at (ia, iz1p, iz2p, j+1)
        MU_cc = zeros(na, nz, nz)
        for iz2p in 1:nz, iz1p in 1:nz, ia2 in 1:na
            MU_cc[ia2, iz1p, iz2p] = mu_c(pc_c[ia2, iz1p, iz2p, j+1], sg)
        end
        # Survivor marginal utilities: MU_s1[ia, iz1p], MU_s2[ia, iz2p]
        MU_s1 = zeros(na, nz)
        MU_s2 = zeros(na, nz)
        for izp in 1:nz, ia2 in 1:na
            MU_s1[ia2, izp] = mu_c(pol_c_s1[ia2, izp, j+1], sg)
            MU_s2[ia2, izp] = mu_c(pol_c_s2[ia2, izp, j+1], sg)
        end
        # Reshape couple MU for BLAS: MU_cc_flat[ia, iz1p*nz+iz2p] → (na × nz²)
        MU_cc_flat = reshape(MU_cc, na, nz*nz)

        # Precompute continuation value matrices for V
        V_cc_next = zeros(na, nz, nz)
        for iz2p in 1:nz, iz1p in 1:nz, ia2 in 1:na
            V_cc_next[ia2, iz1p, iz2p] = V_c[ia2, iz1p, iz2p, j+1]
        end
        V_cc_flat = reshape(V_cc_next, na, nz*nz)
        V_s1_next = zeros(na, nz)
        V_s2_next = zeros(na, nz)
        for izp in 1:nz, ia2 in 1:na
            V_s1_next[ia2, izp] = V_s1[ia2, izp, j+1]
            V_s2_next[ia2, izp] = V_s2[ia2, izp, j+1]
        end

        # ---- Thread over (iz1, iz2) pairs — each writes to non-overlapping slices ----
        nzz = nz * nz
        @threads for idx in 1:nzz
            iz1 = div(idx - 1, nz) + 1
            iz2 = mod(idx - 1, nz) + 1

            zv1 = exp(zg[iz1] + zp1 + grids.age_eff[j])
            zv2 = exp(zg[iz2] + zp2 + grids.age_eff[j])
            wz1 = w * zv1
            wz2 = w * zv2

            aime1 = AIME1_by_z[iz1,iz2]
            aime2 = AIME2_by_z[iz2,iz1]
            if ret
                b1, b2 = benefit_fn(aime1, aime2, par, :couple)
                ssb_tot = b1 + b2
            else
                ssb_tot = 0.0
            end

            # ---- Vectorized Emu via precomputed matrices ----
            # Joint transition weights: Pz[iz1,:] ⊗ Pz[iz2,:] → length nz²
            joint_prob = vec(Pz[iz1,:] * Pz[iz2,:]')   # (nz²,)
            # Couple Emu: MU_cc_flat (na × nz²) * joint_prob (nz²) → (na,)
            Emu_cc = MU_cc_flat * joint_prob
            # Survivor Emu: matrix-vector products
            Emu_s1 = MU_s1 * Pz[iz1,:]   # (na,) — female dies, male survives
            Emu_s2 = MU_s2 * Pz[iz2,:]   # (na,) — male dies, female survives
            # Mortality-weighted Emu
            Emu = sm*sf .* Emu_cc .+ (1-sm)*sf .* Emu_s2 .+ sm*(1-sf) .* Emu_s1

            # EGM: for each a' candidate, find (c, n1, n2, a_today)
            ce = zeros(na); n1e = zeros(na); n2e = zeros(na); ae = zeros(na)
            for ia2 in 1:na
                rhs = bt*(1+rp)*Emu[ia2]
                if ret
                    ce[ia2]  = inv_mu(rhs, sg)
                    n1e[ia2] = 0.0; n2e[ia2] = 0.0
                else
                    cg = inv_mu(rhs, sg)
                    ng1 = 0.3; ng2 = 0.3
                    for k in 1:8
                        mk1 = mtr_lab_couple(wz1*ng1*par[:mu_dollar],
                                             wz2*ng2*par[:mu_dollar],
                                             ssb_tot*par[:mu_dollar], par)
                        mk2 = mtr_lab_couple(wz2*ng2*par[:mu_dollar],
                                             wz1*ng1*par[:mu_dollar],
                                             ssb_tot*par[:mu_dollar], par)
                        ng1 = optn(cg, wz1, mk1, gm, sg)
                        ng2 = optn(cg, wz2, mk2, gm, sg)
                        cg  = inv_mu(rhs, sg)
                    end
                    ce[ia2] = cg; n1e[ia2] = ng1; n2e[ia2] = ng2
                end
                tx = tau_HH_couple(wz1*n1e[ia2], wz2*n2e[ia2],
                                   0.0, 0.0, 0.0, ssb_tot, ce[ia2], par)
                ae[ia2] = (ag[ia2] + ce[ia2] + tx -
                           wz1*n1e[ia2] - wz2*n2e[ia2] - ssb_tot - beq) / (1+rp)
            end

            vld = [isfinite(ae[i]) && isfinite(ce[i]) &&
                   !isnan(ae[i]) && !isnan(ce[i]) for i in 1:na]

            if sum(vld) < 2
                for ia in 1:na
                    yf1  = ret ? 0.0 : wz1*0.3
                    yf2  = ret ? 0.0 : wz2*0.3
                    cash = ag[ia]*(1+rp) + yf1 + yf2 + ssb_tot + beq
                    tfb  = tau_HH_couple(yf1, yf2, 0.0, 0.0, 0.0,
                                         ssb_tot, max(1e-10, cash), par)
                    pc_c[ia,iz1,iz2,j]  = max(1e-10, (cash-tfb)*0.5)
                    pa_c[ia,iz1,iz2,j]  = max(0.0, cash-tfb-pc_c[ia,iz1,iz2,j])
                    pn1_c[ia,iz1,iz2,j] = ret ? 0.0 : 0.3
                    pn2_c[ia,iz1,iz2,j] = ret ? 0.0 : 0.3
                    V_c[ia,iz1,iz2,j]   = utility_couple(
                        pc_c[ia,iz1,iz2,j], pn1_c[ia,iz1,iz2,j],
                        pn2_c[ia,iz1,iz2,j], gm, sg)
                end
                continue
            end

            # Sort by endogenous asset grid
            av = ae[vld]; cv = ce[vld]; n1v = n1e[vld]; n2v = n2e[vld]
            ord = sortperm(av)
            av = av[ord]; cv = cv[ord]; n1v = n1v[ord]; n2v = n2v[ord]
            uniq = vcat(true, [av[i] != av[i-1] for i in 2:length(av)])
            av = av[uniq]; cv = cv[uniq]; n1v = n1v[uniq]; n2v = n2v[uniq]
            amn = av[1]

            # Interpolate onto exogenous grid
            for ia in 1:na
                aval = ag[ia]
                if aval <= amn
                    pa_c[ia,iz1,iz2,j] = 0.0
                    nc1 = ret ? 0.0 : 0.3; nc2 = ret ? 0.0 : 0.3
                    yc1 = wz1*nc1; yc2 = wz2*nc2
                    cash = aval*(1+rp) + yc1 + yc2 + ssb_tot + beq
                    tc = tau_HH_couple(yc1, yc2, 0.0, 0.0, 0.0,
                                       ssb_tot, max(1e-10, cash), par)
                    cc = max(1e-10, cash - tc)
                    if !ret
                        for k in 1:5
                            mc1 = mtr_lab_couple(wz1*nc1*par[:mu_dollar],
                                                 wz2*nc2*par[:mu_dollar],
                                                 ssb_tot*par[:mu_dollar], par)
                            mc2 = mtr_lab_couple(wz2*nc2*par[:mu_dollar],
                                                 wz1*nc1*par[:mu_dollar],
                                                 ssb_tot*par[:mu_dollar], par)
                            nc1 = optn(cc, wz1, mc1, gm, sg)
                            nc2 = optn(cc, wz2, mc2, gm, sg)
                            yc1 = wz1*nc1; yc2 = wz2*nc2
                            cash = aval*(1+rp) + yc1 + yc2 + ssb_tot + beq
                            tc = tau_HH_couple(yc1, yc2, 0.0, 0.0, 0.0,
                                               ssb_tot, max(1e-10, cash), par)
                            cc = max(1e-10, cash - tc)
                        end
                    end
                    pc_c[ia,iz1,iz2,j]  = cc
                    pn1_c[ia,iz1,iz2,j] = nc1
                    pn2_c[ia,iz1,iz2,j] = nc2
                else
                    pc_c[ia,iz1,iz2,j]  = max(1e-10, approx_interp(av, cv, aval))
                    pn1_c[ia,iz1,iz2,j] = ret ? 0.0 :
                        max(0.0, min(0.99, approx_interp(av, n1v, aval)))
                    pn2_c[ia,iz1,iz2,j] = ret ? 0.0 :
                        max(0.0, min(0.99, approx_interp(av, n2v, aval)))
                    yi = wz1*pn1_c[ia,iz1,iz2,j] + wz2*pn2_c[ia,iz1,iz2,j]
                    ci = aval*(1+rp) + yi + ssb_tot + beq
                    ti = tau_HH_couple(wz1*pn1_c[ia,iz1,iz2,j],
                                       wz2*pn2_c[ia,iz1,iz2,j],
                                       0.0, 0.0, 0.0, ssb_tot,
                                       pc_c[ia,iz1,iz2,j], par)
                    pa_c[ia,iz1,iz2,j] = max(0.0, ci - pc_c[ia,iz1,iz2,j] - ti)
                end

                # Value function — vectorized continuation
                uv = utility_couple(pc_c[ia,iz1,iz2,j],
                                    pn1_c[ia,iz1,iz2,j],
                                    pn2_c[ia,iz1,iz2,j], gm, sg)
                inx = argmin(abs.(ag .- pa_c[ia,iz1,iz2,j]))
                ct_cc = dot(joint_prob, V_cc_flat[inx, :])
                ct_s2 = dot(Pz[iz2,:], V_s2_next[inx, :])
                ct_s1 = dot(Pz[iz1,:], V_s1_next[inx, :])
                ctn = sm*sf*ct_cc + (1-sm)*sf*ct_s2 + sm*(1-sf)*ct_s1
                V_c[ia,iz1,iz2,j] = uv + bt*ctn
            end
        end  # @threads
    end

    (V=V_c, pol_a=pa_c, pol_n1=pn1_c, pol_n2=pn2_c, pol_c=pc_c)
end

# ============================================================
# HOUSEHOLD SOLVER — two-pass, couples + survivors
# ============================================================

function solve_household_couples(par::Dict, grids::NamedTuple, prices::Dict,
                                 zp1::Float64, zp2::Float64;
                                 benefit_fn::Function=benefits_current_law,
                                 aime_modifier=nothing)
    nz  = par[:n_z]
    Jr  = par[:J_retire] - par[:J_start] + 1
    zg  = grids.z_grid
    cap_b = par[:ss_cap_frac] * par[:avg_wage_mu]
    erg   = ergodic_dist(grids.Pi_z)

    # ---- Pass 1: fixed hours n1=n2=0.35, compute initial AIME ----
    AIME1_p1 = zeros(nz, nz)
    AIME2_p1 = zeros(nz, nz)
    for iz1 in 1:nz, iz2 in 1:nz
        e1 = zeros(Jr-1); e2 = zeros(Jr-1)
        for jw in 1:(Jr-1)
            zv1 = exp(zg[iz1] + zp1 + grids.age_eff[jw])
            zv2 = exp(zg[iz2] + zp2 + grids.age_eff[jw])
            e1[jw] = min(prices[:w]*zv1*0.35, cap_b)
            e2[jw] = min(prices[:w]*zv2*0.35, cap_b)
        end
        top35 = min(35, Jr-1)
        AIME1_p1[iz1,iz2] = mean(sort(e1, rev=true)[1:top35])
        AIME2_p1[iz2,iz1] = mean(sort(e2, rev=true)[1:top35])
    end

    # Average deceased AIME for survivor problems (weighted by ergodic dist)
    AIME1_deceased_avg = zeros(nz)
    AIME2_deceased_avg = zeros(nz)
    for iz in 1:nz
        AIME1_deceased_avg[iz] = sum(erg[iz2] * AIME1_p1[iz,iz2] for iz2 in 1:nz)
        AIME2_deceased_avg[iz] = sum(erg[iz1] * AIME2_p1[iz,iz1] for iz1 in 1:nz)
    end

    # Pass 1 logged from caller; suppress inside threaded region

    # Solve survivor problems first (needed for couple continuation)
    res_s1 = run_egm_survivor(par, grids, prices, zp1,
                              [mean(AIME1_p1[iz,:]) for iz in 1:nz],
                              mean(AIME2_p1),
                              grids.surv_m, :survivor1, benefit_fn)
    res_s2 = run_egm_survivor(par, grids, prices, zp2,
                              [mean(AIME2_p1[:,iz]) for iz in 1:nz],
                              mean(AIME1_p1),
                              grids.surv_f, :survivor2, benefit_fn)

    # Solve couple problem
    res_c1 = run_egm_couple(par, grids, prices, zp1, zp2,
                            AIME1_p1, AIME2_p1,
                            res_s1.V, res_s1.pol_c,
                            res_s2.V, res_s2.pol_c,
                            benefit_fn)

    # ---- Pass 2: equilibrium hours → recompute AIME → re-solve ----
    aimes_p2 = compute_aime_couples(par, grids, prices, zp1, zp2,
                                    res_c1.pol_n1, res_c1.pol_n2,
                                    aime_modifier=aime_modifier)
    AIME1_p2 = aimes_p2.AIME1
    AIME2_p2 = aimes_p2.AIME2

    aime_chg = (mean(abs.(AIME1_p2 .- AIME1_p1)) + mean(abs.(AIME2_p2 .- AIME2_p1))) /
               max(mean(AIME1_p1) + mean(AIME2_p1), 1e-10) * 100

    # Pass 2 logged from caller

    # Re-average for survivor problems
    AIME1_own_s1 = Float64[mean(AIME1_p2[iz,:]) for iz in 1:nz]
    AIME2_own_s2 = Float64[mean(AIME2_p2[:,iz]) for iz in 1:nz]
    AIME1_dec_avg = mean(AIME1_p2)
    AIME2_dec_avg = mean(AIME2_p2)

    res_s1 = run_egm_survivor(par, grids, prices, zp1,
                              AIME1_own_s1, AIME2_dec_avg,
                              grids.surv_m, :survivor1, benefit_fn)
    res_s2 = run_egm_survivor(par, grids, prices, zp2,
                              AIME2_own_s2, AIME1_dec_avg,
                              grids.surv_f, :survivor2, benefit_fn)

    res_c2 = run_egm_couple(par, grids, prices, zp1, zp2,
                            AIME1_p2, AIME2_p2,
                            res_s1.V, res_s1.pol_c,
                            res_s2.V, res_s2.pol_c,
                            benefit_fn)

    (couple   = res_c2,
     surv1    = res_s1,
     surv2    = res_s2,
     AIME1    = AIME1_p2,
     AIME2    = AIME2_p2,
     AIME1_deceased_avg = AIME1_dec_avg,
     AIME2_deceased_avg = AIME2_dec_avg,
     aime_pct_change = aime_chg)
end

# ============================================================
# DISTRIBUTION — with mortality transitions
# ============================================================

function compute_distribution_couples(par::Dict, grids::NamedTuple,
                                      pol_a_c::Array{Float64,4},
                                      pol_a_s1::Array{Float64,3},
                                      pol_a_s2::Array{Float64,3})
    na = par[:n_a]; nz = par[:n_z]; nj = par[:n_ages]
    ag = grids.a_grid; Pz = grids.Pi_z

    x_c  = zeros(na, nz, nz, nj)   # couple
    x_s1 = zeros(na, nz, nj)       # survivor1 (male alive)
    x_s2 = zeros(na, nz, nj)       # survivor2 (female alive)

    # Initialize: all enter as couples at j=1, uniform over z1×z2
    for iz1 in 1:nz, iz2 in 1:nz
        x_c[1, iz1, iz2, 1] = (1.0 / (nz*nz)) / (1.0 + par[:g_pop])^(nj-1)
    end

    for j in 1:(nj-1)
        sm = grids.surv_m[j+1]
        sf = grids.surv_f[j+1]
        gpf = 1.0 / (1.0 + par[:g_pop])

        # --- Couple transitions ---
        for iz1 in 1:nz, iz2 in 1:nz, ia in 1:na
            m = x_c[ia, iz1, iz2, j]
            m < 1e-15 && continue
            an = max(ag[1], min(ag[na], pol_a_c[ia, iz1, iz2, j]))
            lo = max(1, min(na-1, searchsortedlast(ag, an)))
            hi = lo + 1
            wh = (ag[hi] - ag[lo]) > 1e-12 ? (an - ag[lo])/(ag[hi]-ag[lo]) : 0.0

            # Both survive → couple
            mass_cc = m * sm * sf * gpf
            for iz1p in 1:nz, iz2p in 1:nz
                pr = Pz[iz1,iz1p] * Pz[iz2,iz2p]
                pr < 1e-15 && continue
                cb = mass_cc * pr
                x_c[lo, iz1p, iz2p, j+1] += cb*(1-wh)
                x_c[hi, iz1p, iz2p, j+1] += cb*wh
            end
            # Male dies → survivor2
            mass_s2 = m * (1-sm) * sf * gpf
            for iz2p in 1:nz
                pr = Pz[iz2,iz2p]
                pr < 1e-15 && continue
                cb = mass_s2 * pr
                x_s2[lo, iz2p, j+1] += cb*(1-wh)
                x_s2[hi, iz2p, j+1] += cb*wh
            end
            # Female dies → survivor1
            mass_s1 = m * sm * (1-sf) * gpf
            for iz1p in 1:nz
                pr = Pz[iz1,iz1p]
                pr < 1e-15 && continue
                cb = mass_s1 * pr
                x_s1[lo, iz1p, j+1] += cb*(1-wh)
                x_s1[hi, iz1p, j+1] += cb*wh
            end
        end

        # --- Survivor1 transitions ---
        for iz1 in 1:nz, ia in 1:na
            m = x_s1[ia, iz1, j]
            m < 1e-15 && continue
            an = max(ag[1], min(ag[na], pol_a_s1[ia, iz1, j]))
            lo = max(1, min(na-1, searchsortedlast(ag, an)))
            hi = lo + 1
            wh = (ag[hi] - ag[lo]) > 1e-12 ? (an - ag[lo])/(ag[hi]-ag[lo]) : 0.0
            mass_s = m * sm * gpf  # male survival
            for iz1p in 1:nz
                pr = Pz[iz1,iz1p]
                pr < 1e-15 && continue
                cb = mass_s * pr
                x_s1[lo, iz1p, j+1] += cb*(1-wh)
                x_s1[hi, iz1p, j+1] += cb*wh
            end
        end

        # --- Survivor2 transitions ---
        for iz2 in 1:nz, ia in 1:na
            m = x_s2[ia, iz2, j]
            m < 1e-15 && continue
            an = max(ag[1], min(ag[na], pol_a_s2[ia, iz2, j]))
            lo = max(1, min(na-1, searchsortedlast(ag, an)))
            hi = lo + 1
            wh = (ag[hi] - ag[lo]) > 1e-12 ? (an - ag[lo])/(ag[hi]-ag[lo]) : 0.0
            mass_s = m * sf * gpf  # female survival
            for iz2p in 1:nz
                pr = Pz[iz2,iz2p]
                pr < 1e-15 && continue
                cb = mass_s * pr
                x_s2[lo, iz2p, j+1] += cb*(1-wh)
                x_s2[hi, iz2p, j+1] += cb*wh
            end
        end
    end

    tot = sum(x_c) + sum(x_s1) + sum(x_s2)
    if tot > 0
        x_c  ./= tot
        x_s1 ./= tot
        x_s2 ./= tot
    end
    (x_c=x_c, x_s1=x_s1, x_s2=x_s2)
end

# ============================================================
# AGGREGATION
# ============================================================

function compute_aggregates_couples(par::Dict, grids::NamedTuple,
                                    hh_list::Vector, x_list::Vector,
                                    prices::Dict, benefit_fn::Function)
    n_ct = length(hh_list)
    nj = par[:n_ages]; nz = par[:n_z]; na = par[:n_a]
    Jr = par[:J_retire] - par[:J_start] + 1
    ag = grids.a_grid; zg = grids.z_grid

    # Per-thread accumulators to avoid race conditions
    nt = Threads.maxthreadid()
    A_t  = zeros(nt); L_t  = zeros(nt); C_t  = zeros(nt)
    tr_t = zeros(nt); ssb_t = zeros(nt)
    ss_oasi_t = zeros(nt); ss_hi_t = zeros(nt)
    nw_t = zeros(nt); nr_t = zeros(nt)

    @threads for ict in 1:n_ct
        tid = threadid()
        ct  = par[:couple_types][ict]
        wp  = ct[:weight]
        zp1 = ct[:zp1]; zp2 = ct[:zp2]
        hh  = hh_list[ict]; xd = x_list[ict]
        x_c = xd.x_c; x_s1 = xd.x_s1; x_s2 = xd.x_s2

        # --- Couple contributions ---
        for j in 1:nj
            ret = j >= Jr
            for iz1 in 1:nz, iz2 in 1:nz
                zv1 = exp(zg[iz1] + zp1 + grids.age_eff[j])
                zv2 = exp(zg[iz2] + zp2 + grids.age_eff[j])
                for ia in 1:na
                    m = x_c[ia,iz1,iz2,j] * wp
                    m < 1e-15 && continue
                    ai = ag[ia]
                    n1i = hh.couple.pol_n1[ia,iz1,iz2,j]
                    n2i = hh.couple.pol_n2[ia,iz1,iz2,j]
                    ci  = hh.couple.pol_c[ia,iz1,iz2,j]

                    A_t[tid] += ai * m
                    L_t[tid] += (zv1*n1i + zv2*n2i) * m
                    C_t[tid] += ci * m

                    yl1 = prices[:w]*zv1*n1i
                    yl2 = prices[:w]*zv2*n2i
                    yc  = prices[:pi_corp]*par[:zeta_income_corp]*prices[:psi_cap]*ai
                    yp  = prices[:pi_pass]*par[:zeta_income_pass]*prices[:psi_cap]*ai
                    yd  = prices[:r_G]*prices[:psi_debt]*ai

                    aime1 = hh.AIME1[iz1,iz2]
                    aime2 = hh.AIME2[iz2,iz1]
                    if ret
                        b1, b2 = benefit_fn(aime1, aime2, par, :couple)
                        sb = b1 + b2
                    else
                        sb = 0.0
                    end

                    d = tau_HH_couple_decomposed(yl1, yl2, yc, yp, yd, sb, ci, par)
                    tr_t[tid]      += d.total_tax * m
                    ss_oasi_t[tid] += d.ss_tax_rev_oasi * m
                    ss_hi_t[tid]   += d.ss_tax_rev_hi * m
                    ssb_t[tid]     += sb * m
                    if !ret
                        if yl1 > 1e-10; nw_t[tid] += m; end
                        if yl2 > 1e-10; nw_t[tid] += m; end
                    else
                        nr_t[tid] += 2.0 * m
                    end
                end
            end
        end

        # --- Survivor1 contributions ---
        for j in 1:nj
            ret = j >= Jr
            for iz1 in 1:nz
                zv1 = exp(zg[iz1] + zp1 + grids.age_eff[j])
                for ia in 1:na
                    m = x_s1[ia,iz1,j] * wp
                    m < 1e-15 && continue
                    ai  = ag[ia]
                    n1i = hh.surv1.pol_n[ia,iz1,j]
                    ci  = hh.surv1.pol_c[ia,iz1,j]

                    A_t[tid] += ai * m
                    L_t[tid] += zv1*n1i * m
                    C_t[tid] += ci * m

                    yl1 = prices[:w]*zv1*n1i
                    yc  = prices[:pi_corp]*par[:zeta_income_corp]*prices[:psi_cap]*ai
                    yp  = prices[:pi_pass]*par[:zeta_income_pass]*prices[:psi_cap]*ai
                    yd  = prices[:r_G]*prices[:psi_debt]*ai

                    aime_own = mean(hh.AIME1[iz1,:])
                    if ret
                        b1, _ = benefit_fn(aime_own, hh.AIME2_deceased_avg,
                                           par, :survivor1)
                        sb = b1
                    else
                        sb = 0.0
                    end

                    d = tau_HH_decomposed(yl1, yc, yp, yd, sb, ci, par)
                    tr_t[tid]      += d.total_tax * m
                    ss_oasi_t[tid] += d.ss_tax_rev_oasi * m
                    ss_hi_t[tid]   += d.ss_tax_rev_hi * m
                    ssb_t[tid]     += sb * m
                    if !ret; nw_t[tid] += m; else; nr_t[tid] += m; end
                end
            end
        end

        # --- Survivor2 contributions ---
        for j in 1:nj
            ret = j >= Jr
            for iz2 in 1:nz
                zv2 = exp(zg[iz2] + zp2 + grids.age_eff[j])
                for ia in 1:na
                    m = x_s2[ia,iz2,j] * wp
                    m < 1e-15 && continue
                    ai  = ag[ia]
                    n2i = hh.surv2.pol_n[ia,iz2,j]
                    ci  = hh.surv2.pol_c[ia,iz2,j]

                    A_t[tid] += ai * m
                    L_t[tid] += zv2*n2i * m
                    C_t[tid] += ci * m

                    yl2 = prices[:w]*zv2*n2i
                    yc  = prices[:pi_corp]*par[:zeta_income_corp]*prices[:psi_cap]*ai
                    yp  = prices[:pi_pass]*par[:zeta_income_pass]*prices[:psi_cap]*ai
                    yd  = prices[:r_G]*prices[:psi_debt]*ai

                    aime_own = mean(hh.AIME2[:,iz2])
                    if ret
                        _, b2 = benefit_fn(hh.AIME1_deceased_avg, aime_own,
                                           par, :survivor2)
                        sb = b2
                    else
                        sb = 0.0
                    end

                    d = tau_HH_decomposed(yl2, yc, yp, yd, sb, ci, par)
                    tr_t[tid]      += d.total_tax * m
                    ss_oasi_t[tid] += d.ss_tax_rev_oasi * m
                    ss_hi_t[tid]   += d.ss_tax_rev_hi * m
                    ssb_t[tid]     += sb * m
                    if !ret; nw_t[tid] += m; else; nr_t[tid] += m; end
                end
            end
        end
    end

    (A=sum(A_t), L=sum(L_t), C=sum(C_t),
     tax_rev_HH=sum(tr_t), SS_BEN=sum(ssb_t),
     ss_benefit_tax_rev_oasi=sum(ss_oasi_t),
     ss_benefit_tax_rev_hi=sum(ss_hi_t),
     ss_benefit_tax_rev=sum(ss_oasi_t)+sum(ss_hi_t),
     n_workers=sum(nw_t), n_retirees=sum(nr_t))
end

# ============================================================
# PAYROLL STATS
# ============================================================

function payroll_stats_couples(par::Dict, grids::NamedTuple,
                               hh_list::Vector, x_list::Vector,
                               prices::Dict)
    n_ct = length(hh_list)
    nz = par[:n_z]; na = par[:n_a]; nj = par[:n_ages]
    Jr = par[:J_retire] - par[:J_start] + 1
    zg = grids.z_grid; ag = grids.a_grid
    cap = par[:ss_cap_frac] * par[:avg_wage_mu]

    nt = Threads.maxthreadid()
    tot_t = zeros(nt); tax_t = zeros(nt); nw_t = zeros(nt)

    @threads for ict in 1:n_ct
        tid = threadid()
        ct  = par[:couple_types][ict]
        wp  = ct[:weight]; zp1 = ct[:zp1]; zp2 = ct[:zp2]
        hh  = hh_list[ict]; xd = x_list[ict]

        # Couple payroll
        for j in 1:nj
            j >= Jr && continue
            for iz1 in 1:nz, iz2 in 1:nz
                zv1 = exp(zg[iz1] + zp1 + grids.age_eff[j])
                zv2 = exp(zg[iz2] + zp2 + grids.age_eff[j])
                for ia in 1:na
                    m = xd.x_c[ia,iz1,iz2,j] * wp
                    m < 1e-15 && continue
                    yl1 = prices[:w]*zv1*hh.couple.pol_n1[ia,iz1,iz2,j]
                    yl2 = prices[:w]*zv2*hh.couple.pol_n2[ia,iz1,iz2,j]
                    if yl1 > 1e-10
                        tot_t[tid] += yl1 * m
                        tax_t[tid] += min(yl1, cap) * m
                        nw_t[tid]  += m
                    end
                    if yl2 > 1e-10
                        tot_t[tid] += yl2 * m
                        tax_t[tid] += min(yl2, cap) * m
                        nw_t[tid]  += m
                    end
                end
            end
        end

        # Survivor1 payroll
        for j in 1:nj
            j >= Jr && continue
            for iz1 in 1:nz
                zv1 = exp(zg[iz1] + zp1 + grids.age_eff[j])
                for ia in 1:na
                    m = xd.x_s1[ia,iz1,j] * wp
                    m < 1e-15 && continue
                    yl1 = prices[:w]*zv1*hh.surv1.pol_n[ia,iz1,j]
                    tot_t[tid] += yl1 * m
                    tax_t[tid] += min(yl1,cap) * m
                    nw_t[tid]  += m
                end
            end
        end

        # Survivor2 payroll
        for j in 1:nj
            j >= Jr && continue
            for iz2 in 1:nz
                zv2 = exp(zg[iz2] + zp2 + grids.age_eff[j])
                for ia in 1:na
                    m = xd.x_s2[ia,iz2,j] * wp
                    m < 1e-15 && continue
                    yl2 = prices[:w]*zv2*hh.surv2.pol_n[ia,iz2,j]
                    tot_t[tid] += yl2 * m
                    tax_t[tid] += min(yl2,cap) * m
                    nw_t[tid]  += m
                end
            end
        end
    end

    tot = sum(tot_t); tax = sum(tax_t); nw = sum(nw_t)
    (total=tot, taxable=tax, n_workers=nw,
     avg_earn=tot/max(nw,1e-10),
     pct_above=1.0-tax/max(tot,1e-10))
end

# ============================================================
# PARAMETERS
# ============================================================

function create_params(; benefit_fn::Function=benefits_current_law,
                        aime_modifier=nothing)
    par = Dict{Symbol,Any}(
        :J_start  => 21,  :J_retire => 67, :J_max => 100, :n_ages => 80,
        :g_pop    => 0.000, :aime_nwc_frac => 0, 
        :alpha    => 0.36, :delta => 0.06, :eta => 0.0,
        :zeta_income_corp => 0.55, :zeta_income_pass => 0.45,
        :tau_statutory_corp => 0.21,
        :phi_exp_corp  => 0.50, :phi_int_corp => 1.00,
        :zeta_taxbase_corp => 1.00, :zeta_ded_corp => 0.02,
        :zeta_cred_corp => 0.01,  :zeta_other_corp => 0.02,
        :tau_top_pit   => 0.37,
        :phi_exp_pass  => 0.50, :phi_int_pass => 1.00, :zeta_other_pass => 0.02,
        :nu1     => 3.5, :nu2 => nothing, :h => 1.0, :leverage_ratio_target => 0.32,
        :beta    => 0.97, :gamma => 0.30, :sigma => 2.00,
        :first_rr => 0.9, :second_rr => 0.32, :third_rr => 0.15,
        # Permanent types (individual)
        :z_perm_vals => [-0.45, 0.10, 0.85, 1.20],
        :p_perm_vals => [ 0.40, 0.45, 0.12, 0.03],

        # Assortative mating
        :rho_assort => 0.6,

        :sigma_trans  => 0.063, :sigma_pers => 0.007, :rho_pers => 0.990,

        # Individual (single/survivor) brackets
        :ord_brackets  => Float64[0,9700,39475,84200,160725,204100,510300],
        :ord_rates     => Float64[0.10,0.12,0.22,0.24,0.32,0.35,0.37],
        :pref_brackets => Float64[0,39375,434550],
        :pref_rates    => Float64[0.0,0.15,0.20],

        # MFJ brackets
        :ord_brackets_mfj  => Float64[0,19400,78950,168400,321450,408200,612350],
        :ord_rates_mfj     => Float64[0.10,0.12,0.22,0.24,0.32,0.35,0.37],
        :pref_brackets_mfj => Float64[0,78750,488850],
        :pref_rates_mfj    => Float64[0.0,0.15,0.20],

        :payroll_rate  => 0.106, :payroll_cap => 176100.0,
        :tau_con => 0.0, :tau_lumpsum => 0.0,
        :theta_corp_ORD  => 0.60, :theta_corp_PREF => 0.40,
        :theta_lab_ORD   => 0.95, :theta_lab_PT    => 1.00,
        :theta_pass_ORD  => 1.00, :theta_ss_ORD    => 0.85,
        :ss_tax_thresh1  => 32000.0,  # MFJ thresholds
        :ss_tax_thresh2  => 44000.0,
        :ss_tax_use_provisional => true,
        :ss_thresh_indexing => "cpi",
        :ss_bend1_frac => 0.2335, :ss_bend2_frac => 1.4078, :ss_cap_frac => 2.796,
        :avg_wage_mu   => 1.0,
        :ss_bend1 => 1226.0*12, :ss_bend2 => 7391.0*12, :ss_cap => 176100.0,
        :r_G => 0.03, :debt_to_gdp => 0.78,
        :G_residual_share => 0.18, :T_residual_share => 0.03,
        :closure_year => 20, :r_K_world => 0.05,
        :zeta_debt_foreign_takeup   => 0.40,
        :zeta_capital_foreign_takeup => 0.50,
        :tau_corp_FOR => 0.17, :tau_pass_FOR => 0.08,
        :n_a => 30, :n_z => 5,
        :a_min => 0.0, :a_max => 50.0,
        :tol_vfi => 1e-6, :tol_equil => 1e-4,
        :max_iter_vfi => 500, :max_iter_equil => 15,
        :mu_dollar       => 50000.0,
        :target_avg_earn => 63000.0,
        :g_A_index       => 0.012,

        # Spousal cap for capped-spousal reform
        :spousal_cap => 0,

        # Caregiver credit parameters
        :cg_start_age   => 25,
        :cg_end_age     => 38,
        :cg_credit_level => 0.50,   # fraction of avg_wage_mu
        :cg_n_threshold  => 0.10,
        :cg_max_years    => 10,

        # Store benefit function and aime modifier
        :benefit_fn    => benefit_fn,
        :aime_modifier => aime_modifier,
    )
    par[:spousal_cap] = compute_PIA(7.25*2080/12/par[:mu_dollar], par)
    par


end

# ============================================================
# COUPLE-TYPE CONSTRUCTION
# ============================================================

function build_couple_types(par::Dict)
    zpv = par[:z_perm_vals]
    ppv = par[:p_perm_vals]
    rho = par[:rho_assort]
    n   = length(zpv)

    # Joint weights: independence * assortative mating adjustment
    P = zeros(n, n)
    for i in 1:n, j in 1:n
        P[i,j] = ppv[i] * ppv[j]
        # Increase weight along diagonal (same-type couples)
        sim = 1.0 - abs(i-j) / (n-1)
        P[i,j] *= exp(rho * sim)
    end
    P ./= sum(P)

    types = Vector{Dict{Symbol,Any}}()
    for i in 1:n, j in 1:n
        push!(types, Dict{Symbol,Any}(
            :zp1 => zpv[i],
            :zp2 => zpv[j],
            :ip1 => i, :ip2 => j,
            :weight => P[i,j]
        ))
    end
    par[:couple_types] = types
    par[:couple_weights] = P
    @printf("  Built %d couple types (%.0f%% same-type)\n",
            length(types), sum(diag(P))*100)
    par
end

# ============================================================
# GRIDS
# ============================================================

function setup_grids(par::Dict)
    pers    = tauchen(par[:n_z], par[:rho_pers], sqrt(par[:sigma_pers]))
    a_grid  = collect(range(par[:a_min], par[:a_max], length=par[:n_a]))
    age_eff = Float64[age_efficiency(a, par[:J_start])
                      for a in par[:J_start]:par[:J_max]]
    Jr = par[:J_retire] - par[:J_start] + 1
    age_eff[Jr:par[:n_ages]] .= 0.0
    (a_grid=a_grid, z_grid=pers.z_grid, Pi_z=pers.Pi,
     age_eff=age_eff,
     surv=survival_probs(par),
     surv_m=survival_probs_male(par),
     surv_f=survival_probs_female(par))
end

# ============================================================
# STEADY STATE SOLVER
# ============================================================

function solve_steady_state(par::Dict;
                            benefit_fn::Function=benefits_current_law,
                            aime_modifier=nothing)
    println("=== Solving Steady State (Couples Model) ===")
    @printf("  Benefit rule: %s\n", string(Symbol(benefit_fn)))
    @printf("  z=[%s]  p=[%s]  g_pop=%.3f  target_earn=\$%s\n",
            join(par[:z_perm_vals], "/"),
            join(par[:p_perm_vals], "/"),
            par[:g_pop],
            format_comma(par[:target_avg_earn]))

    build_couple_types(par)
    grids  = setup_grids(par)
    n_ct   = length(par[:couple_types])
    KL     = 3.0; pK = 1.0

    hh_list = Vector{Any}(undef, n_ct)
    x_list  = Vector{Any}(undef, n_ct)
    agg     = nothing
    prices  = Dict{Symbol,Any}()
    w=0.0; rK=0.0; Kn=0.0; La=0.0; Aa=0.0; psc=0.95
    dep=0.0; wm=0.0; rm=0.0; ps=nothing

    for it in 1:par[:max_iter_equil]
        @printf("\n Iter %d: K/L=%.4f mu\$=%.0f avg_w_mu=%.4f\n",
                it, KL, par[:mu_dollar], par[:avg_wage_mu])
        w   = mpl(KL, 1.0, par[:alpha])
        rK  = mpk(KL, 1.0, par[:alpha])
        par[:pi_approx] = rK - par[:delta]
        pic = rK - par[:delta]
        par[:nu2] = calibrate_nu2(par, pic)
        Yg  = production(KL, 1.0, par[:alpha])
        Dg  = par[:debt_to_gdp]*Yg
        Ag  = pK*KL + Dg
        psc = max(0.01, (Ag-Dg)/Ag)
        psd = 1.0 - psc
        prices = Dict{Symbol,Any}(
            :w       => w,
            :pi_corp => pic,
            :pi_pass => pic,
            :r_G     => par[:r_G],
            :pK      => pK,
            :beq     => 0.0,
            :psi_cap => psc,
            :psi_debt=> psd
        )

        # ---- Solve each couple type in parallel ----
        logs = Vector{String}(undef, n_ct)
        @threads for ict in 1:n_ct
            ct = par[:couple_types][ict]
            logs[ict] = @sprintf("  [type %d/%d zp1=%.2f zp2=%.2f w=%.3f]",
                                 ict, n_ct, ct[:zp1], ct[:zp2], ct[:weight])
            hh_list[ict] = solve_household_couples(par, grids, prices,
                                                   ct[:zp1], ct[:zp2],
                                                   benefit_fn=benefit_fn,
                                                   aime_modifier=aime_modifier)
            x_list[ict] = compute_distribution_couples(par, grids,
                                                       hh_list[ict].couple.pol_a,
                                                       hh_list[ict].surv1.pol_a,
                                                       hh_list[ict].surv2.pol_a)
        end
        for msg in logs; println(msg); end

        agg = compute_aggregates_couples(par, grids, hh_list, x_list,
                                         prices, benefit_fn)
        La  = agg.L; Aa = agg.A
        Kn  = max(0.01, psc*Aa/pK)
        KLn = La > 0 ? Kn/La : KL

        ps = payroll_stats_couples(par, grids, hh_list, x_list, prices)
        par[:mu_dollar]   = par[:target_avg_earn] / max(ps.avg_earn, 1e-10)
        par[:avg_wage_mu] = ps.avg_earn

        wm = agg.n_workers
        rm = agg.n_retirees
        dep     = rm / max(wm, 1e-10)
        cost    = agg.SS_BEN / max(ps.taxable, 1e-10) * 100
        avg_ben = agg.SS_BEN / max(rm, 1e-10) * par[:mu_dollar]
        err     = abs(KLn - KL) / max(abs(KL), 1e-8)
        mean_aime_chg = mean([hh.aime_pct_change for hh in hh_list])

        @printf("  err=%.5f dep=%.3f pct_above=%.1f%% cost=%.2f%% avg_ben=\$%s AIME_chg=%.1f%%\n",
                err, dep, ps.pct_above*100, cost,
                format_comma(round(avg_ben)), mean_aime_chg)

        if err < par[:tol_equil]
            println("  Converged!")
            Y_ss = production(Kn, La, par[:alpha])
            return (par=par, grids=grids, hh_list=hh_list, x_list=x_list,
                    agg=agg, prices=prices, K=Kn, L=La, Y=Y_ss,
                    D=par[:debt_to_gdp]*Y_ss,
                    w=w, r_K=rK, KL=KL, ps=ps,
                    dep_ratio=dep, worker_mass=wm, retiree_mass=rm,
                    benefit_fn=benefit_fn)
        end
        KL = 0.7*KL + 0.3*KLn
    end

    println("  Did not converge")
    Y_ss = production(max(0.01, psc*Aa/pK), La, par[:alpha])
    (par=par, grids=grids, hh_list=hh_list, x_list=x_list,
     agg=agg, prices=prices,
     K=max(0.01, psc*Aa/pK), L=La, Y=Y_ss,
     D=par[:debt_to_gdp]*Y_ss,
     w=w, r_K=rK, KL=KL, ps=ps,
     dep_ratio=dep, worker_mass=wm, retiree_mass=rm,
     benefit_fn=benefit_fn)
end

# ============================================================
# EXTRACT STEADY-STATE RATIOS
# ============================================================

function extract_ss_ratios(ss; benefit_fn::Function=benefits_current_law)
    par = ss.par; mu = par[:mu_dollar]
    ps  = payroll_stats_couples(par, ss.grids, ss.hh_list, ss.x_list, ss.prices)
    agg = ss.agg

    wm = agg.n_workers
    rm = agg.n_retirees
    dep_ratio = rm / max(wm, 1e-10)

    fica_rev      = par[:payroll_rate] * ps.taxable
    ss_ben        = agg.SS_BEN
    total_tax_rev = agg.tax_rev_HH
    ss_btax_rev   = agg.ss_benefit_tax_rev
    ss_btax_oasi  = agg.ss_benefit_tax_rev_oasi
    ss_btax_hi    = agg.ss_benefit_tax_rev_hi

    Y = ss.Y; K = ss.K; L = ss.L; w = ss.w; rK = ss.r_K
    pic = rK - par[:delta]
    corp_profit = par[:zeta_income_corp]*Y - w*par[:zeta_income_corp]*L -
                  par[:delta]*K*par[:zeta_income_corp]
    corp_tax = par[:tau_statutory_corp] *
               max(0.0, corp_profit) *
               (1 - par[:phi_exp_corp]*0.3 - par[:zeta_ded_corp])

    (Y=Y, K=K, L=L, w=w, C=agg.C, A=agg.A,
     fica_rev=fica_rev, ss_ben=ss_ben,
     ss_balance=fica_rev-ss_ben,
     ss_benefit_tax_rev=ss_btax_rev,
     ss_benefit_tax_rev_oasi=ss_btax_oasi,
     ss_benefit_tax_rev_hi=ss_btax_hi,
     total_tax_rev=total_tax_rev,
     corp_tax=corp_tax,
     income_tax=total_tax_rev-fica_rev,
     G=par[:G_residual_share]*Y,
     D=ss.D,
     dep_ratio=dep_ratio,
     worker_mass=wm, retiree_mass=rm,
     payroll_rate=par[:payroll_rate],
     avg_earnings=ps.avg_earn,
     taxable_payroll=ps.taxable,
     total_payroll=ps.total,
     pct_above_cap=ps.pct_above,
     mu_dollar=mu,
     ss_ben_per_retiree=ss_ben/max(rm,1e-10),
     fica_per_worker=fica_rev/max(wm,1e-10),
     earn_per_worker=ps.total/max(wm,1e-10),
     ss_benefit_tax_per_retiree=ss_btax_rev/max(rm,1e-10),
     ss_benefit_tax_oasi_per_retiree=ss_btax_oasi/max(rm,1e-10),
     ss_benefit_tax_hi_per_retiree=ss_btax_hi/max(rm,1e-10))
end

# ============================================================
# PROJECTION HELPERS
# ============================================================

function build_dependency_path(n_years::Int;
                               dep_start::Float64=0.29,
                               dep_end::Float64=0.45,
                               speed::Float64=0.08,
                               midpoint::Float64=25.0)
    [dep_start + (dep_end-dep_start) / (1.0 + exp(-speed*(t-midpoint)))
     for t in 1:n_years]
end

awi_cap_path(cap_base::Float64, cum_wage::Vector{Float64}) =
    cap_base .* cum_wage

# ============================================================
# PROJECT ECONOMY — with reform scoring
# ============================================================

function project_economy(
    ss;
    n_years::Int          = 75,
    start_year::Int       = 2025,
    g_A::Float64          = 0.012,
    g_pop::Float64        = 0.005,
    inflation::Float64    = 0.025,
    dep_path              = nothing,
    ss_cola               = "wage",
    cap_base_dollars::Float64 = 176100.0,
    bracket_indexing::String  = "cpi",
    trust_fund_init::Float64  = 2.3e12,
    trust_fund_rate::Float64  = 0.035,
    ss_reform             = nothing,
    reform_year           = nothing,
    reform_phase_in::Int  = 10,
    label::String         = "Current Law",
    scenario_periods      = nothing,
    thresh_sens::Float64  = 0.3,
    pia_factor_indexing::Bool = false
)
    @printf("=== Projecting %d years: %d-%d [%s] ===\n",
            n_years, start_year, start_year+n_years-1, label)

    base = extract_ss_ratios(ss)
    mu   = base.mu_dollar; par = ss.par

    ssa_covered_workers = 180e6
    N_hh = ssa_covered_workers / base.worker_mass
    @printf("  N_hh: %.1fM (anchored to %sM covered workers)\n",
            N_hh/1e6, format_comma(round(ssa_covered_workers/1e6)))
    @printf("  Implied GDP: \$%.1fT\n",
            base.Y * mu * N_hh / 1e12)

    has_reform = !isnothing(ss_reform) && !isnothing(reform_year)
    ref = nothing
    if has_reform
        ref = extract_ss_ratios(ss_reform)
        @printf("  Reform at %d, phase-in %d yrs\n", reform_year, reform_phase_in)
    end

    if isnothing(dep_path)
        dep_path = build_dependency_path(n_years, dep_start=base.dep_ratio)
    end
    length(dep_path) < n_years &&
        (dep_path = vcat(dep_path, fill(dep_path[end], n_years - length(dep_path))))

    year_cal  = collect(start_year:(start_year+n_years-1))
    g_A_vec   = fill(g_A,       n_years)
    inf_vec   = fill(inflation,  n_years)
    g_pop_vec = fill(g_pop,     n_years)

    if !isnothing(scenario_periods)
        for ep in scenario_periods
            idx = findall(y -> y >= ep[:year_start] && y <= ep[:year_end], year_cal)
            isempty(idx) && continue
            haskey(ep,:g_A)       && (g_A_vec[idx]   .= ep[:g_A])
            haskey(ep,:inflation) && (inf_vec[idx]   .= ep[:inflation])
            haskey(ep,:g_pop)     && (g_pop_vec[idx] .= ep[:g_pop])
            @printf("  Scenario episode %d-%d: g_A=%.3f inf=%.3f\n",
                    ep[:year_start], ep[:year_end],
                    get(ep,:g_A,g_A), get(ep,:inflation,inflation))
        end
    end

    cum_A     = cumprod([1.0; (1.0 .+ g_A_vec[2:end])])
    cum_pop   = cumprod([1.0; (1.0 .+ g_pop_vec[2:end])])
    cum_price = cumprod([1.0; (1.0 .+ inf_vec[2:end])])
    cum_wage  = cumprod([1.0; (1.0 .+ g_A_vec[2:end] .+ inf_vec[2:end])])
    cum_price_lag = [1.0; cum_price[1:end-1]]

    cap_nom_vec = awi_cap_path(cap_base_dollars, cum_wage)
    @printf("  Cap: \$%s -> \$%s (yr 10) -> \$%s (yr %d)\n",
            format_comma(cap_base_dollars),
            format_comma(cap_nom_vec[min(10,n_years)]),
            format_comma(cap_nom_vec[n_years]), n_years)

    cum_new_ben = ones(n_years)
    if ss_cola == "wage"
        cum_new_ben = copy(cum_A)
    elseif ss_cola == "cpi"
        cum_new_ben = ones(n_years)
    else
        rate = Float64(ss_cola)
        for t in 2:n_years; cum_new_ben[t] = cum_new_ben[t-1]*(1+rate); end
    end

    working_years = par[:J_retire] - par[:J_start]

    btax_rate_oasi_base = base.ss_benefit_tax_oasi_per_retiree /
                          max(base.ss_ben_per_retiree, 1e-12)
    btax_rate_hi_base   = base.ss_benefit_tax_hi_per_retiree /
                          max(base.ss_ben_per_retiree, 1e-12)
    btax_rate_oasi_ref = 0.0; btax_rate_hi_ref = 0.0
    if has_reform
        btax_rate_oasi_ref = ref.ss_benefit_tax_oasi_per_retiree /
                             max(ref.ss_ben_per_retiree, 1e-12)
        btax_rate_hi_ref   = ref.ss_benefit_tax_hi_per_retiree /
                             max(ref.ss_ben_per_retiree, 1e-12)
    end

    GDP_real                 = zeros(n_years)
    GDP_nominal              = zeros(n_years)
    avg_earnings_nominal     = zeros(n_years)
    payroll_cap_nom          = zeros(n_years)
    taxable_payroll_nom      = zeros(n_years)
    fica_revenue_nom         = zeros(n_years)
    ss_outlays_nom           = zeros(n_years)
    avg_ben_per_retiree_nom  = zeros(n_years)
    ss_benefit_tax_oasi_nom  = zeros(n_years)
    ss_benefit_tax_hi_nom    = zeros(n_years)
    ss_benefit_tax_total_nom = zeros(n_years)
    trust_fund_interest_nom  = zeros(n_years)
    ss_cash_flow_nom         = zeros(n_years)
    ss_total_income_nom      = zeros(n_years)
    ss_balance_nom           = zeros(n_years)
    trust_fund_nom_v         = zeros(n_years)
    fica_rate_eff            = zeros(n_years)
    ss_cost_rate             = zeros(n_years)
    ss_cash_flow_pct         = zeros(n_years)
    ss_balance_pct           = zeros(n_years)
    total_tax_rev_nom        = zeros(n_years)
    govt_spending_nom        = zeros(n_years)
    debt_nom                 = zeros(n_years)
    debt_to_gdp_v            = zeros(n_years)

    # --- Precompute AIME and bend points for PIA factor indexing ---
    b1_mu = par[:ss_bend1] / mu
    b2_mu = par[:ss_bend2] / mu
    avg_aime_mu = b2_mu
    pia_base_level = base.ss_ben_per_retiree
    for _ in 1:20
        pia_try = 0.90*min(avg_aime_mu, b1_mu) +
                  0.32*max(0.0, min(avg_aime_mu, b2_mu) - b1_mu) +
                  0.15*max(0.0, avg_aime_mu - b2_mu)
        err = pia_try - pia_base_level
        mr  = avg_aime_mu <= b1_mu ? 0.90 : avg_aime_mu <= b2_mu ? 0.32 : 0.15
        avg_aime_mu -= err / max(mr, 0.01)
        avg_aime_mu  = max(avg_aime_mu, 0.01)
    end
    pia_current_law = 0.90*min(avg_aime_mu, b1_mu) +
                      0.32*max(0.0, min(avg_aime_mu, b2_mu) - b1_mu) +
                      0.15*max(0.0, avg_aime_mu - b2_mu)

    trust_fund   = trust_fund_init
    avg_ben_real = base.ss_ben_per_retiree

    for t in 1:n_years
        yr  = year_cal[t]
        lam = 0.0
        if has_reform && yr >= reform_year
            lam = min(1.0, (yr - reform_year) / reform_phase_in)
        end

        dep_t     = dep_path[t]
        dep_scale = dep_t / base.dep_ratio
        bl(bv,rv) = (1-lam)*bv + lam*rv

        if lam > 0
            fica_rate_t      = bl(base.payroll_rate,    ref.payroll_rate)
            pct_above_t      = bl(base.pct_above_cap,   ref.pct_above_cap)
            inc_tax_t        = bl(base.income_tax,      ref.income_tax)
            corp_tax_t       = bl(base.corp_tax,        ref.corp_tax)
            Y_t              = bl(base.Y,               ref.Y)
            earn_pw_t        = bl(base.earn_per_worker, ref.earn_per_worker)
            btax_oasi_rate_t = bl(btax_rate_oasi_base,  btax_rate_oasi_ref)
            btax_hi_rate_t   = bl(btax_rate_hi_base,    btax_rate_hi_ref)
            new_ben_scale_t  = bl(base.ss_ben_per_retiree,
                                  ref.ss_ben_per_retiree) / base.ss_ben_per_retiree
        else
            fica_rate_t      = base.payroll_rate
            pct_above_t      = base.pct_above_cap
            inc_tax_t        = base.income_tax
            corp_tax_t       = base.corp_tax
            Y_t              = base.Y
            earn_pw_t        = base.earn_per_worker
            btax_oasi_rate_t = btax_rate_oasi_base
            btax_hi_rate_t   = btax_rate_hi_base
            new_ben_scale_t  = 1.0
        end

        # --- PIA factor indexing: override new_ben_scale_t if enabled ---
        if pia_factor_indexing && !isnothing(reform_year) && yr >= reform_year
            t_ref = findfirst(==(reform_year), year_cal)
            price_wage_ratio = isnothing(t_ref) ? 1.0 : cum_A[t_ref] / cum_A[t]
            f1 = 0.90 * price_wage_ratio
            f2 = 0.32 * price_wage_ratio
            f3 = 0.15 * price_wage_ratio
            pia_yr = f1*min(avg_aime_mu, b1_mu) +
                     f2*max(0.0, min(avg_aime_mu, b2_mu) - b1_mu) +
                     f3*max(0.0, avg_aime_mu - b2_mu)
            new_ben_scale_t = pia_yr / max(pia_current_law, 1e-10)
        end

        N_hh_t     = N_hh * cum_pop[t]
        GDP_real_t = Y_t * cum_A[t] * mu * N_hh_t
        GDP_nom_t  = GDP_real_t * cum_price[t]
        avg_earn_t = earn_pw_t * cum_A[t] * mu * cum_price[t]

        total_pay_nom_t   = base.total_payroll * cum_A[t] * mu * cum_price[t] * N_hh_t
        wage_cap_ratio    = cum_wage[t] / (cap_nom_vec[t] / cap_base_dollars)
        pct_above_adj     = min(0.50, pct_above_t * wage_cap_ratio)
        taxable_pay_nom_t = total_pay_nom_t * (1.0 - pct_above_adj)
        fica_rev_nom_t    = fica_rate_t * taxable_pay_nom_t

        phi_t          = min(0.20, 1.0 / (working_years * dep_t))
        new_ben_real_t = base.ss_ben_per_retiree * cum_new_ben[t] * new_ben_scale_t
        avg_ben_real   = (1.0-phi_t)*avg_ben_real + phi_t*new_ben_real_t
        avg_ben_nom    = avg_ben_real * mu * cum_price_lag[t]
        n_ret_t        = base.retiree_mass * dep_scale * N_hh_t
        ss_outlays_nom_t = avg_ben_nom * n_ret_t

        base_frac_oasi   = max(btax_oasi_rate_t / 0.85, 1e-6)
        creep_oasi       = 1.0 / (1 + (1/base_frac_oasi - 1) *
                                  exp(-thresh_sens * log(cum_price[t])))
        btax_oasi_adj    = (creep_oasi / base_frac_oasi) * btax_oasi_rate_t

        base_frac_hi     = max(btax_hi_rate_t / 0.85, 1e-6)
        creep_hi         = 1.0 / (1 + (1/base_frac_hi - 1) *
                                  exp(-0.6 * thresh_sens * log(cum_price[t])))
        btax_hi_adj      = (creep_hi / base_frac_hi) * btax_hi_rate_t

        ss_btax_oasi_nom_t  = btax_oasi_adj * avg_ben_nom * n_ret_t
        ss_btax_hi_nom_t    = btax_hi_adj   * avg_ben_nom * n_ret_t
        ss_btax_total_nom_t = ss_btax_oasi_nom_t + ss_btax_hi_nom_t

        r_nom_t        = (1.0 + par[:r_G]) * (1.0 + inf_vec[t]) - 1.0
        tf_interest_t  = trust_fund > 0 ? trust_fund * trust_fund_rate : 0.0
        ss_cash_flow_t = fica_rev_nom_t + ss_btax_oasi_nom_t - ss_outlays_nom_t
        ss_total_inc_t = fica_rev_nom_t + ss_btax_oasi_nom_t + tf_interest_t
        ss_balance_t   = ss_total_inc_t - ss_outlays_nom_t
        trust_fund    += ss_balance_t

        other_tax_nom_t = (inc_tax_t + corp_tax_t) * cum_A[t] * mu * cum_price[t] * N_hh_t
        total_tax_nom_t = fica_rev_nom_t + other_tax_nom_t
        G_nom_t         = base.G * cum_A[t] * mu * cum_price[t] * N_hh_t

        debt_t = t == 1 ?
            base.D * mu * N_hh * cum_price[t] :
            debt_nom[t-1] * 1e9 * (1 + r_nom_t) +
            (G_nom_t + ss_outlays_nom_t - total_tax_nom_t)

        GDP_real[t]                = GDP_real_t / 1e9
        GDP_nominal[t]             = GDP_nom_t  / 1e9
        avg_earnings_nominal[t]    = avg_earn_t
        payroll_cap_nom[t]         = cap_nom_vec[t]
        taxable_payroll_nom[t]     = taxable_pay_nom_t / 1e9
        fica_revenue_nom[t]        = fica_rev_nom_t    / 1e9
        ss_outlays_nom[t]          = ss_outlays_nom_t  / 1e9
        avg_ben_per_retiree_nom[t] = avg_ben_nom / 1e3
        ss_benefit_tax_oasi_nom[t] = ss_btax_oasi_nom_t  / 1e9
        ss_benefit_tax_hi_nom[t]   = ss_btax_hi_nom_t    / 1e9
        ss_benefit_tax_total_nom[t]= ss_btax_total_nom_t / 1e9
        trust_fund_interest_nom[t] = tf_interest_t / 1e9
        ss_cash_flow_nom[t]        = ss_cash_flow_t / 1e9
        ss_total_income_nom[t]     = ss_total_inc_t / 1e9
        ss_balance_nom[t]          = ss_balance_t   / 1e9
        trust_fund_nom_v[t]        = trust_fund     / 1e9
        fica_rate_eff[t]           = fica_rate_t * 100
        ss_cost_rate[t]            = ss_outlays_nom_t / max(taxable_pay_nom_t,1) * 100
        ss_cash_flow_pct[t]        = ss_cash_flow_t  / max(taxable_pay_nom_t,1) * 100
        ss_balance_pct[t]          = ss_balance_t    / max(taxable_pay_nom_t,1) * 100
        total_tax_rev_nom[t]       = total_tax_nom_t / 1e9
        govt_spending_nom[t]       = G_nom_t / 1e9
        debt_nom[t]                = debt_t  / 1e9
        debt_to_gdp_v[t]           = debt_t  / GDP_nom_t
    end

    dep_yr = findall(v -> v < 0, trust_fund_nom_v)
    depl   = isempty(dep_yr) ? nothing : year_cal[dep_yr[1]]
    cf_yr  = findall(v -> v < 0, ss_cash_flow_nom)
    cf_def = isempty(cf_yr)  ? nothing : year_cal[cf_yr[1]]

    # --- Print actuarial summary ---
    windows = filter(w -> w <= n_years, [10, 30, 75])
    println("\n--- Actuarial Summary [$label] ---")
    @printf("  %-8s %10s %10s %10s %12s %10s\n",
            "Window","FICA(\$B)","Outlays(\$B)","Balance(\$B)","TF End(\$B)","CostRate")
    println(repeat("-", 68))
    for w in windows
        @printf("  %-8s %10.1f %10.1f %10.1f %12.1f %9.2f%%\n",
                "$(w)-yr",
                sum(fica_revenue_nom[1:w]),
                sum(ss_outlays_nom[1:w]),
                sum(ss_balance_nom[1:w]),
                trust_fund_nom_v[w],
                mean(ss_cost_rate[1:w]))
    end
    if !isnothing(depl)
        @printf("\n  TF depletion:        %d\n", depl)
    else
        @printf("\n  TF solvent through:  %d\n", start_year+n_years-1)
    end
    !isnothing(cf_def) && @printf("  Cash-flow deficit:   %d\n", cf_def)

    # --- Year-by-year table ---
    show_t = sort(unique(filter(t -> t <= n_years, [1,2,3,5,10,15,20,25,30,40,50,75])))
    println("\n--- Year-by-Year Projection [$label] (selected years) ---")
    @printf("  %-6s %-4s %5s %5s %9s %8s %8s %8s %8s %8s %8s\n",
            "Year","Dep","gA%","CPI%","Cap(\$K)","CostRate",
            "CshFlow","Balance","TF(\$T)","AvgBen\$K","GDP\$T")
    println(repeat("-", 96))
    for t in show_t
        @printf("  %-6d %4.3f %4.1f%% %4.1f%% %9.1f %7.2f%% %7.2f%% %7.2f%% %8.2f %8.1f %8.1f\n",
                year_cal[t], dep_path[t],
                g_A_vec[t]*100, inf_vec[t]*100,
                payroll_cap_nom[t]/1e3,
                ss_cost_rate[t], ss_cash_flow_pct[t], ss_balance_pct[t],
                trust_fund_nom_v[t]/1000,
                avg_ben_per_retiree_nom[t],
                GDP_nominal[t]/1000)
    end

    (year=year_cal, t=1:n_years,
     dep_ratio=dep_path, price_level=cum_price,
     cum_productivity=cum_A, cum_population=cum_pop,
     g_A_annual=g_A_vec, inflation_annual=inf_vec,
     GDP_real=GDP_real, GDP_nominal=GDP_nominal,
     avg_earnings_nominal=avg_earnings_nominal,
     payroll_cap_nom=payroll_cap_nom,
     taxable_payroll_nom=taxable_payroll_nom,
     fica_revenue_nom=fica_revenue_nom,
     ss_outlays_nom=ss_outlays_nom,
     avg_ben_per_retiree_nom=avg_ben_per_retiree_nom,
     ss_benefit_tax_oasi_nom=ss_benefit_tax_oasi_nom,
     ss_benefit_tax_hi_nom=ss_benefit_tax_hi_nom,
     ss_benefit_tax_total_nom=ss_benefit_tax_total_nom,
     trust_fund_interest_nom=trust_fund_interest_nom,
     ss_cash_flow_nom=ss_cash_flow_nom,
     ss_total_income_nom=ss_total_income_nom,
     ss_balance_nom=ss_balance_nom,
     trust_fund_nom=trust_fund_nom_v,
     fica_rate_eff=fica_rate_eff,
     ss_cost_rate=ss_cost_rate,
     ss_cash_flow_pct=ss_cash_flow_pct,
     ss_balance_pct=ss_balance_pct,
     total_tax_rev_nom=total_tax_rev_nom,
     govt_spending_nom=govt_spending_nom,
     debt_nom=debt_nom,
     debt_to_gdp=debt_to_gdp_v,
     depletion_year=depl,
     cashflow_deficit_year=cf_def,
     label=label)
end

# ============================================================
# CONVENIENCE: run all four policy variants
# ============================================================

function score_all_reforms(;
    n_years::Int     = 75,
    start_year::Int  = 2025,
    reform_year::Int = 2027,
    reform_phase_in::Int = 10,
    kwargs...
)
    println("=" ^ 60)
    println("  SCORING SPOUSAL/SURVIVOR BENEFIT REFORMS")
    println("=" ^ 60)

    # --- Current law baseline ---
    par_base = create_params(benefit_fn=benefits_current_law)
    ss_base  = solve_steady_state(par_base, benefit_fn=benefits_current_law)

    proj_base = project_economy(ss_base;
        n_years=n_years, start_year=start_year,
        label="Current Law", kwargs...)

    # --- Reform 1: Capped spousal ---
    par_cap = create_params(benefit_fn=benefits_capped_spousal)
    ss_cap  = solve_steady_state(par_cap, benefit_fn=benefits_capped_spousal)

    proj_cap = project_economy(ss_base;
        n_years=n_years, start_year=start_year,
        ss_reform=ss_cap, reform_year=reform_year,
        reform_phase_in=reform_phase_in,
        label="Capped Spousal", kwargs...)

    # --- Reform 2: Earnings sharing ---
    par_es = create_params(benefit_fn=benefits_earnings_sharing)
    ss_es  = solve_steady_state(par_es, benefit_fn=benefits_earnings_sharing)

    proj_es = project_economy(ss_base;
        n_years=n_years, start_year=start_year,
        ss_reform=ss_es, reform_year=reform_year,
        reform_phase_in=reform_phase_in,
        label="Earnings Sharing", kwargs...)

    # --- Reform 3: Caregiver credits ---
    par_cg = create_params(benefit_fn=benefits_caregiver_credits,
                           aime_modifier=caregiver_modifier)
    ss_cg  = solve_steady_state(par_cg,
                                benefit_fn=benefits_caregiver_credits,
                                aime_modifier=caregiver_modifier)

    proj_cg = project_economy(ss_base;
        n_years=n_years, start_year=start_year,
        ss_reform=ss_cg, reform_year=reform_year,
        reform_phase_in=reform_phase_in,
        label="Caregiver Credits", kwargs...)

    # --- Comparison summary ---
    println("\n" * "=" ^ 60)
    println("  REFORM COMPARISON SUMMARY")
    println("=" ^ 60)

    function summary_row(proj)
        w10 = min(10, n_years); w75 = min(75, n_years)
        depl = isnothing(proj.depletion_year) ?
               "solvent" : string(proj.depletion_year)
        cost10 = mean(proj.ss_cost_rate[1:w10])
        cost75 = mean(proj.ss_cost_rate[1:w75])
        outlay75 = sum(proj.ss_outlays_nom[1:w75])
        (label=proj.label, depl=depl, cost10=cost10, cost75=cost75,
         outlay75=outlay75)
    end

    rows = [summary_row(p) for p in [proj_base, proj_cap, proj_es, proj_cg]]

    @printf("\n  %-20s %10s %10s %10s %14s\n",
            "Policy", "TF Depl", "Cost10y%", "Cost75y%", "Outlays75y(\$T)")
    println("  " * repeat("-", 66))
    for r in rows
        @printf("  %-20s %10s %9.2f%% %9.2f%% %13.1f\n",
                r.label, r.depl, r.cost10, r.cost75, r.outlay75/1000)
    end

    base_out = sum(proj_base.ss_outlays_nom[1:min(75,n_years)])
    println("\n  Savings vs. current law (75-yr outlays):")
    for r in rows[2:end]
        diff = r.outlay75 - base_out/1000*1000
        @printf("    %-20s: \$%.1fT (%+.1f%%)\n",
                r.label, diff/1000,
                diff / base_out * 100)
    end

    (baseline=proj_base, capped_spousal=proj_cap,
     earnings_sharing=proj_es, caregiver_credits=proj_cg,
     ss_baseline=ss_base, ss_capped=ss_cap,
     ss_sharing=ss_es, ss_caregiver=ss_cg)
end;