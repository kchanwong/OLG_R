# ============================================================
# PWBM Social Security OLG Model — Julia Port
# Translated from pwbm_model_fixed.R (v9, equil-hours AIME)
#
# Dependencies:
#   using Pkg
#   Pkg.add(["Distributions", "Printf", "Statistics", "LinearAlgebra"])
# ============================================================

using Distributions
using Printf
using Statistics
using LinearAlgebra

# ============================================================
# UTILITY: number formatting helper
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

function age_efficiency(age::Int, J_start::Int=21)
    x = (age - J_start + 1) / 40.0
    0.8*x - 0.4*x^2
end

# ============================================================
# SSA Period Life Table (2022, as used in 2025 Trustees Report)
# Source: https://www.ssa.gov/oact/STATS/table4c6.html
# q(a) = probability of dying within one year at exact age a.
# Gender-neutral: average of male and female columns.
# Index: age 0 = index 1, age 120 = index 121.
# ============================================================
const SSA_Q_MALE = Float64[
    # age 0-20
    0.006064, 0.000491, 0.000309, 0.000248, 0.000199,  # 0-4
    0.000167, 0.000143, 0.000126, 0.000121, 0.000121,  # 5-9
    0.000127, 0.000143, 0.000171, 0.000227, 0.000320,  # 10-14
    0.000451, 0.000622, 0.000826, 0.001026, 0.001182,  # 15-19
    0.001301,                                            # 20
    # age 21-120
    0.001404, 0.001498, 0.001586, 0.001679, 0.001776,  # 21-25
    0.001881, 0.001985, 0.002095, 0.002219, 0.002332,  # 26-30
    0.002445, 0.002562, 0.002653, 0.002716, 0.002791,  # 31-35
    0.002894, 0.002994, 0.003091, 0.003217, 0.003353,  # 36-40
    0.003499, 0.003642, 0.003811, 0.003996, 0.004175,  # 41-45
    0.004388, 0.004666, 0.004973, 0.005305, 0.005666,  # 46-50
    0.006069, 0.006539, 0.007073, 0.007675, 0.008348,  # 51-55
    0.009051, 0.009822, 0.010669, 0.011548, 0.012458,  # 56-60
    0.013403, 0.014450, 0.015571, 0.016737, 0.017897,  # 61-65
    0.019017, 0.020213, 0.021569, 0.023088, 0.024828,  # 66-70
    0.026705, 0.028761, 0.031116, 0.033861, 0.037088,  # 71-75
    0.041126, 0.045241, 0.049793, 0.054768, 0.060660,  # 76-80
    0.067027, 0.073999, 0.081737, 0.090458, 0.100525,  # 81-85
    0.111793, 0.124494, 0.138398, 0.153207, 0.169704,  # 86-90
    0.187963, 0.208395, 0.230808, 0.253914, 0.277402,  # 91-95
    0.300882, 0.324326, 0.347332, 0.369430, 0.391927,  # 96-100
    0.414726, 0.437722, 0.460800, 0.483840, 0.508032,  # 101-105
    0.533434, 0.560105, 0.588111, 0.617516, 0.648392,  # 106-110
    0.680812, 0.714852, 0.750595, 0.788125, 0.827531,  # 111-115
    0.868907, 0.912353, 0.957970, 1.000000, 1.000000,  # 116-120
]

const SSA_Q_FEMALE = Float64[
    # age 0-20
    0.005119, 0.000398, 0.000240, 0.000198, 0.000160,  # 0-4
    0.000134, 0.000118, 0.000109, 0.000106, 0.000106,  # 5-9
    0.000111, 0.000121, 0.000140, 0.000162, 0.000188,  # 10-14
    0.000224, 0.000276, 0.000337, 0.000395, 0.000450,  # 15-19
    0.000496,                                            # 20
    # age 21-120
    0.000532, 0.000567, 0.000610, 0.000650, 0.000699,  # 21-25
    0.000743, 0.000796, 0.000855, 0.000924, 0.000988,  # 26-30
    0.001053, 0.001123, 0.001198, 0.001263, 0.001324,  # 31-35
    0.001403, 0.001493, 0.001596, 0.001700, 0.001803,  # 36-40
    0.001905, 0.002009, 0.002116, 0.002223, 0.002352,  # 41-45
    0.002516, 0.002712, 0.002936, 0.003177, 0.003407,  # 46-50
    0.003642, 0.003917, 0.004238, 0.004619, 0.005040,  # 51-55
    0.005493, 0.005987, 0.006509, 0.007067, 0.007658,  # 56-60
    0.008305, 0.008991, 0.009681, 0.010343, 0.011018,  # 61-65
    0.011743, 0.012532, 0.013512, 0.014684, 0.016025,  # 66-70
    0.017468, 0.019195, 0.021195, 0.023452, 0.025980,  # 71-75
    0.029153, 0.032394, 0.035888, 0.039676, 0.044156,  # 76-80
    0.049087, 0.054635, 0.061066, 0.068431, 0.076841,  # 81-85
    0.086205, 0.096851, 0.109019, 0.121867, 0.135805,  # 86-90
    0.151108, 0.168020, 0.186340, 0.206432, 0.228086,  # 91-95
    0.250406, 0.273699, 0.296984, 0.319502, 0.342716,  # 96-100
    0.366532, 0.390844, 0.415531, 0.440463, 0.466891,  # 101-105
    0.494904, 0.524599, 0.556075, 0.589439, 0.624805,  # 106-110
    0.662294, 0.702031, 0.744153, 0.788125, 0.827531,  # 111-115
    0.868907, 0.912353, 0.957970, 1.000000, 1.000000,  # 116-120
]

# Gender-neutral average death probability, ages 0-120
const SSA_Q_AVG = (SSA_Q_MALE .+ SSA_Q_FEMALE) ./ 2.0

function survival_probs(par::Dict)
    # s(a) = 1 - q(a); s(J_max) = 0 forced (hard terminal age)
    # SSA_Q_AVG is 1-indexed: age a → index a+1
    ages = par[:J_start]:par[:J_max]
    s = Float64[1.0 - SSA_Q_AVG[a + 1] for a in ages]
    s[end] = 0.0
    s
end

production(K, L, alpha) = K^alpha * L^(1-alpha)
mpk(K, L, alpha)        = alpha * (L/K)^(1-alpha)
mpl(K, L, alpha)        = (1-alpha) * (K/L)^alpha

function calibrate_nu2(par::Dict, rho::Float64)
    tp = par[:tau_statutory_corp] * par[:phi_int_corp]
    par[:leverage_ratio_target] * (tp * rho / (1 - tp))^(-1 / (par[:nu1] - 1))
end

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

function compute_PIA(b::Float64, par::Dict)
    # b is in mu-units; bend points are stored as dollar values, convert to mu-units
    b1 = par[:ss_bend1] / par[:mu_dollar]
    b2 = par[:ss_bend2] / par[:mu_dollar]
    b <= b1 && return 0.9*b
    b <= b2 && return 0.9*b1 + 0.32*(b - b1)
    0.9*b1 + 0.32*(b2 - b1) + 0.15*(b - b2)
end

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
    ord  = par[:theta_lab_ORD]*yl_d + par[:theta_corp_ORD]*yc_d +
           par[:theta_pass_ORD]*yp_d + yd_d + ss_tx
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
            (oasi=par[:theta_ss_ORD]*ss_d,
             total=par[:theta_ss_ORD]*ss_d,
             hi=0.0)
    pref = par[:theta_corp_PREF]*yc_d
    ot0  = apply_tax_schedule(non_ss,               par[:ord_brackets], par[:ord_rates])
    ot1  = apply_tax_schedule(non_ss + tiers.oasi,  par[:ord_brackets], par[:ord_rates])
    ot2  = apply_tax_schedule(non_ss + tiers.total, par[:ord_brackets], par[:ord_rates])
    pt   = apply_tax_schedule(pref, par[:pref_brackets], par[:pref_rates])
    pay  = par[:payroll_rate] * min(yl_d, par[:payroll_cap])
    ttax = (ot2 + pt + pay + par[:tau_con]*consumption*mu + par[:tau_lumpsum]) / mu
    (total_tax=ttax,
     ss_tax_rev_oasi=(ot1-ot0)/mu,
     ss_tax_rev_hi=(ot2-ot1)/mu)
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

function utility(c::Float64, n::Float64, gamma::Float64=0.30, sigma::Float64=2.0)
    c <= 0.0 && return -1e10
    u_c = abs(sigma - 1.0) < 1e-8 ? log(c) : (c^(1-sigma) - 1) / (1-sigma)
    l   = max(1.0 - n, 1e-10)
    u_l = gamma > 0.0 ? gamma * log(l) : 0.0
    u_c + u_l
end

function mu_c(c::Float64, n::Float64, gamma::Float64=0.30, sigma::Float64=2.0)
    c <= 0.0 && return 1e10
    max(c, 1e-10)^(-sigma)
end

function inv_mu(rhs::Float64, n::Float64, gamma::Float64=0.30, sigma::Float64=2.0)
    rhs <= 0.0 && return 1e10
    max(rhs, 1e-30)^(-1.0 / sigma)
end

function optn(c::Float64, wz::Float64, mtr::Float64,
              gamma::Float64=0.30, sigma::Float64=2.0)
    wa = wz*(1.0 - mtr)
    wa < 1e-10 && return 0.0
    # FOC: gamma/(1-n) = wa * c^(-sigma)  =>  1-n = gamma/(wa * c^(-sigma))
    denom = wa * max(c, 1e-10)^(-sigma)
    clamp(1.0 - gamma / max(denom, 1e-10), 0.0, 0.99)
end

# Linear interpolation matching R's approx(..., rule=2)
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
# COMPUTE_AIME — equilibrium-hours AIME
# ============================================================

function compute_aime(par::Dict, grids::NamedTuple, prices::Dict,
                      z_perm_val::Float64, pol_n::Array{Float64,3})
    nz       = par[:n_z]
    Jr       = par[:J_retire] - par[:J_start] + 1
    zg       = grids.z_grid
    zp       = z_perm_val
    cap_b    = par[:ss_cap_frac] * par[:avg_wage_mu]
    g_idx    = par[:g_A_index]
    age_60_j = 67 - par[:J_start]   # career years up to and including age 60
    b_by_z   = zeros(nz)

    for iz in 1:nz
        earn_path = zeros(Jr-1)
        for jw in 1:(Jr-1)
            zv    = exp(zg[iz] + zp + grids.age_eff[jw])
            n_avg = mean(pol_n[:, iz, jw])
            raw   = prices[:w] * zv * n_avg
            if jw <= age_60_j
                years_to_60   = age_60_j - jw
                index_factor  = (1.0 + g_idx)^years_to_60
                earn_path[jw] = min(raw * index_factor, cap_b)
            else
                earn_path[jw] = min(raw, cap_b)
            end
        end
        b_by_z[iz] = mean(sort(earn_path, rev=true)[1:min(35, Jr-1)])
    end
    b_by_z
end

# ============================================================
# EGM CORE
# ============================================================

function run_egm(par::Dict, grids::NamedTuple, prices::Dict,
                 z_perm_val::Float64, b_by_z::Vector{Float64})
    na = par[:n_a]; nz = par[:n_z]; nj = par[:n_ages]
    Jr = par[:J_retire] - par[:J_start] + 1
    ag = grids.a_grid; zg = grids.z_grid
    Pz = grids.Pi_z;   sv = grids.surv
    bt = par[:beta]; gm = par[:gamma]; sg = par[:sigma]
    w  = prices[:w]; rp = prices[:pi_corp]
    beq = prices[:beq]; zp = z_perm_val

    V   = zeros(na, nz, nj)
    pa  = zeros(na, nz, nj)
    pn  = zeros(na, nz, nj)
    pcc = zeros(na, nz, nj)

    # --- Terminal condition (age nj) ---
    for iz in 1:nz, ia in 1:na
        ss   = compute_PIA(b_by_z[iz], par)
        cash = ag[ia]*(1+rp) + ss + beq
        pcc[ia,iz,nj] = max(1e-10, cash*0.95)
        V[ia,iz,nj]   = utility(pcc[ia,iz,nj], 0.0, gm, sg)
    end

    # --- Backward induction ---
    for j in (nj-1):-1:1
        ret = j >= Jr
        sn  = sv[j+1]
        ssb_vec = ret ? Float64[compute_PIA(b_by_z[iz], par) for iz in 1:nz] :
                        zeros(nz)

        for iz in 1:nz
            zv  = exp(zg[iz] + zp + grids.age_eff[j])
            wz  = w * zv
            ssb = ssb_vec[iz]

            # Expected marginal utility of consumption tomorrow
            Emu = zeros(na)
            for ia2 in 1:na, iz2 in 1:nz
                Emu[ia2] += Pz[iz,iz2] * mu_c(pcc[ia2,iz2,j+1],
                                               pn[ia2,iz2,j+1], gm, sg)
            end

            # EGM: solve for (c,n,a_today) on endogenous grid
            ce = zeros(na); ne = zeros(na); ae = zeros(na)
            for ia2 in 1:na
                rhs = bt*sn*(1+rp)*Emu[ia2]
                if ret
                    ce[ia2] = inv_mu(rhs, 0.0, gm, sg)
                    ne[ia2] = 0.0
                else
                    cg = inv_mu(rhs, 0.3, gm, sg); ng = 0.3
                    for k in 1:6
                        mk = mtr_lab(wz*ng*par[:mu_dollar],
                                     ssb*par[:mu_dollar], par)
                        ng = optn(cg, wz, mk, gm, sg)
                        cg = inv_mu(rhs, ng, gm, sg)
                    end
                    ce[ia2] = cg; ne[ia2] = ng
                end
                tx      = tau_HH(wz*ne[ia2], 0.0, 0.0, 0.0, ssb, ce[ia2], par)
                ae[ia2] = (ag[ia2] + ce[ia2] + tx - wz*ne[ia2] - ssb - beq) / (1+rp)
            end

            vld = [isfinite(ae[i]) && isfinite(ce[i]) && !isnan(ae[i]) && !isnan(ce[i])
                   for i in 1:na]

            # Fallback if not enough valid points
            if sum(vld) < 2
                for ia in 1:na
                    yfb  = ret ? 0.0 : wz*0.3
                    cash = ag[ia]*(1+rp) + yfb + ssb + beq
                    tfb  = tau_HH(yfb, 0.0, 0.0, 0.0, ssb, max(1e-10, cash), par)
                    pcc[ia,iz,j] = max(1e-10, (cash-tfb)*0.5)
                    pa[ia,iz,j]  = max(0.0,   cash-tfb-pcc[ia,iz,j])
                    pn[ia,iz,j]  = ret ? 0.0 : 0.3
                    V[ia,iz,j]   = utility(pcc[ia,iz,j], pn[ia,iz,j], gm, sg)
                end
                continue
            end

            # Sort by endogenous asset grid, remove duplicates
            av = ae[vld]; cv = ce[vld]; nv = ne[vld]
            ord = sortperm(av)
            av = av[ord]; cv = cv[ord]; nv = nv[ord]
            uniq = vcat(true, [av[i] != av[i-1] for i in 2:length(av)])
            av = av[uniq]; cv = cv[uniq]; nv = nv[uniq]
            amn = av[1]

            # Interpolate onto exogenous asset grid
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
                            mc   = mtr_lab(wz*nc*par[:mu_dollar],
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
# HOUSEHOLD — two-pass EGM with equilibrium-hours AIME
# ============================================================

function solve_household(par::Dict, grids::NamedTuple, prices::Dict,
                         z_perm_val::Float64=0.0)
    zp    = z_perm_val
    nz    = par[:n_z]
    Jr    = par[:J_retire] - par[:J_start] + 1
    zg    = grids.z_grid
    cap_b = par[:ss_cap_frac] * par[:avg_wage_mu]

    # --- Pass 1: AIME with fixed n=0.35 ---
    b_by_z_p1 = zeros(nz)
    for iz in 1:nz
        ep = zeros(Jr-1)
        for jw in 1:(Jr-1)
            zv      = exp(zg[iz] + zp + grids.age_eff[jw])
            ep[jw]  = min(prices[:w]*zv*0.35, cap_b)
        end
        b_by_z_p1[iz] = mean(sort(ep, rev=true)[1:min(35, Jr-1)])
    end
    @printf("   [zp=%.2f] pass 1 (n=0.35 AIME)...\n", zp)
    res1 = run_egm(par, grids, prices, zp, b_by_z_p1)

    # --- Pass 2: AIME with equilibrium hours from pass 1 ---
    b_by_z_p2 = compute_aime(par, grids, prices, zp, res1.pol_n)
    aime_chg   = mean(abs.(b_by_z_p2 .- b_by_z_p1)) /
                 max(mean(b_by_z_p1), 1e-10) * 100
    @printf("   [zp=%.2f] pass 2 (equil-hours AIME, mean chg=%.1f%%)...\n",
            zp, aime_chg)
    res2 = run_egm(par, grids, prices, zp, b_by_z_p2)

    (V=res2.V, pol_a=res2.pol_a, pol_n=res2.pol_n, pol_c=res2.pol_c,
     b_by_z=b_by_z_p2, b_by_z_pass1=b_by_z_p1, aime_pct_change=aime_chg)
end

# ============================================================
# DISTRIBUTION, AGGREGATES, PAYROLL
# ============================================================

function compute_distribution(par::Dict, grids::NamedTuple,
                              pol_a::Array{Float64,3})
    na = par[:n_a]; nz = par[:n_z]; nj = par[:n_ages]
    ag = grids.a_grid; Pz = grids.Pi_z
    x  = zeros(na, nz, nj)

    for iz in 1:nz
        x[1,iz,1] = (1.0/nz) / (1.0 + par[:g_pop])^(nj-1)
    end

    for j in 1:(nj-1)
        sn = grids.surv[j+1]
        for iz in 1:nz, ia in 1:na
            m = x[ia,iz,j]
            m < 1e-15 && continue
            mass = m*sn / (1.0 + par[:g_pop])
            an   = max(ag[1], min(ag[na], pol_a[ia,iz,j]))
            lo   = max(1, min(na-1, searchsortedlast(ag, an)))
            hi   = lo + 1
            wh   = (ag[hi] - ag[lo]) > 1e-12 ?
                   (an - ag[lo]) / (ag[hi] - ag[lo]) : 0.0
            for iz2 in 1:nz
                pr = Pz[iz,iz2]
                pr < 1e-15 && continue
                cb = mass*pr
                x[lo,iz2,j+1] += cb*(1-wh)
                x[hi,iz2,j+1] += cb*wh
            end
        end
    end
    tot = sum(x)
    tot > 0 && (x ./= tot)
    x
end

function compute_aggregates(par::Dict, grids::NamedTuple,
                            hh_list::Vector, x_list::Vector,
                            prices::Dict)
    n_perm = length(par[:z_perm_vals])
    nj = par[:n_ages]; nz = par[:n_z]
    Jr = par[:J_retire] - par[:J_start] + 1
    ag = grids.a_grid; zg = grids.z_grid
    A=0.0; L=0.0; C=0.0; tr=0.0; ssb=0.0; ss_oasi=0.0; ss_hi=0.0

    for ip in 1:n_perm
        zp = par[:z_perm_vals][ip]; wp = par[:p_perm_vals][ip]
        hh = hh_list[ip]; x = x_list[ip]; bz = hh.b_by_z
        for j in 1:nj
            ret = j >= Jr
            for iz in 1:nz
                zv  = exp(zg[iz] + zp + grids.age_eff[j])
                mv  = x[:,iz,j]
                idx = findall(v -> v > 1e-15, mv)
                isempty(idx) && continue
                mi  = mv[idx] .* wp
                ai  = ag[idx]
                ni  = hh.pol_n[idx,iz,j]
                ci  = hh.pol_c[idx,iz,j]
                A  += sum(ai .* mi)
                L  += sum(zv .* ni .* mi)
                C  += sum(ci .* mi)
                yl  = prices[:w]*zv .* ni
                yc  = prices[:pi_corp]*par[:zeta_income_corp]*prices[:psi_cap] .* ai
                yp  = prices[:pi_pass]*par[:zeta_income_pass]*prices[:psi_cap] .* ai
                yd  = prices[:r_G]*prices[:psi_debt] .* ai
                sb  = ret ? compute_PIA(bz[iz], par) : 0.0
                for ii in eachindex(idx)
                    d      = tau_HH_decomposed(yl[ii], yc[ii], yp[ii], yd[ii],
                                               sb, ci[ii], par)
                    tr      += d.total_tax        * mi[ii]
                    ss_oasi += d.ss_tax_rev_oasi  * mi[ii]
                    ss_hi   += d.ss_tax_rev_hi    * mi[ii]
                end
                ssb += sb * sum(mi)
            end
        end
    end
    (A=A, L=L, C=C, tax_rev_HH=tr, SS_BEN=ssb,
     ss_benefit_tax_rev_oasi=ss_oasi,
     ss_benefit_tax_rev_hi=ss_hi,
     ss_benefit_tax_rev=ss_oasi+ss_hi)
end

function payroll_stats(par::Dict, grids::NamedTuple,
                       hh_list::Vector, x_list::Vector,
                       prices::Dict)
    n_perm = length(par[:z_perm_vals])
    nz = par[:n_z]; nj = par[:n_ages]
    Jr  = par[:J_retire] - par[:J_start] + 1
    zg  = grids.z_grid
    cap = par[:ss_cap_frac] * par[:avg_wage_mu]
    tot=0.0; tax=0.0; nw=0.0

    for ip in 1:n_perm
        zp = par[:z_perm_vals][ip]; wp = par[:p_perm_vals][ip]
        hh = hh_list[ip]; x = x_list[ip]
        for j in 1:nj
            j >= Jr && continue
            for iz in 1:nz
                zv  = exp(zg[iz] + zp + grids.age_eff[j])
                mv  = x[:,iz,j]
                idx = findall(v -> v > 1e-15, mv)
                isempty(idx) && continue
                mi  = mv[idx] .* wp
                yl  = prices[:w]*zv .* hh.pol_n[idx,iz,j]
                tot += sum(yl .* mi)
                tax += sum(min.(yl, cap) .* mi)
                nw  += sum(mi)
            end
        end
    end
    (total=tot, taxable=tax, n_workers=nw,
     avg_earn=tot/max(nw,1e-10),
     pct_above=1.0-tax/max(tot,1e-10))
end

# ============================================================
# PARAMETERS
# ============================================================

function create_params()
    Dict{Symbol,Any}(
        :J_start  => 21,  :J_retire => 67, :J_max => 100, :n_ages => 80,
        :g_pop    => 0.000,
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
        :z_perm_vals => [-0.45, 0.10, 0.85, 1.20],
        :p_perm_vals => [ 0.40, 0.45, 0.12, 0.03],
        :sigma_trans  => 0.063, :sigma_pers => 0.007, :rho_pers => 0.990,
        :ord_brackets  => Float64[0,9700,39475,84200,160725,204100,510300],
        :ord_rates     => Float64[0.10,0.12,0.22,0.24,0.32,0.35,0.37],
        :pref_brackets => Float64[0,39375,434550],
        :pref_rates    => Float64[0.0,0.15,0.20],
        :payroll_rate  => 0.106, :payroll_cap => 176100.0,
        :tau_con => 0.0, :tau_lumpsum => 0.0,
        :theta_corp_ORD  => 0.60, :theta_corp_PREF => 0.40,
        :theta_lab_ORD   => 0.95, :theta_lab_PT    => 1.00,
        :theta_pass_ORD  => 1.00, :theta_ss_ORD    => 0.85,
        :ss_tax_thresh1  => 25000.0, :ss_tax_thresh2 => 34000.0,
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
        :g_A_index       => 0.012
    )
end

function setup_grids(par::Dict)
    pers    = tauchen(par[:n_z], par[:rho_pers], sqrt(par[:sigma_pers]))
    a_grid  = collect(range(par[:a_min], par[:a_max], length=par[:n_a]))
    age_eff = Float64[age_efficiency(a, par[:J_start])
                      for a in par[:J_start]:par[:J_max]]
    Jr = par[:J_retire] - par[:J_start] + 1
    age_eff[Jr:par[:n_ages]] .= 0.0
    (a_grid=a_grid, z_grid=pers.z_grid, Pi_z=pers.Pi,
     age_eff=age_eff, surv=survival_probs(par))
end

# ============================================================
# STEADY STATE SOLVER
# ============================================================

function solve_steady_state(par::Dict)
    println("=== Solving Steady State (v9 — equil-hours AIME) ===")
    @printf("  z=[%s]  p=[%s]  g_pop=%.3f  target_earn=\$%s\n",
            join(par[:z_perm_vals], "/"),
            join(par[:p_perm_vals], "/"),
            par[:g_pop],
            format_comma(par[:target_avg_earn]))

    grids  = setup_grids(par)
    n_perm = length(par[:z_perm_vals])
    KL     = 3.0; pK = 1.0

    # Mutable working variables
    hh_list = Vector{Any}(undef, n_perm)
    x_list  = Vector{Any}(undef, n_perm)
    agg     = nothing
    prices  = Dict{Symbol,Any}()
    w=0.0; rK=0.0; Kn=0.0; La=0.0; Aa=0.0; psc=0.95
    dep=0.0; wm=0.0; rm=0.0; ps=nothing

    for it in 1:par[:max_iter_equil]
        @printf(" Iter %d: K/L=%.4f mu\$=%.0f avg_w_mu=%.4f\n",
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

        for ip in 1:n_perm
            @printf("  [type %d/%d z=%.2f p=%.2f]\n",
                    ip, n_perm, par[:z_perm_vals][ip], par[:p_perm_vals][ip])
            hh_list[ip] = solve_household(par, grids, prices, par[:z_perm_vals][ip])
            x_list[ip]  = compute_distribution(par, grids, hh_list[ip].pol_a)
        end

        agg = compute_aggregates(par, grids, hh_list, x_list, prices)
        La  = agg.L; Aa = agg.A
        Kn  = max(0.01, psc*Aa/pK)
        KLn = La > 0 ? Kn/La : KL

        ps = payroll_stats(par, grids, hh_list, x_list, prices)
        par[:mu_dollar]   = par[:target_avg_earn] / max(ps.avg_earn, 1e-10)
        par[:avg_wage_mu] = ps.avg_earn

        Jr = par[:J_retire] - par[:J_start] + 1
        wm = 0.0; rm = 0.0
        for ip in 1:n_perm
            wp = par[:p_perm_vals][ip]; x = x_list[ip]
            wm += sum(x[:,:,1:(Jr-1)]) * wp
            rm += sum(x[:,:,Jr:par[:n_ages]]) * wp
        end
        dep     = rm / wm
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
                    dep_ratio=dep, worker_mass=wm, retiree_mass=rm)
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
     dep_ratio=dep, worker_mass=wm, retiree_mass=rm)
end

# ============================================================
# EXTRACT STEADY-STATE RATIOS
# ============================================================

function compute_taxable_payroll(ss)
    ps = payroll_stats(ss.par, ss.grids, ss.hh_list, ss.x_list, ss.prices)
    (taxable=ps.taxable, total=ps.total, n_workers=ps.n_workers,
     avg_earn=ps.avg_earn, pct_above=ps.pct_above)
end

function extract_ss_ratios(ss)
    par    = ss.par; mu = par[:mu_dollar]
    tp     = compute_taxable_payroll(ss)
    n_perm = length(par[:z_perm_vals])
    Jr     = par[:J_retire] - par[:J_start] + 1

    wm = 0.0; rm = 0.0
    for ip in 1:n_perm
        wp = par[:p_perm_vals][ip]; x = ss.x_list[ip]
        wm += sum(x[:,:,1:(Jr-1)]) * wp
        rm += sum(x[:,:,Jr:par[:n_ages]]) * wp
    end
    dep_ratio = rm / wm

    fica_rev      = par[:payroll_rate] * tp.taxable
    ss_ben        = ss.agg.SS_BEN
    total_tax_rev = ss.agg.tax_rev_HH
    ss_btax_rev   = ss.agg.ss_benefit_tax_rev
    ss_btax_oasi  = ss.agg.ss_benefit_tax_rev_oasi
    ss_btax_hi    = ss.agg.ss_benefit_tax_rev_hi

    Y = ss.Y; K = ss.K; L = ss.L; w = ss.w; rK = ss.r_K
    pic = rK - par[:delta]
    corp_profit = par[:zeta_income_corp]*Y - w*par[:zeta_income_corp]*L -
                  par[:delta]*K*par[:zeta_income_corp]
    corp_tax = par[:tau_statutory_corp] *
               max(0.0, corp_profit) *
               (1 - par[:phi_exp_corp]*0.3 - par[:zeta_ded_corp])

    (Y=Y, K=K, L=L, w=w, C=ss.agg.C, A=ss.agg.A,
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
     avg_earnings=tp.avg_earn,
     taxable_payroll=tp.taxable,
     total_payroll=tp.total,
     pct_above_cap=tp.pct_above,
     mu_dollar=mu,
     ss_ben_per_retiree=ss_ben/max(rm,1e-10),
     fica_per_worker=fica_rev/max(wm,1e-10),
     earn_per_worker=tp.total/max(wm,1e-10),
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
# PROJECT ECONOMY — cohort-turnover benefit model + AWI cap
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
    economy_scale::Float64    = 28.0,
    ss_reform             = nothing,
    reform_year           = nothing,
    reform_phase_in::Int  = 10,
    label::String         = "Current Law",
    scenario_periods      = nothing,
    thresh_sens::Float64  = 0.3
)
    @printf("=== Projecting %d years: %d-%d [%s] ===\n",
            n_years, start_year, start_year+n_years-1, label)

    base = extract_ss_ratios(ss)
    mu   = base.mu_dollar; par = ss.par

    ssa_covered_workers = 180e6
    N_hh = ssa_covered_workers / base.worker_mass
    @printf("  N_hh: %.1fM (anchored to %sM covered workers)\n",
            N_hh/1e6, format_comma(round(ssa_covered_workers/1e6)))
    @printf("  Implied GDP: \$%.1fT  (economy_scale arg ignored)\n",
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
            haskey(ep,:g_A)        && (g_A_vec[idx]   .= ep[:g_A])
            haskey(ep,:inflation)  && (inf_vec[idx]   .= ep[:inflation])
            haskey(ep,:g_pop)      && (g_pop_vec[idx] .= ep[:g_pop])
            @printf("  Scenario episode %d-%d: g_A=%.3f inf=%.3f\n",
                    ep[:year_start], ep[:year_end],
                    get(ep,:g_A,g_A), get(ep,:inflation,inflation))
        end
    end

    # Cumulative indices — t=1 normalised to 1, growth begins year 2
    cum_A     = cumprod([1.0; (1.0 .+ g_A_vec[2:end])])
    cum_pop   = cumprod([1.0; (1.0 .+ g_pop_vec[2:end])])
    cum_price = cumprod([1.0; (1.0 .+ inf_vec[2:end])])
    cum_wage  = cumprod([1.0; (1.0 .+ g_A_vec[2:end] .+ inf_vec[2:end])])

    # [FIX 4] Benefits are COLA-adjusted with a ~1-year lag
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

    # [FIX 3] Pre-compute dimensionless benefit-tax rates
    btax_rate_oasi_base = base.ss_benefit_tax_oasi_per_retiree /
                          max(base.ss_ben_per_retiree, 1e-12)
    btax_rate_hi_base   = base.ss_benefit_tax_hi_per_retiree /
                          max(base.ss_ben_per_retiree, 1e-12)
    if has_reform
        btax_rate_oasi_ref = ref.ss_benefit_tax_oasi_per_retiree /
                             max(ref.ss_ben_per_retiree, 1e-12)
        btax_rate_hi_ref   = ref.ss_benefit_tax_hi_per_retiree /
                             max(ref.ss_ben_per_retiree, 1e-12)
    end

    # Output storage
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
            fica_rate_t    = bl(base.payroll_rate,    ref.payroll_rate)
            pct_above_t    = bl(base.pct_above_cap,   ref.pct_above_cap)
            inc_tax_t      = bl(base.income_tax,      ref.income_tax)
            corp_tax_t     = bl(base.corp_tax,        ref.corp_tax)
            Y_t            = bl(base.Y,               ref.Y)
            earn_pw_t      = bl(base.earn_per_worker, ref.earn_per_worker)
            # [FIX 3] Blend dimensionless rates, not dollar amounts
            btax_oasi_rate_t = bl(btax_rate_oasi_base, btax_rate_oasi_ref)
            btax_hi_rate_t   = bl(btax_rate_hi_base,   btax_rate_hi_ref)
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
        # [FIX 4] Nominalize with lagged price level (1-year COLA lag)
        avg_ben_nom    = avg_ben_real * mu * cum_price_lag[t]
        n_ret_t        = base.retiree_mass * dep_scale * N_hh_t
        ss_outlays_nom_t = avg_ben_nom * n_ret_t

        # [FIX 2] Bracket creep: logistic saturation of SS benefit taxation
        base_frac_oasi   = max(btax_oasi_rate_t / 0.85, 1e-6)
        creep_oasi       = 1.0 / (1 + (1/base_frac_oasi - 1) *
                                  exp(-thresh_sens * log(cum_price[t])))
        btax_oasi_adj    = (creep_oasi / base_frac_oasi) * btax_oasi_rate_t

        base_frac_hi     = max(btax_hi_rate_t / 0.85, 1e-6)
        creep_hi         = 1.0 / (1 + (1/base_frac_hi - 1) *
                                  exp(-0.6 * thresh_sens * log(cum_price[t])))
        btax_hi_adj      = (creep_hi / base_frac_hi) * btax_hi_rate_t

        # [FIX 1] Benefit taxation anchored to avg_ben_nom (not cum_A * mu * price)
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
    println("\n--- Actuarial Summary ---")
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
    println("\n--- Year-by-Year Projection (selected years) ---")
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

    # Return all results as a NamedTuple
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
end;