#!/usr/bin/env python3
"""Build the Simulink-led, equation-rich IEEE manuscript package."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_ieee_paper_template as base  # noqa: E402

ROOT = SCRIPT_DIR.parents[1]
OUT = ROOT / "ML" / "reports" / "ieee_paper_simulink_enhanced_2026-08-04"
FIG = OUT / "figures"

base.OUT = OUT
base.FIG_DIR = FIG
base.TITLE = (
    "Simulink Modeling, Fault Injection, and Physics-Guided Diagnosis of a "
    "Bidirectional DC-DC Battery Converter"
)
base.AFFILIATION = (
    "The Chinese University of Hong Kong, Shenzhen, Shenzhen 518172, China"
)
base.AUTHOR_NOTE = (
    "Qiang Wang, Xiaojing Xiong, and Bianjia Wang contributed equally and share "
    "second authorship. Zhongxin Cheng and Ying Jia are the corresponding authors. "
    "[VERIFY: insert the corresponding-author e-mail addresses before submission.]"
)
base.DRAFT_NOTICE = (
    "COMPLETE EDITABLE SAMPLE MANUSCRIPT FOR AUTHOR REVISION. The evidence is from "
    "Simulink and computer analysis only. No hardware, HIL, laboratory, or genuinely "
    "external dataset has been used; the machine-learning results are therefore a "
    "secondary within-simulation assessment, not a deployment claim."
)
base.ABSTRACT = (
    "This paper develops an equation-consistent MATLAB/Simulink framework for modeling, "
    "fault injection, and diagnosis of a non-isolated bidirectional DC-DC converter in a "
    "battery energy-storage interface. The verified switched model contains a 200-V, "
    "9.8-Ah lithium-ion battery, a 15-mH inductor, a 2.2-mF DC-link capacitor, a "
    "synchronous IGBT half bridge, cascaded discrete PI control, a supervisory state "
    "machine, fault injection, and energy protection. Switching equations, an averaged "
    "state-space model, the implemented forward-Euler PI and back-calculation anti-windup "
    "laws, gate dead time, sensor sampling, and four switch-fault mechanisms are written "
    "explicitly. With a 1-us electrical step and 20-kHz PWM, current-reference tests "
    "tracked +20 A in charge and -20 A in discharge. A 600-W load step changed mean "
    "DC-bus voltage by -3.23 V and -2.89 V, respectively, while steady current standard "
    "deviation remained 0.022-0.032 A. System-level current was nearly invariant to a "
    "50-mOhm on-resistance fault because the current loop compensated through duty ratio. "
    "Direct 1-us switch voltage-current sensing instead yielded the physics estimator "
    "median(|v_sw/i_sw|), separating nominal 1-mOhm and unseen 50-mOhm conditions in 16 "
    "independent Simulink runs at a frozen 10.5-mOhm threshold. A measurement-chain "
    "projection further identified 12 mOhm as the supported stress-test boundary; faults "
    "at or below 8 mOhm were not guaranteed. As a secondary study, 416 simulation runs "
    "over 28 grouped operating points were compared using seven machine-learning models. "
    "Logistic regression, random forest, and extremely randomized trees each reached a "
    "grouped out-of-fold macro-F1 of 1.000, but these scores are explicitly limited to "
    "the simulated domain. Rigorous converter equations and measurement placement provide "
    "stronger evidence than classifier complexity when external validation is unavailable."
)
base.KEYWORDS = (
    "bidirectional DC-DC converter; battery energy storage; Simulink; switched model; "
    "fault injection; PI control; observability; high-resistance fault; machine learning"
)

P, H, B, EQ, T, F = base.P, base.H, base.B, base.EQ, base.T, base.F

base.ELEMENTS = [
    H(1, "I. INTRODUCTION"),
    P(
        "Bidirectional DC-DC converters regulate energy exchange between a battery and a "
        "DC bus. The same power stage must remain stable in charge, discharge, transition, "
        "and protection states, so topology, control, sensing, and diagnosis cannot be "
        "treated independently. State-space averaging provides the established analytical "
        "basis for switching converters {cite:middlebrook1976}. Modern bidirectional "
        "topologies improve conversion efficiency and ripple {cite:lee2019}, while surveys "
        "and microgrid studies emphasize that operating mode and control law determine the "
        "observable electrical response {cite:kondrath2017,sofla2011,xia2018}."
    ),
    P(
        "Fault-tolerant topologies and observer-based algorithms have been proposed for "
        "open-switch faults {cite:mahdavi2023,ding2024}; residual tests and unknown-input "
        "observers have also been used for converter sensor faults "
        "{cite:alsheikh2015,ouahabi2025}. These works motivate a physics-first study. A "
        "classifier cannot recover a fault that the selected measurements do not expose, "
        "and high within-simulation accuracy does not substitute for independent hardware "
        "or external-domain validation. Battery-diagnosis research reaches the same broader "
        "conclusion: model-based and data-driven evidence should be combined, with explicit "
        "attention to fusion, uncertainty, and transferability {cite:jin2024,liu2025,xu2025}."
    ),
    P("The contributions of this study are as follows."),
    B([
        "A complete bidirectional converter implementation is documented from the final Simulink model, including the power stage, supervisory Stateflow logic, two discrete PI loops, 20-kHz PWM, fault injection, direct switch sensing, and energy protection.",
        "The switching and averaged plant equations, exact forward-Euler and back-calculation controller recursions, mode-transition guards, and PWM/dead-time logic are derived using parameters stored in the model.",
        "Sensor bias, full open, partial open, intermittent open, and high on-resistance faults are expressed as explicit mathematical operators. A sensitivity argument explains why current control masks high-resistance degradation in system-level signals.",
        "Healthy start-up, bidirectional current tracking, a 600-W load step, switch-level voltage-current evidence, and independent unseen-resistance runs are reported as the primary simulation results.",
        "Seven classifiers are retained only as a secondary grouped out-of-fold assessment. The lack of external and hardware validation is made a formal claim boundary rather than hidden behind high point estimates.",
    ]),

    H(1, "II. CONVERTER TOPOLOGY AND SIMULINK IMPLEMENTATION"),
    H(2, "A. Power Stage, Variables, and Sign Convention"),
    P(
        "Fig. 1 summarizes the implementation represented by "
        "main_model_fd_v06_switchobservability.slx. The non-isolated synchronous half "
        "bridge connects the DC bus to a battery through the main inductor. S1 is the "
        "upper switch and S2 is the lower switch. The analytical current iL is positive "
        "from the switching node toward the battery; positive iL therefore denotes charge "
        "and negative iL denotes discharge. The logged battery-current channel has the "
        "opposite polarity. The commanded upper-switch state q is one when S1 conducts "
        "and zero when S2 conducts, apart from dead time."
    ),
    F(
        "word/simulink_implementation_architecture.jpg",
        "Fig. 1. Simulink implementation architecture of the bidirectional converter, control, fault injection, sensing, and protection modules.",
        "Redrawn from the verified .slx block inventory. Stored names include Mode_Manager, feedforward_duty, PWM controller, Fault_Injection_Switches, Fault_Diag_Manager, and Energy_Protection_Manager.",
    ),
    P(
        "Let eoc(z) be the battery open-circuit source at state of charge z, rSigma the "
        "sum of inductor, connection, external battery, and internal battery series "
        "resistances, R1 and R2 the on-resistances of S1 and S2, Cdc the DC-link "
        "capacitance, RH the parallel bus load, isrc the bus-source current, and ip the "
        "stepped controlled-current load. Kirchhoff's laws give"
    ),
    EQ(
        "L diL/dt = q vdc - eoc(z) - [rSigma + q R1 + (1-q) R2] iL;   Cdc dvdc/dt = isrc - vdc/RH - ip - q iL.",
        r"\begin{aligned}L\dot i_L&=qv_{dc}-e_{oc}(z)-[r_{\Sigma}+qR_1+(1-q)R_2]i_L,\\ C_{dc}\dot v_{dc}&=i_{src}-\frac{v_{dc}}{R_H}-i_p-qi_L.\end{aligned}",
        "(1)",
    ),
    P(
        "Equation (1) is valid in both directions because the sign of iL changes. Averaging "
        "q over a PWM period gives d=E[q] and req(d)=rSigma+dR1+(1-d)R2. With "
        "x=[iL,vdc]^T and u=[eoc,isrc,ip]^T, the averaged model is"
    ),
    EQ(
        "x_dot = A(d)x + Bu,  A(d)=[[-req/L, d/L],[-d/Cdc,-1/(RH Cdc)]],  B=[[-1/L,0,0],[0,1/Cdc,-1/Cdc]].",
        r"\dot{\mathbf{x}}=\underbrace{\begin{bmatrix}-r_{eq}(d)/L&d/L\\-d/C_{dc}&-1/(R_HC_{dc})\end{bmatrix}}_{\mathbf A(d)}\mathbf{x}+\underbrace{\begin{bmatrix}-1/L&0&0\\0&1/C_{dc}&-1/C_{dc}\end{bmatrix}}_{\mathbf B}\begin{bmatrix}e_{oc}\\i_{src}\\i_p\end{bmatrix}.",
        "(2)",
    ),
    P(
        "With charge-positive current and Qn in coulombs, the battery terminal and SOC "
        "relations may be written"
    ),
    EQ(
        "vbat = eoc(z) + Rb iL;   dz/dt = eta_c iL/Qn for iL>=0, and iL/(eta_d Qn) for iL<0.",
        r"v_{bat}=e_{oc}(z)+R_bi_L,\qquad \dot z=\begin{cases}\eta_c i_L/Q_n,&i_L\ge0,\\i_L/(\eta_dQ_n),&i_L<0.\end{cases}",
        "(3)",
    ),
    P(
        "The Specialized Power Systems Battery block uses a 200-V, 9.8-Ah lithium-ion "
        "preset with 0.20408 ohm internal resistance. An external 0.5-ohm nominal battery "
        "resistance is randomized by experiment. Equation (3) is the paper-level state "
        "description; the switched simulation retains the nonlinear battery block."
    ),
    T(
        "TABLE I. PRINCIPAL POWER-STAGE AND NUMERICAL PARAMETERS",
        ["Item", "Stored/used value", "Role"],
        [
            ["Battery", "200 V, 9.8 Ah, Li-ion", "Nonlinear source and SOC"],
            ["Battery internal resistance", "0.20408 ohm", "Battery preset"],
            ["External battery resistance", "0.5 ohm nominal, +/-10%", "Domain randomization"],
            ["Main inductor", "15 mH, 1 ohm", "Bidirectional transfer"],
            ["DC-link capacitor", "2.2 mF, ESR 1 mOhm", "Bus energy storage"],
            ["Parallel bus load", "200 ohm", "Baseline load"],
            ["Controlled load step", "600 W / 400 V = 1.5 A at 0.35 s", "Dynamic test"],
            ["Switch nominal Ron", "1 mOhm", "Healthy S1/S2"],
            ["PWM / powergui", "20 kHz / discrete 1 us", "Switching / electrical step"],
            ["General logged table", "50 us", "Analysis export"],
            ["Device V-I logging", "1 us before bin aggregation", "High-R observability"],
        ],
        "The 50-us analysis interval is distinct from the 1-us electrical solver step.",
    ),
    H(2, "B. Stored Energy and Power Consistency"),
    P(
        "Energy_Protection_Manager evaluates instantaneous power and stored energy. The "
        "electrical storage estimate and discrete derivative are"
    ),
    EQ(
        "E[k] = 0.5 L iL[k]^2 + 0.5 Cdc vdc[k]^2;   pstored[k] = LPF{(E[k]-E[k-1])/TEP}.",
        r"E[k]=\tfrac12Li_L^2[k]+\tfrac12C_{dc}v_{dc}^2[k],\qquad p_{stored}[k]=\mathcal L_{\alpha}\!\left\{\frac{E[k]-E[k-1]}{T_{EP}}\right\}.",
        "(4)",
    ),
    P(
        "For any p[k], the implemented first-order filter is Lalpha{p}[k] = alpha "
        "Lalpha{p}[k-1]+(1-alpha)p[k], with alpha=exp(-50 us/1 ms). The monitored "
        "power-balance residual is"
    ),
    EQ(
        "rP[k] = LPF{vdc isrc} + LPF{vbat ibat} - vdc iload - pstored[k].",
        r"r_P[k]=\mathcal L_{\alpha}\{v_{dc}i_{src}\}+\mathcal L_{\alpha}\{v_{bat}i_{bat}\}-v_{dc}i_{load}-p_{stored}[k].",
        "(5)",
    ),
    P(
        "This residual is a consistency and protection signal, not proof of a lossless "
        "model. Semiconductor loss, battery polarization, and filter delay contribute to "
        "its nonzero baseline."
    ),

    H(1, "III. SUPERVISORY AND DISCRETE CONTROL EQUATIONS"),
    H(2, "A. Hysteretic Mode Request and Safe Transition"),
    P(
        "The mod_cmd Stateflow chart uses persistence. Define DT[c]=1 when condition c "
        "has remained true continuously for at least T. The automatic request mu is"
    ),
    EQ(
        "mu: 0->1 if D10ms[vdc>=402]; 1->0 if D20ms[vdc<=399.5]; 0->2 if D10ms[vdc<=398]; 2->0 if D20ms[vdc>=400.5].",
        r"\begin{aligned}\mu:0\!\to\!1&\iff \mathcal D_{10\,ms}[v_{dc}\ge402],&1\!\to\!0&\iff \mathcal D_{20\,ms}[v_{dc}\le399.5],\\0\!\to\!2&\iff \mathcal D_{10\,ms}[v_{dc}\le398],&2\!\to\!0&\iff \mathcal D_{20\,ms}[v_{dc}\ge400.5].\end{aligned}",
        "(6)",
    ),
    P(
        "Here mu=1 requests charge and mu=2 requests discharge. Mode_Manager contains "
        "Standby, Precharge, Charge, Discharge, Transition, and Fault_lockout states. "
        "Precharge lasts 5 ms. Reversal is released only after |iL,avg|<0.5 A has held "
        "for 5 ms. Any trip forces Fault_lockout, disables both gates, sets iref to zero, "
        "and resets the PI integrators."
    ),
    H(2, "B. Current Reference, Slew Limit, and Outer PI"),
    P(
        "The stored voltage-error sign is ev[k]=vdc[k]-400 V, so a high bus requests "
        "positive charge and a low bus requests negative discharge. Let rho be "
        "FD_IREF_OVERRIDE_ENABLE. The state-dependent target is"
    ),
    EQ(
        "i0ref = rho IFD + (1-rho)uv in charge; -rho IFD + (1-rho)uv in discharge; 0 otherwise.",
        r"i_{ref}^{0}[k]=\begin{cases}\rho I_{FD}+(1-\rho)u_v[k],&\text{Charge},\\-\rho I_{FD}+(1-\rho)u_v[k],&\text{Discharge},\\0,&\text{otherwise}.\end{cases}",
        "(7)",
    ),
    P(
        "The Section V runs set rho=1 and IFD=20 A, so their bus voltages are operating-"
        "point responses, not a claim of 400-V regulation. The command is rate-limited to "
        "+/-200 A/s and limited to +/-25 A:"
    ),
    EQ(
        "iref[k] = clip(iref[k-1] + clip(i0ref[k]-iref[k-1], -200 Ts, 200 Ts), -25, 25).",
        r"i_{ref}[k]=\operatorname{clip}_{[-25,25]}\!\left(i_{ref}[k-1]+\operatorname{clip}_{[-200T_s,200T_s]}(i_{ref}^{0}[k]-i_{ref}[k-1])\right).",
        "(8)",
    ),
    P(
        "The outer discrete parallel PI uses forward-Euler integration, output saturation, "
        "and back-calculation anti-windup. With Tsv=1 ms, Kpv=2.6, Kiv=50, Kbv=20, and "
        "limits +/-10, its exact recursion is"
    ),
    EQ(
        "uv0[k]=2.6 ev[k]+xiv[k]; uv[k]=clip(uv0[k],-10,10).",
        r"u_v^0[k]=2.6e_v[k]+\xi_v[k],\qquad u_v[k]=\operatorname{clip}_{[-10,10]}(u_v^0[k]).",
        "(9)",
    ),
    EQ(
        "xiv[k+1]=xiv[k]+0.001[50 ev[k]+20(uv[k]-uv0[k])], reset to 0 on reset_pi rising edge.",
        r"\xi_v[k+1]=\xi_v[k]+10^{-3}\!\left[50e_v[k]+20(u_v[k]-u_v^0[k])\right],\quad \xi_v\leftarrow0\ \text{on reset}.",
        "(10)",
    ),
    H(2, "C. Inner Current PI, Feedforward Duty, and PWM"),
    P(
        "With ei[k]=iref[k]-iL,avg[k], the 50-us current controller uses Kpi=0.236, "
        "Kii=50, Kbi=100, output limits +/-0.2, and forward-Euler integration:"
    ),
    EQ(
        "ui0[k]=0.236 ei[k]+xii[k]; ui[k]=clip(ui0[k],-0.2,0.2).",
        r"u_i^0[k]=0.236e_i[k]+\xi_i[k],\qquad u_i[k]=\operatorname{clip}_{[-0.2,0.2]}(u_i^0[k]).",
        "(11)",
    ),
    EQ(
        "xii[k+1]=xii[k]+50e-6[50 ei[k]+100(ui[k]-ui0[k])], reset to 0 on reset_pi rising edge.",
        r"\xi_i[k+1]=\xi_i[k]+50\!\times\!10^{-6}\!\left[50e_i[k]+100(u_i[k]-u_i^0[k])\right],\quad \xi_i\leftarrow0\ \text{on reset}.",
        "(12)",
    ),
    P(
        "feedforward_duty divides battery voltage by max(vdc,1 V), saturates the ratio, "
        "adds the current-PI correction, and applies the final [0,1] limiter:"
    ),
    EQ(
        "dff[k]=clip(vbat[k]/max(vdc[k],1),0,1); d[k]=clip(dff[k]+ui[k],0,1).",
        r"d_{ff}[k]=\operatorname{clip}_{[0,1]}\!\left(\frac{v_{bat}[k]}{\max(v_{dc}[k],1)}\right),\qquad d[k]=\operatorname{clip}_{[0,1]}(d_{ff}[k]+u_i[k]).",
        "(13)",
    ),
    P(
        "The PWM generator runs at fsw=20 kHz with 1-us update. Let p(t;d) be its binary "
        "pulse and tauD=2 us. Delayed-edge AND logic implements nonoverlap:"
    ),
    EQ(
        "g1cmd = Gen p(t;d)p(t-tauD;d); g2cmd = Gen[1-p(t;d)][1-p(t-tauD;d)].",
        r"g_1^{cmd}=G_{en}p(t;d)p(t-\tau_D;d),\qquad g_2^{cmd}=G_{en}[1-p(t;d)][1-p(t-\tau_D;d)].",
        "(14)",
    ),
    T(
        "TABLE II. IMPLEMENTED CONTROL PARAMETERS",
        ["Subsystem", "Parameters", "Implementation"],
        [
            ["Voltage PI", "Kp=2.6, Ki=50, Kb=20, Ts=1 ms", "Forward Euler, output +/-10"],
            ["Current PI", "Kp=0.236, Ki=50, Kb=100, Ts=50 us", "Forward Euler, output +/-0.2"],
            ["Current reference", "+/-25 A, slew +/-200 A/s", "Mode sign and rate limiter"],
            ["Feedforward", "vbat/max(vdc,1 V)", "Saturated to [0,1]"],
            ["PWM", "20 kHz, 1-us update", "Complementary commands"],
            ["Dead time", "2 us", "Delayed-edge AND logic"],
            ["Mode hysteresis", "402/399.5 V charge; 398/400.5 V discharge", "10-ms entry, 20-ms exit"],
            ["Transition release", "|iL,avg|<0.5 A for 5 ms", "Prevents loaded reversal"],
        ],
    ),
]

# Remaining sections are appended to keep the source maintainable.
base.ELEMENTS.extend([
    H(1, "IV. MATHEMATICAL FAULT INJECTION AND DIAGNOSTIC ANALYSIS"),
    H(2, "A. Fault Window and Sensor Measurement Chain"),
    P(
        "All time-local faults use the same window, preventing accidental relabeling of "
        "pre-fault samples. For start tf and end te,"
    ),
    EQ(
        "wf(t)=H(t-tf)-H(t-te).",
        r"w_f(t)=H(t-t_f)-H(t-t_e).",
        "(15)",
    ),
    P(
        "The sensor chain applies deterministic bias, Gaussian noise, quantization, and a "
        "50-us zero-order hold in that order. For signal y, bias b, noise standard "
        "deviation sigma, quantization interval Delta, and tk=kTs,"
    ),
    EQ(
        "ym[k]=Q_Delta{y(tk)+b wf(tk)+sigma eta[k]}, eta~N(0,1); Q_Delta(x)=Delta round(x/Delta).",
        r"y_m[k]=\mathcal Q_{\Delta}\{y(t_k)+b w_f(t_k)+\sigma\eta[k]\},\quad \eta\sim\mathcal N(0,1),\quad \mathcal Q_{\Delta}(x)=\Delta\operatorname{round}(x/\Delta).",
        "(16)",
    ),
    P(
        "The main sensor classes use DC-bus bias bV in {+/-5,+/-10} V and inductor-current "
        "bias bI in {+/-0.5,+/-1} A. Noise standard deviations are 0.05 V for vdc, "
        "0.02 V for vbat, 0.01 A for ibat, and 0.02 A for iL. Voltage quantization is "
        "0.01 V and current quantization is 0.001 A. Thus the ideal bias residual "
        "ry=ym-yhat is approximately b wf plus combined uncertainty; physical "
        "detectability is controlled by |b| relative to model and measurement error."
    ),
    H(2, "B. Full, Partial, and Intermittent Open-Switch Faults"),
    P(
        "Gate-blocking faults are implemented with a normalized periodic phase. For "
        "period Tp and blocked fraction aj in [0,1],"
    ),
    EQ(
        "phi(t)=mod(t,Tp)/Tp; fj(t)=1[aj wf(t) > phi(t)].",
        r"\phi(t)=\frac{\operatorname{mod}(t,T_p)}{T_p},\qquad f_j(t)=\mathbb 1\{a_jw_f(t)>\phi(t)\}.",
        "(17)",
    ),
    P(
        "Since phi is uniform over a period, the blocked-time ratio is approximately aj "
        "inside the window. Full open uses aj=1. Partial open uses aj in {0.25,0.50,0.75} "
        "with Tp=487 us. Intermittent open uses aj in {0.35,0.65} with Tp=40 ms, creating "
        "longer alternating fault and recovery intervals. Actual gate and mismatch are"
    ),
    EQ(
        "gjact(t)=gjcmd(t)[1-fj(t)]; mj(t)=gjcmd XOR gjact.",
        r"g_j^{act}(t)=g_j^{cmd}(t)[1-f_j(t)],\qquad m_j(t)=g_j^{cmd}(t)\oplus g_j^{act}(t).",
        "(18)",
    ),
    P(
        "During an open interval the commanded device loses volt-seconds. A first-order "
        "perturbation of (1) gives L d(delta iL)/dt approximately vdc delta q. Before PI "
        "compensation,"
    ),
    EQ(
        "delta iL(T) approximately -(vdc/L) integral_0^T fj(t) gjcmd(t) dt.",
        r"\Delta i_L(T)\approx-\frac{v_{dc}}{L}\int_0^T f_j(t)g_j^{cmd}(t)\,dt.",
        "(19)",
    ),
    P(
        "Equation (19) explains why a gate fault is observable only when the affected "
        "device was commanded to conduct. Training eligibility therefore requires converter "
        "enable, sufficient commanded gate duty, active fault time, and exclusion of the "
        "10-ms transition window."
    ),
    H(2, "C. High On-Resistance Fault and Direct Physics Estimator"),
    P(
        "High-resistance degradation is not emulated by deleting a gate pulse. The IGBT "
        "Ron parameter is changed from R0=1 mOhm to Rf for the whole run. Its averaged "
        "resistance and loss contribution are"
    ),
    EQ(
        "Rsw(d)=d R1+(1-d)R2; Ploss,sw approximately [dR1+(1-d)R2] iL^2.",
        r"R_{sw}(d)=dR_1+(1-d)R_2,\qquad P_{loss,sw}\approx[dR_1+(1-d)R_2]i_L^2.",
        "(20)",
    ),
    P(
        "At quasi-steady current, the current loop changes duty to balance the added "
        "voltage drop. Solving the first line of (1) yields"
    ),
    EQ(
        "d* = [eoc+(rSigma+R2)iL]/[vdc-(R1-R2)iL].",
        r"d^{\star}=\frac{e_{oc}+(r_{\Sigma}+R_2)i_L}{v_{dc}-(R_1-R_2)i_L}.",
        "(21)",
    ),
    P(
        "Thus delta Ron may appear mainly as delta duty rather than delta current, which "
        "explains the overlapping system-current traces in Fig. 3(a). Direct switch "
        "sensing removes this closed-loop masking. The v06 model logs device voltage and "
        "current at 1 us; in each 50-us bin Bk, the implemented estimate is"
    ),
    EQ(
        "chi_j[n]=1[|ij[n]|>=0.5 A]; Rhat_j[k]=median_{n in Bk,chi=1}|vj[n]/ij[n]|.",
        r"\chi_j[n]=\mathbb 1\{|i_j[n]|\ge0.5\ \mathrm A\},\qquad \widehat R_j[k]=\operatorname*{median}_{n\in\mathcal B_k:\chi_j[n]=1}\left|\frac{v_j[n]}{i_j[n]}\right|.",
        "(22)",
    ),
    P(
        "The median rejects switching-edge outliers and prevents off-state bus voltage "
        "from being interpreted as on-state resistance. A frozen run decision is "
        "D=1{median(Rhat in qualified windows)>10.5 mOhm}. For measured quantities "
        "vm=(1+gv)v+bv+nv and im=(1+gi)i+bi+ni, first-order propagation gives"
    ),
    EQ(
        "delta Rhat/R approximately gv-gi + bv/v - bi/i + nv/v - ni/i.",
        r"\frac{\Delta\widehat R}{R}\approx g_v-g_i+\frac{b_v}{v}-\frac{b_i}{i}+\frac{n_v}{v}-\frac{n_i}{i}.",
        "(23)",
    ),
    P(
        "When v=Ri is only a few millivolts, voltage offset and timing dominate. Under "
        "Gaussian voltage noise, resistance information is proportional to "
        "sum(chi i^2)/sigma_v^2; high conduction current, synchronized sampling, and low "
        "offset are therefore more useful than classifier depth."
    ),
    T(
        "TABLE III. IMPLEMENTED FAULT FAMILIES AND PARAMETERS",
        ["Fault family", "Mathematical setting", "Development values"],
        [
            ["DC-bus sensor bias", "bV wf(t) in (16)", "+/-5 and +/-10 V"],
            ["Inductor-current bias", "bI wf(t) in (16)", "+/-0.5 and +/-1 A"],
            ["S1/S2 full open", "a=1 in (17)", "50, 150, 300 ms duration"],
            ["S1/S2 partial open", "a in (0,1), Tp=487 us", "a=0.25, 0.50, 0.75"],
            ["S1/S2 intermittent", "periodic blocking, Tp=40 ms", "a=0.35, 0.65"],
            ["S1/S2 high Ron", "replace device Ron in (20)", "10, 30, 100 mOhm; validation 50 mOhm"],
        ],
        "High-R begins at t=0 because Ron is a compile-time block parameter in the current implementation.",
    ),
    H(2, "D. General Residual Sensitivity and Observability"),
    P(
        "Linearize the closed-loop model at an operating point as delta xdot=Acl delta x "
        "+Bf f and delta y=C delta x+Df f+n. Over horizon T, the fault-to-output "
        "information is"
    ),
    EQ(
        "Jf(T)=integral_0^T h_f(t)^T Rn^-1 h_f(t) dt, h_f(t)=C exp(Acl t)Bf+Df.",
        r"J_f(T)=\int_0^T\!\mathbf h_f^{T}(t)\mathbf R_n^{-1}\mathbf h_f(t)\,dt,\qquad \mathbf h_f(t)=\mathbf C e^{\mathbf A_{cl}t}\mathbf B_f+\mathbf D_f.",
        "(24)",
    ),
    P(
        "Sensor bias has a direct Df term, and device voltage has "
        "partial(von)/partial(Ron)=isw. High Ron observed only through iL and vdc relies "
        "on the dynamic term C exp(Acl t)Bf, which is attenuated by current control. This "
        "formalizes the improvement obtained by adding switch voltage and current."
    ),

    H(1, "V. SIMULATION DESIGN AND PRIMARY RESULTS"),
    H(2, "A. Numerical Protocol"),
    P(
        "The power network uses powergui discrete mode at 1 us and the model stores ode3 "
        "fixed-step configuration. PWM is 20 kHz. General analysis signals are exported "
        "at 50 us; switch voltage and current are acquired at 1 us and summarized in each "
        "50-us bin. Healthy waveform runs use SOC=50%, IFD=20 A, vdc reference 400 V, "
        "RH=200 ohm, and a 600-W load at 0.35 s. Matched healthy/fault comparisons use the "
        "same random seed. Simulation ends at 1.0 s."
    ),
    H(2, "B. Bidirectional Start-Up and Load Step"),
    F(
        "word/healthy_bidirectional_waveforms.jpg",
        "Fig. 2. Healthy current-reference operation in charge and discharge, including converter enable, the 600-W load step, and duty compensation.",
        "Orange Mode 1 is charge/buck with positive iL; green Mode 2 is discharge/boost with negative iL. Direct current-reference override is enabled.",
    ),
    P(
        "The rate limiter produces the expected 0-to-20-A ramp in approximately 0.10 s. "
        "After 0.90 s, charge-current mean is 19.996 A with standard deviation 0.022 A; "
        "discharge-current mean is -20.010 A with standard deviation 0.032 A. The measured "
        "load increase is 1.48-1.49 A. Mean bus-voltage changes are -3.23 V (-0.87%) in "
        "charge and -2.89 V (-0.70%) in discharge. Duty shifts while current stays at its "
        "command, consistent with (13)."
    ),
    T(
        "TABLE IV. HEALTHY WAVEFORM METRICS",
        ["Metric", "Charge / buck", "Discharge / boost"],
        [
            ["Current reference", "+20 A", "-20 A"],
            ["Steady current mean", "19.996 A", "-20.010 A"],
            ["Steady current standard deviation", "0.022 A", "0.032 A"],
            ["Pre-step mean vdc", "371.465 V", "415.002 V"],
            ["Post-step mean vdc", "368.237 V", "412.116 V"],
            ["Mean vdc change", "-3.228 V (-0.869%)", "-2.886 V (-0.695%)"],
            ["Post-step mean duty", "0.6556", "0.4335"],
        ],
        "Means are taken from stated simulation windows and are not hardware measurements.",
    ),
    H(2, "C. High-Resistance Observability Result"),
    F(
        "word/switch_high_r_observability.jpg",
        "Fig. 3. High-resistance observability: overlapping system current, direct S1 voltage-current evidence, the S2 resistance estimate, and independent unseen-level decisions.",
        "Eight healthy and eight fault validation runs are independent simulations; windows within a run are not independent samples.",
    ),
    P(
        "Healthy and 50-mOhm current traces nearly overlap, confirming the masking predicted "
        "by (21). In contrast, S1 on-state voltage rises from about 20 mV at 20 A to about "
        "1 V, and (22) recovers 1 and 50 mOhm. Across 16 independent unseen-level runs, all "
        "eight fault runs exceeded 10.5 mOhm and no healthy run did. This is clean ideal-"
        "model separation, but the sample size is 16 and the IGBT model omits temperature-"
        "dependent VCE(sat), package inductance, common-mode error, and probe bandwidth."
    ),
    F(
        "word/high_r_detection_curve.jpg",
        "Fig. 4. Measurement-chain stress projection versus switch resistance under increasing noise, gain, offset, quantization, healthy-Ron drift, and timing misalignment.",
        "The 4320 projected cases use 24 existing Simulink current trajectories; this is not a new dynamic power-stage simulation or hardware validation.",
    ),
    P(
        "The projection evaluates 1-50 mOhm and four measurement conditions. At 12 mOhm, "
        "moderate and harsh scenarios each detect 120/120 cases, with exact 95% lower bound "
        "0.9697 and zero pre-fault false alarms; median latency is 12.91 and 12.29 ms. "
        "Faults at or below 8 mOhm are not reliably above threshold, and 10 mOhm is a weak "
        "transition region. The supported projected boundary is therefore at least 12 mOhm."
    ),
])

base.ELEMENTS.extend([
    H(1, "VI. SECONDARY GROUPED MACHINE-LEARNING ASSESSMENT"),
    H(2, "A. Data Representation and Leakage Control"),
    P(
        "The secondary five-class table contains 416 independent simulation runs over "
        "28 operating points. Event-centered 10-ms windows with 5-ms step are aggregated "
        "to 660 numeric features. Labels, configured fault magnitudes, file names, seeds, "
        "and eligibility variables are excluded from predictors. All windows and the run "
        "vote from one operating point remain in one fold. For fold l with group set Gl,"
    ),
    EQ(
        "Train_l={r:g(r) notin Gl}; Test_l={r:g(r) in Gl}; train/test operating-point intersection is empty.",
        r"\mathcal T_\ell=\{r:g(r)\notin\mathcal G_\ell\},\quad \mathcal S_\ell=\{r:g(r)\in\mathcal G_\ell\},\quad g(\mathcal T_\ell)\cap g(\mathcal S_\ell)=\varnothing.",
        "(25)",
    ),
    P(
        "Six grouped folds generate exactly one out-of-fold prediction per run. The feature "
        "table is highly structured: 26 features are constant, 5803 pairs have absolute "
        "correlation above 0.95, PCA needs 10, 21, and 53 components for 90%, 95%, and "
        "99% variance, and entropy effective rank is 9.758. For covariance eigenvalues "
        "lambda_j and pj=lambda_j/sum(lambda),"
    ),
    EQ(
        "reff=exp[-sum_j pj log pj].",
        r"r_{eff}=\exp\!\left(-\sum_jp_j\log p_j\right),\qquad p_j=\lambda_j/\sum_k\lambda_k.",
        "(26)",
    ),
    F(
        "word/pca_geometry.jpg",
        "Fig. 5. Principal-component geometry of the 660-feature within-simulation event table.",
        "PCA is distribution analysis only and does not establish transfer to a different simulator or hardware domain.",
    ),
    H(2, "B. Models, Metrics, and Confidence Bounds"),
    P(
        "Candidates are regularized multinomial logistic regression, random forest, "
        "extremely randomized trees, XGBoost, distance-weighted KNN, MLP, and 1D-CNN. "
        "The 1D-CNN treats an engineered feature vector as an ordered input even though "
        "adjacent positions do not necessarily represent physical time. Run decisions use "
        "out-of-fold window probabilities. For C classes,"
    ),
    EQ(
        "MacroF1=(1/C)sum_c 2PcRc/(Pc+Rc).",
        r"\operatorname{MacroF1}=\frac1C\sum_{c=1}^{C}\frac{2P_cR_c}{P_c+R_c}.",
        "(27)",
    ),
    P(
        "Exact Clopper-Pearson intervals use independent runs, not windows. For k successes "
        "in n runs, the two-sided 95% accuracy interval is"
    ),
    EQ(
        "[BetaInv(0.025;k,n-k+1), BetaInv(0.975;k+1,n-k)].",
        r"\left[B^{-1}(0.025;k,n-k+1),\ B^{-1}(0.975;k+1,n-k)\right].",
        "(28)",
    ),
    F(
        "word/main_model_run_metrics.jpg",
        "Fig. 6. Grouped out-of-fold run-level performance of seven secondary comparison models.",
        "All scores are internal to the current Simulink data-generating process.",
    ),
    T(
        "TABLE V. SECONDARY WITHIN-SIMULATION MODEL COMPARISON",
        ["Model", "Run accuracy", "Macro-F1", "Healthy FAR", "Interpretation"],
        [
            ["Logistic regression", "1.0000", "1.0000", "0.0000", "Preferred calibrated baseline"],
            ["Random forest", "1.0000", "1.0000", "0.0000", "Nonlinear tree reference"],
            ["Extra Trees", "1.0000", "1.0000", "0.0000", "Randomized tree reference"],
            ["KNN", "0.9880", "0.9901", "0.0000", "Local feature geometry"],
            ["XGBoost", "0.9904", "0.9887", "0.0000", "Strong but not superior"],
            ["MLP", "0.9712", "0.9763", "0.0000", "No complexity benefit"],
            ["1D-CNN", "0.8125", "0.8466", "0.3194", "Ordering mismatched to convolution"],
        ],
        "For the three perfect point estimates, exact 95% accuracy lower bound is 0.9912. With zero false alarms among 144 healthy runs, the two-sided 95% upper bound is 0.0253.",
    ),
    P(
        "The perfect point estimates do not show that the physical problem is solved; they "
        "show separability inside the simulated design. Low effective rank explains why a "
        "regularized linear model matches trees and why 1D-CNN adds no credible advantage. "
        "Because external validation is absent, fusion is not used to inflate the headline. "
        "After future hardware validation, calibrated logistic probabilities could be "
        "combined with the independent physics alarm in (22), using a conservative OR for "
        "protection and a two-channel agreement rule for maintenance classification."
    ),

    H(1, "VII. DISCUSSION AND CLAIM BOUNDARY"),
    H(2, "A. Why the Simulation and Equations Are the Main Contribution"),
    P(
        "The strongest result is not 1.000 in Table V. It is the traceable chain from "
        "circuit equations, switching implementation, mode guards, fault operators, and "
        "sensor placement to a diagnostic decision. Equations (1), (13), and (21) predict "
        "that current regulation can conceal switch resistance by changing duty. Equation "
        "(22) supplies a direct physical channel not dependent on the global closed-loop "
        "response, and Fig. 3 matches that prediction. This is stronger than reporting "
        "another classifier with a slightly different internal score."
    ),
    H(2, "B. What Is and Is Not Validated"),
    T(
        "TABLE VI. EVIDENCE LADDER AND PERMITTED CLAIMS",
        ["Evidence", "Available result", "Permitted claim"],
        [
            ["Switched Simulink model", "Bidirectional start-up and load step", "Numerical model behavior"],
            ["Independent Simulink runs", "16 unseen 50-mOhm runs", "Ideal-model threshold separation"],
            ["Measurement projection", "4320 perturbed cases", "Estimator stress sensitivity only"],
            ["Grouped OOF ML", "416 runs, 28 groups", "Within-simulation class separation"],
            ["External simulator/domain", "Not available", "No cross-domain generalization"],
            ["HIL/hardware experiment", "Not available", "No deployment or real-time claim"],
        ],
    ),
    P(
        "The model does not yet include semiconductor junction temperature, nonlinear "
        "VCE(sat), switching energy, package parasitics, probe isolation, ADC aperture, "
        "computation delay, or EMI. High Ron is fixed from t=0 in full power-stage runs "
        "because the parameter is compile-time. The later measurement projection applies "
        "a resistance step to existing current trajectories and therefore does not reproduce "
        "dynamic feedback of degradation on the power stage."
    ),
    P(
        "The ML table also lacks a genuine external source. A supplemental synthetic HDF5 "
        "domain was investigated, but it was generated rather than measured and showed "
        "substantial distribution shift; it is a negative audit, not external validation. "
        "The manuscript must remain positioned as a simulation and computer-evidence paper."
    ),
    H(2, "C. Recommended Next Validation Step"),
    P(
        "If one future validation layer becomes possible, its highest value is not another "
        "neural network. It is processor-in-the-loop or a small converter test recording "
        "synchronized gate state, switch voltage/current, inductor current, and bus voltage "
        "under controlled added resistance. It should include 1, 8, 10, 12, 20, and "
        "50 mOhm equivalents, multiple currents and temperatures, and truly held-out runs. "
        "Those data would test (23) and whether 10.5 mOhm survives nonideal VCE and offset."
    ),

    H(1, "VIII. CONCLUSION"),
    P(
        "A Simulink-led, equation-consistent method was developed for a bidirectional "
        "battery converter. The switched and averaged plant, discrete PI recursions, "
        "supervisory guards, feedforward duty, PWM dead time, sensor chain, gate-blocking "
        "patterns, high-resistance physics, and power residual were documented. Healthy "
        "simulations showed stable +/-20-A current-reference operation and sub-1% mean "
        "bus-voltage change under the applied load step. High switch resistance exposed "
        "a limitation: closed-loop current is a weak diagnostic signal. Direct device "
        "voltage-current sensing restored sensitivity and separated nominal and unseen "
        "50-mOhm conditions in the ideal model. Secondary grouped ML favored simple "
        "logistic and tree models, but no external or hardware claim is made. For current "
        "evidence, fault equations and observability design are the central contribution."
    ),
    H(1, "ACKNOWLEDGMENT"),
    P(
        "This work was supported by the University Development Fund of The Chinese "
        "University of Hong Kong, Shenzhen [VERIFY: replace with the official English fund "
        "name and grant number supplied by the University before submission]."
    ),
    H(1, "DATA AND CODE AVAILABILITY"),
    P(
        "The Simulink model, MATLAB generation scripts, run tables, feature definitions, "
        "grouped evaluation code, and figure scripts are maintained in the project archive. "
        "[VERIFY: insert a permanent repository or approved data-availability statement.]"
    ),
    H(1, "AUTHOR CONTRIBUTIONS"),
    P(
        "[VERIFY WITH ALL AUTHORS.] Mingyu Li: conceptualization, methodology, simulation, "
        "software, analysis, visualization, and original draft. Qiang Wang, Xiaojing Xiong, "
        "and Bianjia Wang: validation, investigation, and review. Zhongxin Cheng and Ying "
        "Jia: supervision, resources, project administration, and review."
    ),
])

base.CHECKLIST = """# Simulink强化版论文使用说明与投稿前核对

这份稿件已经把文章主轴改为“Simulink建模—控制方程—故障注入—可观测性—仿真证据”，机器学习只保留为次要的组内仿真比较。正文不声称硬件、HIL、实验或真实外部数据验证。

## 已经写实并可保留

- 最终模型的关键模块、功率级参数、1 μs电气步长、20 kHz PWM、PI参数、限幅和反算抗饱和参数。
- 充电为正电感电流、放电为负电感电流的统一符号约定，以及开关模型和平均状态空间模型。
- Stateflow的充放电阈值、保持时间、5 ms预充电、0.5 A零电流换向和故障锁止逻辑。
- 传感器偏置、全开路、部分开路、间歇开路和高导通电阻的数学公式与注入参数。
- 健康双向运行、600 W负载阶跃和高阻V-I可观测性图证。
- 416 Run、28工况、660特征、6折工况分组OOF的次要机器学习结果。

## 投稿前必须核对

1. 确认S1为上管、S2为下管，以及图中的功率流方向与最终接线一致。
2. 确认Battery端电压和外接Rbat的测量位置；式(1)已用rSigma避免重复计算压降。
3. 向学校确认基金正式英文名与项目号；当前基金句子是待核对占位语句。
4. 补全两位通讯作者邮箱、作者贡献和共同二作书面确认。
5. 选定IEEE会议或期刊后按页数重排；优先保留式(1)、(11)-(14)、(17)-(24)和Fig. 1-4。
6. 若不做实验，标题、摘要、结论和投稿信都必须保留simulation-only边界。
7. 若以后获得HIL或硬件数据，必须冻结阈值和模型后再测试，不能重新调参后称为外部验证。
8. 当前暂用项目文件夹内13篇参考文献；目标期刊可能要求补充控制、IGBT退化和可观测性原始文献。

## 输出文件

- paper_simulink_enhanced_ieee.docx：可编辑Word完整范文。
- paper_simulink_enhanced_ieee.tex：IEEEtran LaTeX源稿。
- paper_simulink_enhanced_ieee_en.md：英文全文检索稿。
- references.bib：当前13篇本地参考文献。
- figures：架构图、波形图、高阻诊断图和次要ML图。
- simulation_waveform_metrics.json：图中数值的机器可读汇总。
"""


def require_figures() -> None:
    required = [
        "word/simulink_implementation_architecture.jpg",
        "word/healthy_bidirectional_waveforms.jpg",
        "word/switch_high_r_observability.jpg",
        "word/high_r_detection_curve.jpg",
        "word/pca_geometry.jpg",
        "word/main_model_run_metrics.jpg",
    ]
    missing = [name for name in required if not (FIG / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing paper figures: {missing}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    require_figures()
    (OUT / "paper_simulink_enhanced_ieee_en.md").write_text(
        base.build_markdown(), encoding="utf-8"
    )
    (OUT / "paper_simulink_enhanced_ieee.tex").write_text(
        base.build_latex(), encoding="utf-8"
    )
    (OUT / "references.bib").write_text(base.build_bibtex(), encoding="utf-8")
    (OUT / "submission_checklist_cn.md").write_text(
        base.CHECKLIST, encoding="utf-8"
    )
    base.build_docx()
    generated = OUT / "paper_template_ieee.docx"
    target = OUT / "paper_simulink_enhanced_ieee.docx"
    if target.exists():
        target.unlink()
    generated.replace(target)
    print(target)


if __name__ == "__main__":
    main()
