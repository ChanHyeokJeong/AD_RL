import numpy as np
import scipy.integrate
import pandas as pd
from scipy import integrate
import os

print("START")

R = 0.083145
T_base = 298.15
T_ad = 308.15
T_op = T_ad  # Backward-compatible alias; use T_ad as the single digester temperature.
p_atm = 1.013

f_sI_xc = 0.1
f_xI_xc = 0.2
f_ch_xc = 0.2
f_pr_xc = 0.2
f_li_xc = 0.3
N_xc = 0.0376 / 14
N_I = 0.06 / 14
N_aa = 0.007
C_xc = 0.02786
C_sI = 0.03
C_ch = 0.0313
C_pr = 0.03
C_li = 0.022
C_xI = 0.03
C_su = 0.0313
C_aa = 0.03
f_fa_li = 0.95
C_fa = 0.0217
f_h2_su = 0.19
f_bu_su = 0.13
f_pro_su = 0.27
f_ac_su = 0.41
N_bac = 0.08 / 14
C_bu = 0.025
C_pro = 0.0268
C_ac = 0.0313
C_bac = 0.0313
Y_su = 0.1
f_h2_aa = 0.06
f_va_aa = 0.23
f_bu_aa = 0.26
f_pro_aa = 0.05
f_ac_aa = 0.40
C_va = 0.024
Y_aa = 0.08
Y_fa = 0.06
Y_c4 = 0.06
Y_pro = 0.04
C_ch4 = 0.0156
Y_ac = 0.05
Y_h2 = 0.06

k_dis = 0.5
k_hyd_ch = 10
k_hyd_pr = 10
k_hyd_li = 10
K_S_IN = 10 ** -4
k_m_su = 30
K_S_su = 0.5
pH_UL_aa = 5.5
pH_LL_aa = 4
k_m_aa = 50
K_S_aa = 0.3
k_m_fa = 6
K_S_fa = 0.4
K_I_h2_fa = 5 * 10 ** -6
k_m_c4 = 20
K_S_c4 = 0.2
K_I_h2_c4 = 10 ** -5
k_m_pro = 13
K_S_pro = 0.1
K_I_h2_pro = 3.5 * 10 ** -6
k_m_ac = 8
K_S_ac = 0.15
K_I_nh3 = 0.0018
pH_UL_ac = 7
pH_LL_ac = 6
k_m_h2 = 35
K_S_h2 = 7 * 10 ** -6
pH_UL_h2 = 6
pH_LL_h2 = 5
k_dec_X_su = 0.02
k_dec_X_aa = 0.02
k_dec_X_fa = 0.02
k_dec_X_c4 = 0.02
k_dec_X_pro = 0.02
k_dec_X_ac = 0.02
k_dec_X_h2 = 0.02

K_w = 10 ** -14.0 * np.exp((55900 / (100 * R)) * (1 / T_base - 1 / T_ad))
K_a_va = 10 ** -4.86
K_a_bu = 10 ** -4.82
K_a_pro = 10 ** -4.88
K_a_ac = 10 ** -4.76
K_a_co2 = 10 ** -6.35 * np.exp((7646 / (100 * R)) * (1 / T_base - 1 / T_ad))
K_a_IN = 10 ** -9.25 * np.exp((51965 / (100 * R)) * (1 / T_base - 1 / T_ad))

k_A_B_va = 10 ** 10
k_A_B_bu = 10 ** 10
k_A_B_pro = 10 ** 10
k_A_B_ac = 10 ** 10
k_A_B_co2 = 10 ** 10
k_A_B_IN = 10 ** 10

p_gas_h2o = 0.0313 * np.exp(5290 * (1 / T_base - 1 / T_ad))
k_p = 5 * 10 ** 4
k_L_a = 200.0
K_H_co2 = 0.035 * np.exp((-19410 / (100 * R)) * (1 / T_base - 1 / T_ad))
K_H_ch4 = 0.0014 * np.exp((-14240 / (100 * R)) * (1 / T_base - 1 / T_ad))
K_H_h2 = 7.8 * 10 ** -4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T_ad))

V_liq = 3400
V_gas = 300
V_ad = V_liq + V_gas

base_dir = os.path.dirname(__file__)
influent_path = os.path.join(base_dir, 'digester_influent_mean_full.csv')
initial_path = os.path.join(base_dir, 'digester_initial.csv')

influent_state = pd.read_csv(influent_path)
initial_state = pd.read_csv(initial_path)

pH_sp = 7.00
Kp = 28.0  # Backward-compatible alias for NaOH PI proportional gain.
Ki = 12.0  # Backward-compatible alias for NaOH PI integral gain.
Kp_NaOH = Kp
Ki_NaOH = Ki
Kp_HCl = Kp
Ki_HCl = Ki
C_NAOH = 25.0  # kmol/m3, selected concentrated NaOH basis
C_HCL = 11.3  # kmol/m3, approx. 35 wt% HCl with density near 1.18 kg/L
q_NaOH = 0.0  # m3/d as NaOH solution flow
u_NaOH = 0.0  # kmol/d as NaOH molar dosing rate
q_HCl = 0.0  # m3/d as HCl solution flow
u_HCl = 0.0  # kmol/d as HCl molar dosing rate
q_NaOH_min = 0.0
q_NaOH_max = 100.0
q_HCl_min = 0.0
q_HCl_max = 100.0
err_int = 0.0
err_int_NaOH = 0.0
err_int_HCl = 0.0
def get_pH_sp(t_day):
    base = 7.0
    cycle = t_day % 20.0
    
    if cycle < 5:
        return base
    elif cycle < 10:
        return base * 1.2
    elif cycle < 15:
        return base
    else:
        return base * 0.8
    
def setInfluent(i):
    global S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, S_ac_in, S_h2_in, S_ch4_in
    global S_IC_in, S_IN_in, S_I_in, X_xc_in, X_ch_in, X_pr_in, X_li_in, X_su_in, X_aa_in
    global X_fa_in, X_c4_in, X_pro_in, X_ac_in, X_h2_in, X_I_in, S_cation_in, S_anion_in

    S_su_in = influent_state['S_su'][i]
    S_aa_in = influent_state['S_aa'][i]
    S_fa_in = influent_state['S_fa'][i]
    S_va_in = influent_state['S_va'][i]
    S_bu_in = influent_state['S_bu'][i]
    S_pro_in = influent_state['S_pro'][i]
    S_ac_in = influent_state['S_ac'][i]
    S_h2_in = influent_state['S_h2'][i]
    S_ch4_in = influent_state['S_ch4'][i]
    S_IC_in = influent_state['S_IC'][i]
    S_IN_in = influent_state['S_IN'][i]
    S_I_in = influent_state['S_I'][i]

    X_xc_in = influent_state['X_xc'][i]
    X_ch_in = influent_state['X_ch'][i]
    X_pr_in = influent_state['X_pr'][i]
    X_li_in = influent_state['X_li'][i]
    X_su_in = influent_state['X_su'][i]
    X_aa_in = influent_state['X_aa'][i]
    X_fa_in = influent_state['X_fa'][i]
    X_c4_in = influent_state['X_c4'][i]
    X_pro_in = influent_state['X_pro'][i]
    X_ac_in = influent_state['X_ac'][i]
    X_h2_in = influent_state['X_h2'][i]
    X_I_in = influent_state['X_I'][i]

    S_cation_in = influent_state['S_cation'][i]
    S_anion_in = influent_state['S_anion'][i]

S_su = initial_state['S_su'][0]
S_aa = initial_state['S_aa'][0]
S_fa = initial_state['S_fa'][0]
S_va = initial_state['S_va'][0]
S_bu = initial_state['S_bu'][0]
S_pro = initial_state['S_pro'][0]
S_ac = initial_state['S_ac'][0]
S_h2 = initial_state['S_h2'][0]
S_ch4 = initial_state['S_ch4'][0]
S_IC = initial_state['S_IC'][0]
S_IN = initial_state['S_IN'][0]
S_I = initial_state['S_I'][0]

X_xc = initial_state['X_xc'][0]
X_ch = initial_state['X_ch'][0]
X_pr = initial_state['X_pr'][0]
X_li = initial_state['X_li'][0]
X_su = initial_state['X_su'][0]
X_aa = initial_state['X_aa'][0]
X_fa = initial_state['X_fa'][0]
X_c4 = initial_state['X_c4'][0]
X_pro = initial_state['X_pro'][0]
X_ac = initial_state['X_ac'][0]
X_h2 = initial_state['X_h2'][0]
X_I = initial_state['X_I'][0]

S_cation = initial_state['S_cation'][0]
S_anion = initial_state['S_anion'][0]

pH = 7.4655377
S_H_ion = initial_state['S_H_ion'][0]
S_va_ion = initial_state['S_va_ion'][0]
S_bu_ion = initial_state['S_bu_ion'][0]
S_pro_ion = initial_state['S_pro_ion'][0]
S_ac_ion = initial_state['S_ac_ion'][0]
S_hco3_ion = initial_state['S_hco3_ion'][0]
S_nh3 = initial_state['S_nh3'][0]
S_nh4_ion = 0.0041
S_co2 = 0.14
S_gas_h2 = initial_state['S_gas_h2'][0]
S_gas_ch4 = initial_state['S_gas_ch4'][0]
S_gas_co2 = initial_state['S_gas_co2'][0]

K_pH_aa = 10 ** (-1 * (pH_LL_aa + pH_UL_aa) / 2.0)
nn_aa = 3.0 / (pH_UL_aa - pH_LL_aa)
K_pH_ac = 10 ** (-1 * (pH_LL_ac + pH_UL_ac) / 2.0)
n_ac = 3.0 / (pH_UL_ac - pH_LL_ac)
K_pH_h2 = 10 ** (-1 * (pH_LL_h2 + pH_UL_h2) / 2.0)
n_h2 = 3.0 / (pH_UL_h2 - pH_LL_h2)

pH = -np.log10(S_H_ion)

setInfluent(0)
q_ad = 178.4674

state_zero = [
    S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I,
    X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I,
    S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion,
    S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2
]

state_input = [
    S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, S_ac_in, S_h2_in,
    S_ch4_in, S_IC_in, S_IN_in, S_I_in, X_xc_in, X_ch_in, X_pr_in, X_li_in,
    X_su_in, X_aa_in, X_fa_in, X_c4_in, X_pro_in, X_ac_in, X_h2_in, X_I_in,
    S_cation_in, S_anion_in
]

def PI_pH_controller(pH_sp, pH_meas, dt):
    global err_int, err_int_NaOH, err_int_HCl
    global u_NaOH, q_NaOH, u_HCl, q_HCl
    global Kp, Ki, Kp_NaOH, Ki_NaOH, Kp_HCl, Ki_HCl
    global q_NaOH_min, q_NaOH_max, q_HCl_min, q_HCl_max, C_NAOH, C_HCL

    err = pH_sp - pH_meas

    if err >= 0.0:
        err_int_trial = err_int_NaOH + err * dt
        q_unsat = Kp_NaOH * err + Ki_NaOH * err_int_trial
        q_sat = max(q_NaOH_min, min(q_unsat, q_NaOH_max))
        if q_sat == q_unsat:
            err_int_NaOH = err_int_trial
        err_int_HCl = 0.0
        err_int = err_int_NaOH

        q_NaOH = q_sat
        u_NaOH = C_NAOH * q_NaOH
        q_HCl = 0.0
        u_HCl = 0.0
    else:
        acid_err = -err
        err_int_trial = err_int_HCl + acid_err * dt
        q_unsat = Kp_HCl * acid_err + Ki_HCl * err_int_trial
        q_sat = max(q_HCl_min, min(q_unsat, q_HCl_max))
        if q_sat == q_unsat:
            err_int_HCl = err_int_trial
        err_int_NaOH = 0.0
        err_int = -err_int_HCl

        q_NaOH = 0.0
        u_NaOH = 0.0
        q_HCl = q_sat
        u_HCl = C_HCL * q_HCl

def ADM1_ODE(t, state_zero):
    global S_nh4_ion, S_co2, p_gas, q_gas, q_ch4, u_NaOH, u_HCl

    S_su = state_zero[0]
    S_aa = state_zero[1]
    S_fa = state_zero[2]
    S_va = state_zero[3]
    S_bu = state_zero[4]
    S_pro = state_zero[5]
    S_ac = state_zero[6]
    S_h2 = state_zero[7]
    S_ch4 = state_zero[8]
    S_IC = state_zero[9]
    S_IN = state_zero[10]
    S_I = state_zero[11]
    X_xc = state_zero[12]
    X_ch = state_zero[13]
    X_pr = state_zero[14]
    X_li = state_zero[15]
    X_su = state_zero[16]
    X_aa = state_zero[17]
    X_fa = state_zero[18]
    X_c4 = state_zero[19]
    X_pro = state_zero[20]
    X_ac = state_zero[21]
    X_h2 = state_zero[22]
    X_I = state_zero[23]
    S_cation = state_zero[24]
    S_anion = state_zero[25]
    S_H_ion = state_zero[26]
    S_va_ion = state_zero[27]
    S_bu_ion = state_zero[28]
    S_pro_ion = state_zero[29]
    S_ac_ion = state_zero[30]
    S_hco3_ion = state_zero[31]
    S_co2 = state_zero[32]
    S_nh3 = state_zero[33]
    S_nh4_ion = state_zero[34]
    S_gas_h2 = state_zero[35]
    S_gas_ch4 = state_zero[36]
    S_gas_co2 = state_zero[37]

    S_su_in = state_input[0]
    S_aa_in = state_input[1]
    S_fa_in = state_input[2]
    S_va_in = state_input[3]
    S_bu_in = state_input[4]
    S_pro_in = state_input[5]
    S_ac_in = state_input[6]
    S_h2_in = state_input[7]
    S_ch4_in = state_input[8]
    S_IC_in = state_input[9]
    S_IN_in = state_input[10]
    S_I_in = state_input[11]
    X_xc_in = state_input[12]
    X_ch_in = state_input[13]
    X_pr_in = state_input[14]
    X_li_in = state_input[15]
    X_su_in = state_input[16]
    X_aa_in = state_input[17]
    X_fa_in = state_input[18]
    X_c4_in = state_input[19]
    X_pro_in = state_input[20]
    X_ac_in = state_input[21]
    X_h2_in = state_input[22]
    X_I_in = state_input[23]
    S_cation_in = state_input[24]
    S_anion_in = state_input[25]

    S_nh4_ion = (S_IN - S_nh3)
    S_co2 = (S_IC - S_hco3_ion)

    I_pH_aa = (K_pH_aa ** nn_aa) / (S_H_ion ** nn_aa + K_pH_aa ** nn_aa)
    I_pH_ac = (K_pH_ac ** n_ac) / (S_H_ion ** n_ac + K_pH_ac ** n_ac)
    I_pH_h2 = (K_pH_h2 ** n_h2) / (S_H_ion ** n_h2 + K_pH_h2 ** n_h2)
    I_IN_lim = 1 / (1 + (K_S_IN / S_IN))
    I_h2_fa = 1 / (1 + (S_h2 / K_I_h2_fa))
    I_h2_c4 = 1 / (1 + (S_h2 / K_I_h2_c4))
    I_h2_pro = 1 / (1 + (S_h2 / K_I_h2_pro))
    I_nh3 = 1 / (1 + (S_nh3 / K_I_nh3))

    I_5 = I_pH_aa * I_IN_lim
    I_6 = I_5
    I_7 = I_pH_aa * I_IN_lim * I_h2_fa
    I_8 = I_pH_aa * I_IN_lim * I_h2_c4
    I_9 = I_8
    I_10 = I_pH_aa * I_IN_lim * I_h2_pro
    I_11 = I_pH_ac * I_IN_lim * I_nh3
    I_12 = I_pH_h2 * I_IN_lim

    Rho_1 = k_dis * X_xc
    Rho_2 = k_hyd_ch * X_ch
    Rho_3 = k_hyd_pr * X_pr
    Rho_4 = k_hyd_li * X_li
    Rho_5 = k_m_su * S_su / (K_S_su + S_su) * X_su * I_5
    Rho_6 = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6
    Rho_7 = k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7
    Rho_8 = k_m_c4 * (S_va / (K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8
    Rho_9 = k_m_c4 * (S_bu / (K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9
    Rho_10 = k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10
    Rho_11 = k_m_ac * (S_ac / (K_S_ac + S_ac)) * X_ac * I_11
    Rho_12 = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12
    Rho_13 = k_dec_X_su * X_su
    Rho_14 = k_dec_X_aa * X_aa
    Rho_15 = k_dec_X_fa * X_fa
    Rho_16 = k_dec_X_c4 * X_c4
    Rho_17 = k_dec_X_pro * X_pro
    Rho_18 = k_dec_X_ac * X_ac
    Rho_19 = k_dec_X_h2 * X_h2

    p_gas_h2 = S_gas_h2 * R * T_ad / 16
    p_gas_ch4 = S_gas_ch4 * R * T_ad / 64
    p_gas_co2 = S_gas_co2 * R * T_ad

    p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
    q_gas = k_p * (p_gas - p_atm)
    if q_gas < 0:
        q_gas = 0

    q_ch4 = q_gas * (p_gas_ch4 / p_gas)

    Rho_T_8 = k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2)
    Rho_T_9 = k_L_a * (S_ch4 - 64 * K_H_ch4 * p_gas_ch4)
    Rho_T_10 = k_L_a * (S_co2 - K_H_co2 * p_gas_co2)

    diff_S_su = q_ad / V_liq * (S_su_in - S_su) + Rho_2 + (1 - f_fa_li) * Rho_4 - Rho_5
    diff_S_aa = q_ad / V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6
    diff_S_fa = q_ad / V_liq * (S_fa_in - S_fa) + (f_fa_li * Rho_4) - Rho_7
    diff_S_va = q_ad / V_liq * (S_va_in - S_va) + (1 - Y_aa) * f_va_aa * Rho_6 - Rho_8
    diff_S_bu = q_ad / V_liq * (S_bu_in - S_bu) + (1 - Y_su) * f_bu_su * Rho_5 + (1 - Y_aa) * f_bu_aa * Rho_6 - Rho_9
    diff_S_pro = q_ad / V_liq * (S_pro_in - S_pro) + (1 - Y_su) * f_pro_su * Rho_5 + (1 - Y_aa) * f_pro_aa * Rho_6 + (1 - Y_c4) * 0.54 * Rho_8 - Rho_10
    diff_S_ac = q_ad / V_liq * (S_ac_in - S_ac) + (1 - Y_su) * f_ac_su * Rho_5 + (1 - Y_aa) * f_ac_aa * Rho_6 + (1 - Y_fa) * 0.7 * Rho_7 + (1 - Y_c4) * 0.31 * Rho_8 + (1 - Y_c4) * 0.8 * Rho_9 + (1 - Y_pro) * 0.57 * Rho_10 - Rho_11
    diff_S_h2 = 0
    diff_S_ch4 = q_ad / V_liq * (S_ch4_in - S_ch4) + (1 - Y_ac) * Rho_11 + (1 - Y_h2) * Rho_12 - Rho_T_9

    s_1 = (-1 * C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI)
    s_2 = (-1 * C_ch + C_su)
    s_3 = (-1 * C_pr + C_aa)
    s_4 = (-1 * C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa)
    s_5 = (-1 * C_su + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac)
    s_6 = (-1 * C_aa + (1 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac)
    s_7 = (-1 * C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac)
    s_8 = (-1 * C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac)
    s_9 = (-1 * C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac)
    s_10 = (-1 * C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac)
    s_11 = (-1 * C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac)
    s_12 = ((1 - Y_h2) * C_ch4 + Y_h2 * C_bac)
    s_13 = (-1 * C_bac + C_xc)

    Sigma = (
        s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 + s_5 * Rho_5 +
        s_6 * Rho_6 + s_7 * Rho_7 + s_8 * Rho_8 + s_9 * Rho_9 + s_10 * Rho_10 +
        s_11 * Rho_11 + s_12 * Rho_12 + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
    )

    diff_S_IC = q_ad / V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10
    diff_S_IN = q_ad / V_liq * (S_IN_in - S_IN) + (N_xc - f_xI_xc * N_I - f_sI_xc * N_I - f_pr_xc * N_aa) * Rho_1 - Y_su * N_bac * Rho_5 + (N_aa - Y_aa * N_bac) * Rho_6 - Y_fa * N_bac * Rho_7 - Y_c4 * N_bac * Rho_8 - Y_c4 * N_bac * Rho_9 - Y_pro * N_bac * Rho_10 - Y_ac * N_bac * Rho_11 - Y_h2 * N_bac * Rho_12 + (N_bac - N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
    diff_S_I = q_ad / V_liq * (S_I_in - S_I) + f_sI_xc * Rho_1

    diff_X_xc = q_ad / V_liq * (X_xc_in - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19
    diff_X_ch = q_ad / V_liq * (X_ch_in - X_ch) + f_ch_xc * Rho_1 - Rho_2
    diff_X_pr = q_ad / V_liq * (X_pr_in - X_pr) + f_pr_xc * Rho_1 - Rho_3
    diff_X_li = q_ad / V_liq * (X_li_in - X_li) + f_li_xc * Rho_1 - Rho_4
    diff_X_su = q_ad / V_liq * (X_su_in - X_su) + Y_su * Rho_5 - Rho_13
    diff_X_aa = q_ad / V_liq * (X_aa_in - X_aa) + Y_aa * Rho_6 - Rho_14
    diff_X_fa = q_ad / V_liq * (X_fa_in - X_fa) + Y_fa * Rho_7 - Rho_15
    diff_X_c4 = q_ad / V_liq * (X_c4_in - X_c4) + Y_c4 * Rho_8 + Y_c4 * Rho_9 - Rho_16
    diff_X_pro = q_ad / V_liq * (X_pro_in - X_pro) + Y_pro * Rho_10 - Rho_17
    diff_X_ac = q_ad / V_liq * (X_ac_in - X_ac) + Y_ac * Rho_11 - Rho_18
    diff_X_h2 = q_ad / V_liq * (X_h2_in - X_h2) + Y_h2 * Rho_12 - Rho_19
    diff_X_I = q_ad / V_liq * (X_I_in - X_I) + f_xI_xc * Rho_1

    diff_S_cation = q_ad / V_liq * (S_cation_in - S_cation) + u_NaOH / V_liq
    diff_S_anion = q_ad / V_liq * (S_anion_in - S_anion) + u_HCl / V_liq

    diff_S_H_ion = 0
    diff_S_va_ion = 0
    diff_S_bu_ion = 0
    diff_S_pro_ion = 0
    diff_S_ac_ion = 0
    diff_S_hco3_ion = 0
    diff_S_co2 = 0
    diff_S_nh3 = 0
    diff_S_nh4_ion = 0

    diff_S_gas_h2 = (q_gas / V_gas * -1 * S_gas_h2) + (Rho_T_8 * V_liq / V_gas)
    diff_S_gas_ch4 = (q_gas / V_gas * -1 * S_gas_ch4) + (Rho_T_9 * V_liq / V_gas)
    diff_S_gas_co2 = (q_gas / V_gas * -1 * S_gas_co2) + (Rho_T_10 * V_liq / V_gas)

    return (
        diff_S_su, diff_S_aa, diff_S_fa, diff_S_va, diff_S_bu, diff_S_pro, diff_S_ac,
        diff_S_h2, diff_S_ch4, diff_S_IC, diff_S_IN, diff_S_I,
        diff_X_xc, diff_X_ch, diff_X_pr, diff_X_li, diff_X_su, diff_X_aa, diff_X_fa,
        diff_X_c4, diff_X_pro, diff_X_ac, diff_X_h2, diff_X_I,
        diff_S_cation, diff_S_anion, diff_S_H_ion, diff_S_va_ion, diff_S_bu_ion,
        diff_S_pro_ion, diff_S_ac_ion, diff_S_hco3_ion, diff_S_co2, diff_S_nh3,
        diff_S_nh4_ion, diff_S_gas_h2, diff_S_gas_ch4, diff_S_gas_co2
    )

def simulate(t_step, solvermethod):
    r = scipy.integrate.solve_ivp(ADM1_ODE, t_step, state_zero, method=solvermethod)
    return r.y

def DAESolve():
    global S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_nh3, S_H_ion, pH
    global p_gas_h2, S_h2, S_nh4_ion, S_co2, P_gas, q_gas

    eps = 0.0000001
    prevS_H_ion = S_H_ion

    shdelta = 1.0
    shgradeq = 1.0
    S_h2delta = 1.0
    S_h2gradeq = 1.0
    tol = 10 ** (-12)
    maxIter = 1000
    i = 1
    j = 1

    while ((shdelta > tol or shdelta < -tol) and (i <= maxIter)):
        S_va_ion = K_a_va * S_va / (K_a_va + S_H_ion)
        S_bu_ion = K_a_bu * S_bu / (K_a_bu + S_H_ion)
        S_pro_ion = K_a_pro * S_pro / (K_a_pro + S_H_ion)
        S_ac_ion = K_a_ac * S_ac / (K_a_ac + S_H_ion)
        S_hco3_ion = K_a_co2 * S_IC / (K_a_co2 + S_H_ion)
        S_nh3 = K_a_IN * S_IN / (K_a_IN + S_H_ion)
        shdelta = S_cation + (S_IN - S_nh3) + S_H_ion - S_hco3_ion - S_ac_ion / 64.0 - S_pro_ion / 112.0 - S_bu_ion / 160.0 - S_va_ion / 208.0 - K_w / S_H_ion - S_anion
        shgradeq = 1 + K_a_IN * S_IN / ((K_a_IN + S_H_ion) * (K_a_IN + S_H_ion)) + K_a_co2 * S_IC / ((K_a_co2 + S_H_ion) * (K_a_co2 + S_H_ion)) + 1 / 64.0 * K_a_ac * S_ac / ((K_a_ac + S_H_ion) * (K_a_ac + S_H_ion)) + 1 / 112.0 * K_a_pro * S_pro / ((K_a_pro + S_H_ion) * (K_a_pro + S_H_ion)) + 1 / 160.0 * K_a_bu * S_bu / ((K_a_bu + S_H_ion) * (K_a_bu + S_H_ion)) + 1 / 208.0 * K_a_va * S_va / ((K_a_va + S_H_ion) * (K_a_va + S_H_ion)) + K_w / (S_H_ion * S_H_ion)
        S_H_ion = S_H_ion - shdelta / shgradeq
        if S_H_ion <= 0:
            S_H_ion = tol
        i += 1

    pH = -np.log10(S_H_ion)

    while ((S_h2delta > tol or S_h2delta < -tol) and (j <= maxIter)):
        I_pH_aa = (K_pH_aa ** nn_aa) / (prevS_H_ion ** nn_aa + K_pH_aa ** nn_aa)
        I_pH_h2 = (K_pH_h2 ** n_h2) / (prevS_H_ion ** n_h2 + K_pH_h2 ** n_h2)
        I_IN_lim = 1 / (1 + (K_S_IN / S_IN))
        I_h2_fa = 1 / (1 + (S_h2 / K_I_h2_fa))
        I_h2_c4 = 1 / (1 + (S_h2 / K_I_h2_c4))
        I_h2_pro = 1 / (1 + (S_h2 / K_I_h2_pro))

        I_5 = I_pH_aa * I_IN_lim
        I_6 = I_5
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4
        I_9 = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro
        I_12 = I_pH_h2 * I_IN_lim

        Rho_5 = k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5
        Rho_6 = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6
        Rho_7 = k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7
        Rho_8 = k_m_c4 * (S_va / (K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8
        Rho_9 = k_m_c4 * (S_bu / (K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9
        Rho_10 = k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10
        Rho_12 = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12
        p_gas_h2 = S_gas_h2 * R * T_ad / 16
        Rho_T_8 = k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2)

        S_h2delta = q_ad / V_liq * (S_h2_in - S_h2) + (1 - Y_su) * f_h2_su * Rho_5 + (1 - Y_aa) * f_h2_aa * Rho_6 + (1 - Y_fa) * 0.3 * Rho_7 + (1 - Y_c4) * 0.15 * Rho_8 + (1 - Y_c4) * 0.2 * Rho_9 + (1 - Y_pro) * 0.43 * Rho_10 - Rho_12 - Rho_T_8
        S_h2gradeq = -1.0 / V_liq * q_ad - 3.0 / 10.0 * (1 - Y_fa) * k_m_fa * S_fa / (K_S_fa + S_fa) * X_fa * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_fa) * (1 + S_h2 / K_I_h2_fa)) / K_I_h2_fa - 3.0 / 20.0 * (1 - Y_c4) * k_m_c4 * S_va * S_va / (K_S_c4 + S_va) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4) * (1 + S_h2 / K_I_h2_c4)) / K_I_h2_c4 - 1.0 / 5.0 * (1 - Y_c4) * k_m_c4 * S_bu * S_bu / (K_S_c4 + S_bu) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4) * (1 + S_h2 / K_I_h2_c4)) / K_I_h2_c4 - 43.0 / 100.0 * (1 - Y_pro) * k_m_pro * S_pro / (K_S_pro + S_pro) * X_pro * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_pro) * (1 + S_h2 / K_I_h2_pro)) / K_I_h2_pro - k_m_h2 / (K_S_h2 + S_h2) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) + k_m_h2 * S_h2 / ((K_S_h2 + S_h2) * (K_S_h2 + S_h2)) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) - k_L_a

        S_h2 = S_h2 - S_h2delta / S_h2gradeq
        if S_h2 <= 0:
            S_h2 = tol
        j += 1

t = influent_state['time']

state_columns = [
    "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4",
    "S_IC", "S_IN", "S_I", "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa",
    "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I", "S_cation", "S_anion",
    "S_H_ion", "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion", "S_hco3_ion",
    "S_co2", "S_nh3", "S_nh4_ion", "S_gas_h2", "S_gas_ch4", "S_gas_co2"
]

simulate_results = pd.DataFrame([state_zero], columns=state_columns)
solvermethod = 'DOP853'
CONTROL_INTERVAL_DAYS = 3.0 / 24.0  # 3 h
t0 = 0
n = 0
control_timer = 0.0

gasflow = pd.DataFrame({'q_gas': [0], 'q_ch4': [0]})
control_log = pd.DataFrame({'pH': [pH], 'pH_sp': [pH_sp], 'u_NaOH_kmol_d': [u_NaOH], 'q_NaOH_m3_d': [q_NaOH], 'u_HCl_kmol_d': [u_HCl], 'q_HCl_m3_d': [q_HCl]})
total_ch4 = 0

for u in t[1:]:
    n += 1
    setInfluent(n)

    state_input = [
        S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, S_ac_in, S_h2_in,
        S_ch4_in, S_IC_in, S_IN_in, S_I_in, X_xc_in, X_ch_in, X_pr_in, X_li_in,
        X_su_in, X_aa_in, X_fa_in, X_c4_in, X_pro_in, X_ac_in, X_h2_in, X_I_in,
        S_cation_in, S_anion_in
    ]

    tstep = [t0, u]

    sim = simulate(tstep, solvermethod)

    S_su = sim[0][-1]
    S_aa = sim[1][-1]
    S_fa = sim[2][-1]
    S_va = sim[3][-1]
    S_bu = sim[4][-1]
    S_pro = sim[5][-1]
    S_ac = sim[6][-1]
    S_h2 = sim[7][-1]
    S_ch4 = sim[8][-1]
    S_IC = sim[9][-1]
    S_IN = sim[10][-1]
    S_I = sim[11][-1]
    X_xc = sim[12][-1]
    X_ch = sim[13][-1]
    X_pr = sim[14][-1]
    X_li = sim[15][-1]
    X_su = sim[16][-1]
    X_aa = sim[17][-1]
    X_fa = sim[18][-1]
    X_c4 = sim[19][-1]
    X_pro = sim[20][-1]
    X_ac = sim[21][-1]
    X_h2 = sim[22][-1]
    X_I = sim[23][-1]
    S_cation = sim[24][-1]
    S_anion = sim[25][-1]
    S_H_ion = sim[26][-1]
    S_va_ion = sim[27][-1]
    S_bu_ion = sim[28][-1]
    S_pro_ion = sim[29][-1]
    S_ac_ion = sim[30][-1]
    S_hco3_ion = sim[31][-1]
    S_co2 = sim[32][-1]
    S_nh3 = sim[33][-1]
    S_nh4_ion = sim[34][-1]
    S_gas_h2 = sim[35][-1]
    S_gas_ch4 = sim[36][-1]
    S_gas_co2 = sim[37][-1]

    DAESolve()

    p_gas_h2 = S_gas_h2 * R * T_ad / 16
    p_gas_ch4 = S_gas_ch4 * R * T_ad / 64
    p_gas_co2 = S_gas_co2 * R * T_ad
    p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
    q_gas = k_p * (p_gas - p_atm)
    if q_gas < 0:
        q_gas = 0

    q_ch4 = q_gas * (p_gas_ch4 / p_gas)
    if q_ch4 < 0:
        q_ch4 = 0

    gasflow = pd.concat([gasflow, pd.DataFrame([{'q_gas': q_gas, 'q_ch4': q_ch4}])], ignore_index=True)

    S_nh4_ion = S_IN - S_nh3
    S_co2 = S_IC - S_hco3_ion
    total_ch4 = total_ch4 + q_ch4

    dt = u - t0

    # 🔥 setpoint 시간 기반 업데이트
    pH_sp = get_pH_sp(u)

    control_timer += dt
    if control_timer >= CONTROL_INTERVAL_DAYS - 1e-12:
        PI_pH_controller(pH_sp, pH, control_timer)
        control_timer = 0.0

    state_zero = [
        S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I,
        X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I,
        S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion,
        S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2
    ]

    simulate_results = pd.concat([simulate_results, pd.DataFrame([state_zero], columns=state_columns)], ignore_index=True)
    control_log = pd.concat([control_log, pd.DataFrame([{'pH': pH, 'pH_sp': pH_sp, 'u_NaOH_kmol_d': u_NaOH, 'q_NaOH_m3_d': q_NaOH, 'u_HCl_kmol_d': u_HCl, 'q_HCl_m3_d': q_HCl}])], ignore_index=True)

    t0 = u

simulate_results.insert(0, 'time', t.values[:len(simulate_results)])
gasflow.insert(0, 'time', t.values[:len(gasflow)])
control_log.insert(0, 'time', t.values[:len(control_log)])

out_state = os.path.join(base_dir, "dynamic_out_mean_full_PI.csv")
out_gas = os.path.join(base_dir, "gasflow_PI.csv")
out_ctl = os.path.join(base_dir, "NaOH_control_log_PI.csv")

simulate_results.to_csv(out_state, index=False)
gasflow.to_csv(out_gas, index=False)
control_log.to_csv(out_ctl, index=False)

print("END")
print(out_state)
print(out_gas)
print(out_ctl)