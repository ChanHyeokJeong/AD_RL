import numpy as np
import scipy.integrate
import copy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path
import time


## unit for each parameter is commented after it is declared (inline)
## if the suggested value for the parameter is different -
## The original default value from the original ADM1 report by Batstone et al (2002), is commented after each unit (inline)

##constant definition from the Rosen et al (2006) BSM2 report
R =  0.083145 #bar.M^-1.K^-1
T_base =  298.15 #K
p_atm =  1.013 #bar
T_op =  308.15 #k ##T_ad #=35 C

##parameter definition from the Rosen et al (2006) BSM2 report bmadm1_report
# Stoichiometric parameter
f_sI_xc =  0.1
f_xI_xc =  0.2
f_ch_xc =  0.2
f_pr_xc =  0.2
f_li_xc =  0.3
N_xc =  0.0376 / 14
N_I =  0.06 / 14 #kmole N.kg^-1COD
N_aa =  0.007 #kmole N.kg^-1COD
C_xc =  0.02786 #kmole C.kg^-1COD
C_sI =  0.03 #kmole C.kg^-1COD
C_ch =  0.0313 #kmole C.kg^-1COD
C_pr =  0.03 #kmole C.kg^-1COD
C_li =  0.022 #kmole C.kg^-1COD
C_xI =  0.03 #kmole C.kg^-1COD
C_su =  0.0313 #kmole C.kg^-1COD
C_aa =  0.03 #kmole C.kg^-1COD
f_fa_li =  0.95
C_fa =  0.0217 #kmole C.kg^-1COD
f_h2_su =  0.19
f_bu_su =  0.13
f_pro_su =  0.27
f_ac_su =  0.41
N_bac =  0.08 / 14 #kmole N.kg^-1COD
C_bu =  0.025 #kmole C.kg^-1COD
C_pro =  0.0268 #kmole C.kg^-1COD
C_ac =  0.0313 #kmole C.kg^-1COD
C_bac =  0.0313 #kmole C.kg^-1COD
Y_su =  0.1
f_h2_aa =  0.06
f_va_aa =  0.23
f_bu_aa =  0.26
f_pro_aa =  0.05
f_ac_aa =  0.40
C_va =  0.024 #kmole C.kg^-1COD
Y_aa =  0.08
Y_fa =  0.06
Y_c4 =  0.06
Y_pro =  0.04
C_ch4 =  0.0156 #kmole C.kg^-1COD
Y_ac =  0.05
Y_h2 =  0.06


# Biochemical parameter values from the Rosen et al (2006) BSM2 report
k_dis =  0.5 #d^-1
k_hyd_ch =  10 #d^-1
k_hyd_pr =  10 #d^-1
k_hyd_li =  10 #d^-1
K_S_IN =  10 ** -4 #M
k_m_su =  30 #d^-1
K_S_su =  0.5 #kgCOD.m^-3
pH_UL_aa =  5.5
pH_LL_aa =  4
k_m_aa =  50 #d^-1
K_S_aa =  0.3 ##kgCOD.m^-3
k_m_fa =  6 #d^-1
K_S_fa =  0.4 #kgCOD.m^-3
K_I_h2_fa =  5 * 10 ** -6 #kgCOD.m^-3
k_m_c4 =  20 #d^-1
K_S_c4 =  0.2 #kgCOD.m^-3
K_I_h2_c4 =  10 ** -5 #kgCOD.m^-3
k_m_pro =  13 #d^-1
K_S_pro =  0.1 #kgCOD.m^-3
K_I_h2_pro =  3.5 * 10 ** -6 #kgCOD.m^-3
k_m_ac =  8 #kgCOD.m^-3
K_S_ac =  0.15 #kgCOD.m^-3
K_I_nh3 =  0.0018 #M
pH_UL_ac =  7
pH_LL_ac =  6
k_m_h2 =  35 #d^-1
K_S_h2 =  7 * 10 ** -6 #kgCOD.m^-3
pH_UL_h2 =  6
pH_LL_h2 =  5
k_dec_X_su =  0.02 #d^-1
k_dec_X_aa =  0.02 #d^-1
k_dec_X_fa =  0.02 #d^-1
k_dec_X_c4 =  0.02 #d^-1
k_dec_X_pro =  0.02 #d^-1
k_dec_X_ac =  0.02 #d^-1
k_dec_X_h2 =  0.02 #d^-1
## M is kmole m^-3

# Physico-chemical parameter values from the Rosen et al (2006) BSM2 report
T_ad =  308.15 #K


K_w =  10 ** -14.0 * np.exp((55900 / (100 * R)) * (1 / T_base - 1 / T_ad)) #M #2.08 * 10 ^ -14

K_a_va =  10 ** -4.86 #M  ADM1 value = 1.38 * 10 ^ -5
K_a_bu =  10 ** -4.82 #M #1.5 * 10 ^ -5
K_a_pro =  10 ** -4.88 #M #1.32 * 10 ^ -5
K_a_ac =  10 ** -4.76 #M #1.74 * 10 ^ -5


K_a_co2 =  10 ** -6.35 * np.exp((7646 / (100 * R)) * (1 / T_base - 1 / T_ad)) #M #4.94 * 10 ^ -7
K_a_IN =  10 ** -9.25 * np.exp((51965 / (100 * R)) * (1 / T_base - 1 / T_ad)) #M #1.11 * 10 ^ -9


k_A_B_va =  10 ** 10 #M^-1 * d^-1
k_A_B_bu =  10 ** 10 #M^-1 * d^-1
k_A_B_pro =  10 ** 10 #M^-1 * d^-1
k_A_B_ac =  10 ** 10 #M^-1 * d^-1
k_A_B_co2 =  10 ** 10 #M^-1 * d^-1
k_A_B_IN =  10 ** 10 #M^-1 * d^-1


p_gas_h2o =  0.0313 * np.exp(5290 * (1 / T_base - 1 / T_ad)) #bar #0.0557
k_p = 5 * 10 ** 4 #m^3.d^-1.bar^-1 #only for BSM2 AD conditions, recalibrate for other AD cases #gas outlet friction
k_L_a =  200.0 #d^-1
K_H_co2 =  0.035 * np.exp((-19410 / (100 * R))* (1 / T_base - 1 / T_ad)) #Mliq.bar^-1 #0.0271
K_H_ch4 =  0.0014 * np.exp((-14240 / (100 * R)) * (1 / T_base - 1 / T_ad)) #Mliq.bar^-1 #0.00116
K_H_h2 =  7.8 * 10 ** -4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T_ad)) #Mliq.bar^-1 #7.38*10^-4

# Physical parameter values used in BSM2 from the Rosen et al (2006) BSM2 report
V_liq =  3400 #m^3
V_gas =  300 #m^3
V_ad = V_liq + V_gas #m^-3

# ADDED BEGIN: Thermal model parameters for reactor-temperature system identification.
# Reason: add reactor temperature as one extra dynamic state while preserving
# the original ADM1 concentration-state equations and list-based state layout.
# Role: define heat-balance constants, open-loop methane heater flow, and
# step-test settings. Heat units are MJ/d so dT/dt is K/d.
# Reference: uploaded textbook photos, examples 19.4-19.6:
#   sludge heating load, digester heat loss, gas use for heating.
output_label = "ode_air145_step50_restore200_lossfactor24"
sludge_heat_capacity = 0.0035 #MJ.kg^-1.K^-1
sludge_density = 1024.5 #kg.m^-3, midpoint of 1.015-1.034 g.cm^-3
# ADDED: lower reactor inventory heat capacity for faster thermal response.
# Reason: replace the water-like liquid heat capacity with a low sludge-like
# wet organic heat capacity converted to a volumetric basis.
# Role: Cv_reactor = Cp_sludge*rho_sludge, so the dT/dt denominator remains
# MJ.K^-1 after multiplying by V_liq.
# Reference: specific-heat tables list water near 4.18 kJ.kg^-1.K^-1 and
# wet organic/tissue material near 3.5 kJ.kg^-1.K^-1; use the lower value as
# a conservative low sludge heat-capacity surrogate.
liquid_heat_capacity = sludge_heat_capacity * sludge_density #MJ.m^-3.K^-1
methane_lhv = 35.881 #MJ.m^-3
heater_efficiency = 0.31
air_temp = 14.5 #C
soil_temp = 21.1 #C
heat_loss_time_factor = 24.0 #h.d^-1 when U is J.m^-2.h^-1.K^-1

# Book example assumptions for sludge heating and digester heat loss.
design_population = 25000
raw_sludge_solids_per_capita = 0.11 #kg.cap^-1.d^-1
raw_sludge_solids_fraction = 0.045
digester_diameter = 22.9 #m
digester_cover_area = np.pi / 4.0 * digester_diameter ** 2
digester_air_wall_height = 3.05 #m
digester_soil_wall_height = 4.57 #m
digester_air_wall_area = np.pi * digester_diameter * digester_air_wall_height
digester_soil_wall_area = np.pi * digester_diameter * digester_soil_wall_height
cover_u = 3268 #J.m^-2.h^-1.K^-1
air_wall_u = 6128 #J.m^-2.h^-1.K^-1
soil_wall_u = 1634 #J.m^-2.h^-1.K^-1
raw_sludge_solids = design_population * raw_sludge_solids_per_capita
book_wet_sludge_mass = raw_sludge_solids / raw_sludge_solids_fraction

# Book example assumptions for available digester gas.
volatile_solids_fraction = 0.70
volatile_solids_destruction = 0.65
biogas_yield = 0.874 #m^3.kg^-1
biogas_methane_fraction = 0.70
produced_biogas = (
    raw_sludge_solids
    * volatile_solids_fraction
    * volatile_solids_destruction
    * biogas_yield
)
produced_methane = produced_biogas * biogas_methane_fraction
step_start = 70.0 #d
step_end = 200.0 #d
step_fraction = 0.50
step_absolute = None #m^3.d^-1, use a number instead of step_fraction if needed
max_sim_days = None #d, use a number for a short smoke test if needed

# ADDED BEGIN: methanogenesis temperature-shock inhibition.
# Reason: fast reactor-temperature changes can reduce methanogenesis even when
# the current temperature is otherwise favorable, because methanogenic biomass
# adapts slowly to a new operating temperature.
# Role: T_adapt is a first-order adaptation state and F_K multiplies only the
# two methanogenesis rates, Rho_11 and Rho_12.
# Reference: Yuki thesis Eq. 4.54-4.55; tau_a=30 d, s_hg=5 K.
use_methanogenesis_temp_shock = True
methanogenesis_adaptation_tau = 30.0 #d
methanogenesis_shock_s_hg = 5.0 #K, F_K=0.5 when abs(T_reactor-T_adapt)=5 K
methanogenesis_shock_sigma = np.sqrt(
  -methanogenesis_shock_s_hg ** 2 / (2.0 * np.log(0.5))
) #K

def methanogenesis_temp_shock_factor(t_reactor, t_adapt):
  if not use_methanogenesis_temp_shock:
    return 1.0
  return np.exp(
    -((t_reactor - t_adapt) ** 2) / (2.0 * methanogenesis_shock_sigma ** 2)
  )
# ADDED END: methanogenesis temperature-shock inhibition.
# ADDED END: Thermal model parameters.

# reading influent and initial condition data from csv files
BASE_DIR = Path(__file__).resolve().parent
influent_state = pd.read_csv(BASE_DIR / "digester_influent.csv")
initial_state = pd.read_csv(BASE_DIR / "digester_initial.csv")

# ADDED: required thermal CSV columns.
# Reason: T_reactor is now initialized from digester_initial.csv and T_in is
# read from digester_influent.csv, so the temperature assumptions live beside
# the original ADM1 input data instead of separate target-temperature constants.
# Role: fail early with a clear message if the appended temperature columns
# are missing.
if 'T_reactor' not in initial_state:
    raise KeyError("digester_initial.csv must include a T_reactor column in degC.")
if 'T_in' not in influent_state:
    raise KeyError("digester_influent.csv must include a T_in column in degC.")
if 'Q' not in influent_state:
    raise KeyError("digester_influent.csv must include a Q column in m3/d.")


# ADDED: convert CSV liquid feed flow to wet sludge mass flow for heat balance.
# Reason: the ADM1 material balance uses q_ad from digester_influent.csv, so
# the thermal inlet load should use the same feed flow instead of an
# independently calculated textbook wet-sludge mass.
# Role: m_wet = Q * rho, with Q in m3/d and rho in kg/m3.
# Reference: uploaded textbook photo, example 19.4 for m*Cp*dT; user-provided
# sludge density range 1.015-1.034 g.cm^-3, represented by its midpoint.
def wet_sludge_mass_flow(q_flow):
    return q_flow * sludge_density

# Function to set influent values for influent state variables at each simulation step
def setInfluent(i):
    global S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, S_ac_in, S_h2_in,S_ch4_in, S_IC_in, S_IN_in, S_I_in,X_xc_in, X_ch_in,X_pr_in,X_li_in,X_su_in,X_aa_in,X_fa_in,X_c4_in,X_pro_in,X_ac_in,X_h2_in,X_I_in,S_cation_in,S_anion_in,q_ad,T_in
    ##variable definition
    # Input values (influent/feed) 
    S_su_in = influent_state['S_su'][i] #kg COD.m^-3
    S_aa_in = influent_state['S_aa'][i] #kg COD.m^-3
    S_fa_in = influent_state['S_fa'][i] #kg COD.m^-3
    S_va_in = influent_state['S_va'][i] #kg COD.m^-3
    S_bu_in = influent_state['S_bu'][i] #kg COD.m^-3
    S_pro_in = influent_state['S_pro'][i] #kg COD.m^-3
    S_ac_in = influent_state['S_ac'][i] #kg COD.m^-3
    S_h2_in = influent_state['S_h2'][i] #kg COD.m^-3
    S_ch4_in = influent_state['S_ch4'][i]  #kg COD.m^-3
    S_IC_in = influent_state['S_IC'][i] #kmole C.m^-3
    S_IN_in = influent_state['S_IN'][i] #kmole N.m^-3
    S_I_in = influent_state['S_I'][i] #kg COD.m^-3
    
    X_xc_in = influent_state['X_xc'][i] #kg COD.m^-3
    X_ch_in = influent_state['X_ch'][i] #kg COD.m^-3
    X_pr_in = influent_state['X_pr'][i] #kg COD.m^-3
    X_li_in = influent_state['X_li'][i] #kg COD.m^-3
    X_su_in = influent_state['X_su'][i] #kg COD.m^-3
    X_aa_in = influent_state['X_aa'][i] #kg COD.m^-3
    X_fa_in = influent_state['X_fa'][i] #kg COD.m^-3
    X_c4_in = influent_state['X_c4'][i] #kg COD.m^-3
    X_pro_in = influent_state['X_pro'][i] #kg COD.m^-3
    X_ac_in = influent_state['X_ac'][i] #kg COD.m^-3
    X_h2_in = influent_state['X_h2'][i] #kg COD.m^-3
    X_I_in = influent_state['X_I'][i] #kg COD.m^-3
    
    S_cation_in = influent_state['S_cation'][i] #kmole.m^-3
    S_anion_in = influent_state['S_anion'][i] #kmole.m^-3
    # ADDED: reactor feed flow from the influent CSV.
    # Reason: q_ad appears in all original ADM1 dilution terms, and using the
    # CSV Q column keeps material balance and heat balance on the same feed flow.
    # Role: Q is read in m3/d and used as q_ad for this simulation step.
    q_ad = influent_state['Q'][i] #m^3.d^-1
    # ADDED: thermal influent temperature for the heat balance.
    # Reason: ADM1 influent concentrations stay unchanged, but sludge
    # temperature needs its own inlet value for Q_sludge = m*Cp*(T_in - T).
    # Role: read T_in from the last CSV column in degC, then convert to K.
    # Reference: uploaded textbook photo, example 19.4.
    T_in = influent_state['T_in'][i] + 273.15


# ADDED: digester heat-loss helper used by the temperature state only.
# Reason: keep the original ADM1 ODE block readable and avoid mixing
# textbook area/U-value calculations into the biochemical equations.
# Role: Q_loss = cover + exposed wall + buried wall, converted J/h to MJ/d.
# Reference: uploaded textbook photo, example 19.5.
def calculate_digester_heat_loss(t_reactor):
    cover_loss = digester_cover_area * cover_u * (t_reactor - (air_temp + 273.15)) * heat_loss_time_factor / 1_000_000.0
    air_wall_loss = digester_air_wall_area * air_wall_u * (t_reactor - (air_temp + 273.15)) * heat_loss_time_factor / 1_000_000.0
    soil_wall_loss = digester_soil_wall_area * soil_wall_u * (t_reactor - (soil_temp + 273.15)) * heat_loss_time_factor / 1_000_000.0
    return {
        "cover_heat_loss": cover_loss,
        "air_wall_heat_loss": air_wall_loss,
        "soil_wall_heat_loss": soil_wall_loss,
        "wall_heat_loss": cover_loss + air_wall_loss + soil_wall_loss,
    }


# ADDED: open-loop methane heater flow at the 35 C operating point.
# Reason: set the baseline MV manually from the steady-state heat balance,
# as requested before applying the methane-flow step for identification.
# Role: q_ch4_base = (Q_loss - Q_sludge) / (LHV_CH4 * eta_heater).
# Reference: uploaded textbook photos, examples 19.4-19.6.
def calculate_open_loop_ch4_flow(t_target, t_in, q_flow):
    feed_heat = wet_sludge_mass_flow(q_flow) * sludge_heat_capacity * (t_in - t_target)
    heat_loss = calculate_digester_heat_loss(t_target)
    required_heat = heat_loss["wall_heat_loss"] - feed_heat
    if required_heat <= 0:
        return 0.0
    return required_heat / (methane_lhv * heater_efficiency)


# ADDED: methane-flow step schedule for system identification.
# Reason: make q_ch4_heater the manipulated variable while leaving ADM1
# gas-production output q_ch4 untouched.
# Role: hold baseline before step_start, apply a relative or absolute step
# between step_start and step_end, then return to baseline.
# Reference: system-identification step-test requirement.
def q_ch4_heater_at(time_d):
    if time_d < step_start:
        return q_ch4_heater_base
    if step_end is not None and time_d >= step_end:
        return q_ch4_heater_base
    if step_absolute is not None:
        return max(0.0, q_ch4_heater_base + step_absolute)
    return max(0.0, q_ch4_heater_base * (1.0 + step_fraction))


# ADDED: methane-use fraction from the PyADM1 methane output.
# Reason: the basic PyADM1 gas algebraic equations already calculate q_ch4,
# so methane use should be compared with that model output instead of only
# the textbook gas-availability estimate.
# Role: f_use = q_ch4_heater / q_ch4_pyadm, with NaN when q_ch4 is unavailable.
# Reference: original PyADM1 gas-flow output q_ch4; requested methane MV use.
def methane_use_fraction_from_pyadm(q_ch4_heater, q_ch4_available):
    if q_ch4_available is None or q_ch4_available <= 0:
        return np.nan
    return q_ch4_heater / q_ch4_available


# ADDED: compact thermal balance terms.
# Reason: reuse identical heat terms in the ODE, output logging, and
# parameter verification without changing original ADM1 equations.
# Role: return Q_sludge, Q_loss, Q_heater, gas-use ratios, and MV value.
# Reference: uploaded textbook photos, examples 19.4-19.6.
def thermal_terms(t_reactor, t_in, time_d, q_flow, q_ch4_available=None):
    q_ch4_heater = q_ch4_heater_at(time_d)
    wet_sludge_mass = wet_sludge_mass_flow(q_flow)
    feed_heat = wet_sludge_mass * sludge_heat_capacity * (t_in - t_reactor)
    heat_loss = calculate_digester_heat_loss(t_reactor)
    heater_heat = q_ch4_heater * methane_lhv * heater_efficiency
    heat_terms = {
        "q_ch4_heater": q_ch4_heater,
        "q_ad": q_flow,
        "wet_sludge_mass": wet_sludge_mass,
        "q_biogas_heater": q_ch4_heater / biogas_methane_fraction,
        "produced_biogas_energy": produced_methane * methane_lhv,
        "methane_use_fraction": methane_use_fraction_from_pyadm(q_ch4_heater, q_ch4_available),
        "feed_heat": feed_heat,
        "heater_heat": heater_heat,
    }
    heat_terms.update(heat_loss)
    return heat_terms
    

# initiate variables (initial values for the reactor state at the initial time (t0)
S_su = initial_state['S_su'][0] #kg COD.m^-3 monosaccharides
S_aa = initial_state['S_aa'][0] #kg COD.m^-3 amino acids
S_fa = initial_state['S_fa'][0] #kg COD.m^-3 total long chain fatty acids 
S_va = initial_state['S_va'][0] #kg COD.m^-3 total valerate
S_bu = initial_state['S_bu'][0] #kg COD.m^-3 total butyrate
S_pro = initial_state['S_pro'][0] #kg COD.m^-3 total propionate
S_ac = initial_state['S_ac'][0] #kg COD.m^-3 total acetate
S_h2 = initial_state['S_h2'][0] #kg COD.m^-3 hydrogen gas 
S_ch4 = initial_state['S_ch4'][0] #kg COD.m^-3 methane gas
S_IC = initial_state['S_IC'][0] #kmole C.m^-3 inorganic carbon
S_IN = initial_state['S_IN'][0] #kmole N.m^-3 inorganic nitrogen
S_I = initial_state['S_I'][0] #kg COD.m^-3 soluble inerts

X_xc = initial_state['X_xc'][0] #kg COD.m^-3 composites
X_ch = initial_state['X_ch'][0] #kg COD.m^-3 carbohydrates
X_pr = initial_state['X_pr'][0] #kg COD.m^-3 proteins
X_li = initial_state['X_li'][0] #kg COD.m^-3 lipids
X_su = initial_state['X_su'][0] #kg COD.m^-3 sugar degraders
X_aa = initial_state['X_aa'][0] #kg COD.m^-3 amino acid degraders 
X_fa = initial_state['X_fa'][0] #kg COD.m^-3 LCFA degraders
X_c4 = initial_state['X_c4'][0] #kg COD.m^-3 valerate and butyrate degraders
X_pro = initial_state['X_pro'][0] #kg COD.m^-3 propionate degraders
X_ac = initial_state['X_ac'][0] #kg COD.m^-3 acetate degraders
X_h2 = initial_state['X_h2'][0] #kg COD.m^-3 hydrogen degraders
X_I = initial_state['X_I'][0] #kg COD.m^-3 particulate inerts

S_cation = initial_state['S_cation'][0] #kmole.m^-3 cations (metallic ions, strong base)
S_anion = initial_state['S_anion'][0] #kmole.m^-3 anions (metallic ions, strong acid)

pH = 7.4655377 # initial pH
S_H_ion = initial_state['S_H_ion'][0] #kmole H.m^-3
S_va_ion = initial_state['S_va_ion'][0] #kg COD.m^-3 valerate
S_bu_ion = initial_state['S_bu_ion'][0] #kg COD.m^-3 butyrate
S_pro_ion = initial_state['S_pro_ion'][0] #kg COD.m^-3 propionate
S_ac_ion = initial_state['S_ac_ion'][0] #kg COD.m^-3 acetate
S_hco3_ion = initial_state['S_hco3_ion'][0] #kmole C.m^-3 bicarbonate
S_nh3 = initial_state['S_nh3'][0] #kmole N.m^-3 ammonia
S_nh4_ion = initial_state['S_nh4_ion'][0] #kmole N.m^-3
S_co2 = initial_state['S_co2'][0] #kmole C.m^-3
S_gas_h2 = initial_state['S_gas_h2'][0] #kg COD.m^-3 hydrogen concentration in gas phase
S_gas_ch4 = initial_state['S_gas_ch4'][0] #kg COD.m^-3 methane concentration in gas phase
S_gas_co2 = initial_state['S_gas_co2'][0]#kmole C.m^-3 carbon dioxide concentration in gas phas

# related to pH inhibition taken from BSM2 report, they are global variables to avoid repeating them in DAE part
K_pH_aa =  (10 ** (-1 * (pH_LL_aa + pH_UL_aa) / 2.0))
nn_aa =  (3.0 / (pH_UL_aa - pH_LL_aa)) #we need a differece between N_aa and n_aa to avoid typos and nn_aa refers to the n_aa in BSM2 report
K_pH_ac = (10 ** (-1 * (pH_LL_ac + pH_UL_ac) / 2.0))
n_ac =  (3.0 / (pH_UL_ac - pH_LL_ac))
K_pH_h2 =  (10 ** (-1 * (pH_LL_h2 + pH_UL_h2) / 2.0))
n_h2 =  (3.0 / (pH_UL_h2 - pH_LL_h2))

#pH equation
pH = - np.log10(S_H_ion)

setInfluent(0) #setting the influent for the initial time (t0) to be ready for the start of the simulation
# ADDED: 39th/40th states and baseline heater MV.
# Reason: keep the original 38 ADM1 states in order and append sludge
# temperature/adaptation temperature at the end instead of renumbering
# existing states.
# Role: T_reactor is integrated by the added heat balance; q_ch4_heater_base
# is the 35 C open-loop methane use computed from textbook heat loads;
# T_adapt follows T_reactor for methanogenesis shock inhibition.
# Reference: original PyADM1 state_zero order; uploaded textbook examples
# 19.4-19.6; Yuki thesis Eq. 4.54-4.55.
T_reactor = initial_state['T_reactor'][0] + 273.15
T_adapt = initial_state['T_adapt'][0] + 273.15 if 'T_adapt' in initial_state else T_reactor
q_ad_initial = q_ad
T_in_initial = T_in
T_reactor_initial = T_reactor
T_adapt_initial = T_adapt
q_ch4_heater_base = calculate_open_loop_ch4_flow(T_reactor_initial, T_in_initial, q_ad_initial)


state_zero = [S_su,
              S_aa,
              S_fa,
              S_va,
              S_bu,
              S_pro,
              S_ac,
              S_h2,
              S_ch4,
              S_IC,
              S_IN,
              S_I,
              X_xc,
              X_ch,
              X_pr,
              X_li,
              X_su,
              X_aa,
              X_fa,
              X_c4,
              X_pro,
              X_ac,
              X_h2,
              X_I,
              S_cation,
              S_anion,
              S_H_ion,
              S_va_ion,
              S_bu_ion,
              S_pro_ion,
              S_ac_ion,
              S_hco3_ion,
              S_co2,
              S_nh3,
              S_nh4_ion,
              S_gas_h2,
              S_gas_ch4,
              S_gas_co2,
              # ADDED: appended thermal state; original 38 ADM1 states above
              # stay in their original order for easier comparison/debugging.
              T_reactor,
              # ADDED: appended methanogen-adaptation temperature state.
              # Reason: represent delayed biological adaptation separately from
              # the instantaneous liquid/reactor temperature.
              # Role: drives F_K inhibition for Rho_11/Rho_12.
              # Reference: Yuki thesis Eq. 4.54-4.55.
              T_adapt]

state_input = [S_su_in,
              S_aa_in,
              S_fa_in,
              S_va_in,
              S_bu_in,
              S_pro_in,
              S_ac_in,
              S_h2_in,
              S_ch4_in,
              S_IC_in,
              S_IN_in,
              S_I_in,
              X_xc_in,
              X_ch_in,
              X_pr_in,
              X_li_in,
              X_su_in,
              X_aa_in,
              X_fa_in,
              X_c4_in,
              X_pro_in,
              X_ac_in,
              X_h2_in,
              X_I_in,
              S_cation_in,
              S_anion_in,
              # ADDED: inlet sludge temperature used only by heat balance.
              T_in]


# Function for calulating the derivatives related to ADM1 system of equations from the Rosen et al (2006) BSM2 report
def ADM1_ODE(t, state_zero):
  global S_nh4_ion, S_co2, p_gas, q_gas, q_ch4
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
  S_cation =  state_zero[24]
  S_anion = state_zero[25]
  S_H_ion =  state_zero[26]
  S_va_ion = state_zero[27]
  S_bu_ion = state_zero[28]
  S_pro_ion = state_zero[29]
  S_ac_ion = state_zero[30]
  S_hco3_ion =  state_zero[31]
  S_co2 = state_zero[32]
  S_nh3 = state_zero[33]
  S_nh4_ion =  state_zero[34]
  S_gas_h2 = state_zero[35]
  S_gas_ch4 = state_zero[36]
  S_gas_co2 = state_zero[37]
  # ADDED: appended temperature state; avoids changing original ADM1 indices.
  T_reactor = state_zero[38]
  # ADDED: appended adaptation-temperature state for methanogenesis shock.
  # Reason: quick temperature changes should inhibit methanogens until the
  # adapted temperature catches up.
  # Role: T_adapt is used only in F_K and has no direct heat-balance term.
  # Reference: Yuki thesis Eq. 4.54-4.55.
  T_adapt = state_zero[39]


  S_su_in = state_input[0]
  S_aa_in = state_input[1]
  S_fa_in = state_input[2]
  S_va_in = state_input[3]
  S_bu_in =  state_input[4]
  S_pro_in =  state_input[5]
  S_ac_in =  state_input[6]
  S_h2_in =   state_input[7]
  S_ch4_in = state_input[8]
  S_IC_in = state_input[9]
  S_IN_in =  state_input[10]
  S_I_in = state_input[11]
  X_xc_in =  state_input[12]
  X_ch_in = state_input[13]
  X_pr_in = state_input[14]
  X_li_in =  state_input[15]
  X_su_in =  state_input[16]
  X_aa_in =  state_input[17]
  X_fa_in =  state_input[18]
  X_c4_in =  state_input[19]
  X_pro_in =  state_input[20]
  X_ac_in =  state_input[21]
  X_h2_in =  state_input[22]
  X_I_in = state_input[23]
  S_cation_in = state_input[24]
  S_anion_in = state_input[25]
  # ADDED: inlet sludge temperature used by diff_T_reactor only.
  T_in = state_input[26]

  S_nh4_ion =  (S_IN - S_nh3)

  S_co2 =  (S_IC - S_hco3_ion)

  I_pH_aa =  ((K_pH_aa ** nn_aa) / (S_H_ion ** nn_aa + K_pH_aa ** nn_aa))
  I_pH_ac =  ((K_pH_ac ** n_ac) / (S_H_ion ** n_ac + K_pH_ac ** n_ac))
  I_pH_h2 =  ((K_pH_h2 ** n_h2) / (S_H_ion ** n_h2 + K_pH_h2 ** n_h2))
  I_IN_lim =  (1 / (1 + (K_S_IN / S_IN)))
  I_h2_fa =  (1 / (1 + (S_h2 / K_I_h2_fa)))
  I_h2_c4 =  (1 / (1 + (S_h2 / K_I_h2_c4)))
  I_h2_pro =  (1 / (1 + (S_h2 / K_I_h2_pro)))
  I_nh3 =  (1 / (1 + (S_nh3 / K_I_nh3)))

  I_5 =  (I_pH_aa * I_IN_lim)
  I_6 = I_5
  I_7 =  (I_pH_aa * I_IN_lim * I_h2_fa)
  I_8 =  (I_pH_aa * I_IN_lim * I_h2_c4)
  I_9 = I_8
  I_10 =  (I_pH_aa * I_IN_lim * I_h2_pro)
  I_11 =  (I_pH_ac * I_IN_lim * I_nh3)
  I_12 =  (I_pH_h2 * I_IN_lim)

 

  # biochemical process rates from Rosen et al (2006) BSM2 report
  Rho_1 =  (k_dis * X_xc)   # Disintegration
  Rho_2 =  (k_hyd_ch * X_ch)  # Hydrolysis of carbohydrates
  Rho_3 =  (k_hyd_pr * X_pr)  # Hydrolysis of proteins
  Rho_4 =  (k_hyd_li * X_li)  # Hydrolysis of lipids
  Rho_5 =  k_m_su * S_su / (K_S_su + S_su) * X_su * I_5  # Uptake of sugars
  Rho_6 =  (k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6)  # Uptake of amino-acids
  Rho_7 =  (k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7)  # Uptake of LCFA (long-chain fatty acids)
  Rho_8 =  (k_m_c4 * (S_va / (K_S_c4 + S_va )) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8)  # Uptake of valerate
  Rho_9 =  (k_m_c4 * (S_bu / (K_S_c4 + S_bu )) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9)  # Uptake of butyrate
  Rho_10 =  (k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10)  # Uptake of propionate
  # ADDED: short-term temperature-shock inhibition for methanogenesis.
  # Reason: acetoclastic and hydrogenotrophic methanogens are sensitive to
  # quick mismatch between current liquid temperature and adapted temperature.
  # Role: F_K multiplies only Rho_11/Rho_12; all other ADM1 rates stay in the
  # original PyADM1 form.
  # Reference: Yuki thesis Eq. 4.54-4.55.
  F_K = methanogenesis_temp_shock_factor(T_reactor, T_adapt)
  Rho_11 =  (k_m_ac * (S_ac / (K_S_ac + S_ac)) * X_ac * I_11 * F_K)  # Uptake of acetate
  Rho_12 =  (k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12 * F_K)  # Uptake of hydrogen
  Rho_13 =  (k_dec_X_su * X_su)  # Decay of X_su
  Rho_14 =  (k_dec_X_aa * X_aa)  # Decay of X_aa
  Rho_15 =  (k_dec_X_fa * X_fa)  # Decay of X_fa
  Rho_16 =  (k_dec_X_c4 * X_c4)  # Decay of X_c4
  Rho_17 =  (k_dec_X_pro * X_pro)  # Decay of X_pro
  Rho_18 =  (k_dec_X_ac * X_ac)  # Decay of X_ac
  Rho_19 =  (k_dec_X_h2 * X_h2)  # Decay of X_h2

  # acid-base rates for the BSM2 ODE implementation from Rosen et al (2006) BSM2 report
  Rho_A_4 =  (k_A_B_va * (S_va_ion * (K_a_va + S_H_ion) - K_a_va * S_va))
  Rho_A_5 =  (k_A_B_bu * (S_bu_ion * (K_a_bu + S_H_ion) - K_a_bu * S_bu))
  Rho_A_6 =  (k_A_B_pro * (S_pro_ion * (K_a_pro + S_H_ion) - K_a_pro * S_pro))
  Rho_A_7 =  (k_A_B_ac * (S_ac_ion * (K_a_ac + S_H_ion) - K_a_ac * S_ac))
  Rho_A_10 =  (k_A_B_co2 * (S_hco3_ion * (K_a_co2 + S_H_ion) - K_a_co2 * S_IC))
  Rho_A_11 =  (k_A_B_IN * (S_nh3 * (K_a_IN + S_H_ion) - K_a_IN * S_IN))

  # gas phase algebraic equations from Rosen et al (2006) BSM2 report
  # ADDED: use the dynamic reactor temperature in gas partial pressures.
  # Reason: original PyADM1 used constant T_op; adding T_reactor as a state
  # requires gas-law calculations to follow that operating temperature.
  # Role: preserve original gas equations except replacing T_op with T_reactor.
  # Reference: original PyADM1 gas phase algebraic equations; ADM1/BSM2 gas law.
  p_gas_h2 =  (S_gas_h2 * R * T_reactor / 16)
  p_gas_ch4 =  (S_gas_ch4 * R * T_reactor / 64)
  p_gas_co2 =  (S_gas_co2 * R * T_reactor)


  p_gas=  (p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o)
  q_gas =  (k_p * (p_gas- p_atm))
  if q_gas < 0:    q_gas = 0

  q_ch4 = q_gas * (p_gas_ch4/p_gas) # methane flow

  # gas transfer rates from Rosen et al (2006) BSM2 report
  Rho_T_8 =  (k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2))
  Rho_T_9 =  (k_L_a * (S_ch4 - 64 * K_H_ch4 * p_gas_ch4))
  Rho_T_10 =  (k_L_a * (S_co2 - K_H_co2 * p_gas_co2))

  ##differential equaitons from Rosen et al (2006) BSM2 report
  # differential equations 1 to 12 (soluble matter)
  diff_S_su = q_ad / V_liq * (S_su_in - S_su) + Rho_2 + (1 - f_fa_li) * Rho_4 - Rho_5  # eq1

  diff_S_aa = q_ad / V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6  # eq2

  diff_S_fa = q_ad / V_liq * (S_fa_in - S_fa) + (f_fa_li * Rho_4) - Rho_7  # eq3

  diff_S_va = q_ad / V_liq * (S_va_in - S_va) + (1 - Y_aa) * f_va_aa * Rho_6 - Rho_8  # eq4

  diff_S_bu = q_ad / V_liq * (S_bu_in - S_bu) + (1 - Y_su) * f_bu_su * Rho_5 + (1 - Y_aa) * f_bu_aa * Rho_6 - Rho_9  # eq5

  diff_S_pro = q_ad / V_liq * (S_pro_in - S_pro) + (1 - Y_su) * f_pro_su * Rho_5 + (1 - Y_aa) * f_pro_aa * Rho_6 + (1 - Y_c4) * 0.54 * Rho_8 - Rho_10  # eq6

  diff_S_ac = q_ad / V_liq * (S_ac_in - S_ac) + (1 - Y_su) * f_ac_su * Rho_5 + (1 - Y_aa) * f_ac_aa * Rho_6 + (1 - Y_fa) * 0.7 * Rho_7 + (1 - Y_c4) * 0.31 * Rho_8 + (1 - Y_c4) * 0.8 * Rho_9 + (1 - Y_pro) * 0.57 * Rho_10 - Rho_11  # eq7

  #diff_S_h2 is defined with DAE paralel equaitons

  diff_S_ch4 = q_ad / V_liq * (S_ch4_in - S_ch4) + (1 - Y_ac) * Rho_11 + (1 - Y_h2) * Rho_12 - Rho_T_9  # eq9


  ## eq10 start##
  s_1 =  (-1 * C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI) 
  s_2 =  (-1 * C_ch + C_su)
  s_3 =  (-1 * C_pr + C_aa)
  s_4 =  (-1 * C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa)
  s_5 =  (-1 * C_su + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac)
  s_6 =  (-1 * C_aa + (1 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac)
  s_7 =  (-1 * C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac)
  s_8 =  (-1 * C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac)
  s_9 =  (-1 * C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac)
  s_10 =  (-1 * C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac)
  s_11 =  (-1 * C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac)
  s_12 =  ((1 - Y_h2) * C_ch4 + Y_h2 * C_bac)
  s_13 =  (-1 * C_bac + C_xc) 

  Sigma =  (s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 + s_5 * Rho_5 + s_6 * Rho_6 + s_7 * Rho_7 + s_8 * Rho_8 + s_9 * Rho_9 + s_10 * Rho_10 + s_11 * Rho_11 + s_12 * Rho_12 + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19))

  diff_S_IC = q_ad / V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10
  ## eq10 end##


 
  diff_S_IN = q_ad / V_liq * (S_IN_in - S_IN) + (N_xc - f_xI_xc * N_I - f_sI_xc * N_I-f_pr_xc * N_aa) * Rho_1 - Y_su * N_bac * Rho_5 + (N_aa - Y_aa * N_bac) * Rho_6 - Y_fa * N_bac * Rho_7 - Y_c4 * N_bac * Rho_8 - Y_c4 * N_bac * Rho_9 - Y_pro * N_bac * Rho_10 - Y_ac * N_bac * Rho_11 - Y_h2 * N_bac * Rho_12 + (N_bac - N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19) # eq11 


  diff_S_I = q_ad / V_liq * (S_I_in - S_I) + f_sI_xc * Rho_1  # eq12


  # Differential equations 13 to 24 (particulate matter)
  diff_X_xc = q_ad / V_liq * (X_xc_in - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19  # eq13 

  diff_X_ch = q_ad / V_liq * (X_ch_in - X_ch) + f_ch_xc * Rho_1 - Rho_2  # eq14 

  diff_X_pr = q_ad / V_liq * (X_pr_in - X_pr) + f_pr_xc * Rho_1 - Rho_3  # eq15 

  diff_X_li = q_ad / V_liq * (X_li_in - X_li) + f_li_xc * Rho_1 - Rho_4  # eq16 

  diff_X_su = q_ad / V_liq * (X_su_in - X_su) + Y_su * Rho_5 - Rho_13  # eq17

  diff_X_aa = q_ad / V_liq * (X_aa_in - X_aa) + Y_aa * Rho_6 - Rho_14  # eq18

  diff_X_fa = q_ad / V_liq * (X_fa_in - X_fa) + Y_fa * Rho_7 - Rho_15  # eq19

  diff_X_c4 = q_ad / V_liq * (X_c4_in - X_c4) + Y_c4 * Rho_8 + Y_c4 * Rho_9 - Rho_16  # eq20

  diff_X_pro = q_ad / V_liq * (X_pro_in - X_pro) + Y_pro * Rho_10 - Rho_17  # eq21

  diff_X_ac = q_ad / V_liq * (X_ac_in - X_ac) + Y_ac * Rho_11 - Rho_18  # eq22

  diff_X_h2 = q_ad / V_liq * (X_h2_in - X_h2) + Y_h2 * Rho_12 - Rho_19  # eq23

  diff_X_I = q_ad / V_liq * (X_I_in - X_I) + f_xI_xc * Rho_1  # eq24 

  # Differential equations 25 and 26 (cations and anions)
  diff_S_cation = q_ad / V_liq * (S_cation_in - S_cation)  # eq25

  diff_S_anion = q_ad / V_liq * (S_anion_in - S_anion)  # eq26

  diff_S_h2 = 0

  # Differential equations 27 to 32 (ion states, only for ODE implementation)
  diff_S_va_ion = 0  # eq27

  diff_S_bu_ion = 0  # eq28

  diff_S_pro_ion = 0  # eq29

  diff_S_ac_ion = 0  # eq30

  diff_S_hco3_ion = 0  # eq31

  diff_S_nh3 = 0  # eq32

  # Gas phase equations: Differential equations 33 to 35
  diff_S_gas_h2 = (q_gas / V_gas * -1 * S_gas_h2) + (Rho_T_8 * V_liq / V_gas)  # eq33

  diff_S_gas_ch4 = (q_gas / V_gas * -1 * S_gas_ch4) + (Rho_T_9 * V_liq / V_gas)  # eq34

  diff_S_gas_co2 = (q_gas / V_gas * -1 * S_gas_co2) + (Rho_T_10 * V_liq / V_gas)  # eq35

  diff_S_H_ion = 0
  diff_S_co2 = 0
  diff_S_nh4_ion = 0 #to keep the output same length as input for ADM1_ODE funcion
  # ADDED: differential equation for the appended temperature state.
  # Reason: extend ADM1 with reactor-temperature dynamics without changing
  # the original biological/chemical differential equations above.
  # Role: dT/dt = (Q_sludge - Q_loss + Q_heater)/(rhoCp*V_liq).
  # Reference: uploaded textbook photos, examples 19.4-19.6.
  heat = thermal_terms(T_reactor, T_in, t, q_ad)
  diff_T_reactor = (
    heat["feed_heat"] - heat["wall_heat_loss"] + heat["heater_heat"]
  ) / (liquid_heat_capacity * V_liq)
  diff_T_adapt = (T_reactor - T_adapt) / methanogenesis_adaptation_tau

  # ADDED: append diff_T_reactor and diff_T_adapt after original 38 ADM1 derivatives.
  return diff_S_su, diff_S_aa, diff_S_fa, diff_S_va, diff_S_bu, diff_S_pro, diff_S_ac, diff_S_h2, diff_S_ch4, diff_S_IC, diff_S_IN, diff_S_I, diff_X_xc, diff_X_ch, diff_X_pr, diff_X_li, diff_X_su, diff_X_aa, diff_X_fa, diff_X_c4, diff_X_pro, diff_X_ac, diff_X_h2, diff_X_I, diff_S_cation, diff_S_anion, diff_S_H_ion, diff_S_va_ion,  diff_S_bu_ion, diff_S_pro_ion, diff_S_ac_ion, diff_S_hco3_ion, diff_S_co2,  diff_S_nh3, diff_S_nh4_ion, diff_S_gas_h2, diff_S_gas_ch4, diff_S_gas_co2, diff_T_reactor, diff_T_adapt

# Function for integration of ADM1 differential equations 
def simulate(t_step, solvermethod):
  r = scipy.integrate.solve_ivp(ADM1_ODE, t_step, state_zero, method= solvermethod)
  return r.y

# Function for DAE equations adopted from the Rosen et al (2006) BSM2 report bmadm1_report
def DAESolve():
  global S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_nh3, S_H_ion, pH, p_gas_h2, S_h2, S_nh4_ion, S_co2, P_gas, q_gas
  
  ##  DAE calculations 
  eps = 0.0000001
  prevS_H_ion = S_H_ion
  
  #initial values for Newton-Raphson solver parameter
  shdelta = 1.0
  shgradeq = 1.0
  S_h2delta = 1.0
  S_h2gradeq = 1.0
  tol = 10 ** (-12) #solver accuracy tolerance
  maxIter = 1000 #maximum number of iterations for solver
  i = 1
  j = 1
  
  ## DAE solver for S_H_ion from Rosen et al. (2006)
  while ((shdelta > tol or shdelta < -tol) and (i <= maxIter)):
    S_va_ion = K_a_va * S_va / (K_a_va + S_H_ion)
    S_bu_ion = K_a_bu * S_bu / (K_a_bu + S_H_ion)
    S_pro_ion = K_a_pro * S_pro / (K_a_pro + S_H_ion)
    S_ac_ion = K_a_ac * S_ac / (K_a_ac + S_H_ion)
    S_hco3_ion = K_a_co2 * S_IC / (K_a_co2 + S_H_ion)
    S_nh3 = K_a_IN * S_IN / (K_a_IN + S_H_ion)
    shdelta = S_cation + (S_IN - S_nh3) + S_H_ion - S_hco3_ion - S_ac_ion / 64.0 - S_pro_ion / 112.0 - S_bu_ion / 160.0 - S_va_ion / 208.0 - K_w / S_H_ion - S_anion
    shgradeq = 1 + K_a_IN * S_IN / ((K_a_IN + S_H_ion) * (K_a_IN + S_H_ion)) + K_a_co2 * S_IC / ((K_a_co2 + S_H_ion) * (K_a_co2 + S_H_ion))               + 1 / 64.0 * K_a_ac * S_ac / ((K_a_ac + S_H_ion) * (K_a_ac + S_H_ion))               + 1 / 112.0 * K_a_pro * S_pro / ((K_a_pro + S_H_ion) * (K_a_pro + S_H_ion))               + 1 / 160.0 * K_a_bu * S_bu / ((K_a_bu + S_H_ion) * (K_a_bu + S_H_ion))               + 1 / 208.0 * K_a_va * S_va / ((K_a_va + S_H_ion) * (K_a_va + S_H_ion))               + K_w / (S_H_ion * S_H_ion)
    S_H_ion = S_H_ion - shdelta / shgradeq
    if S_H_ion <= 0:
        S_H_ion = tol
    i+=1
  
  # pH calculation
  pH = - np.log10(S_H_ion)
  
  #DAE solver for S_h2 from Rosen et al. (2006) 
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
    # ADDED: keep DAE hydrogen equation consistent with ODE methanogenesis shock.
    # Reason: Rho_12 appears in the algebraic S_h2 residual and derivative.
    # Role: reduce hydrogenotrophic methanogenesis during rapid temperature
    # mismatch in the same way as ADM1_ODE.
    # Reference: Yuki thesis Eq. 4.54-4.55.
    F_K = methanogenesis_temp_shock_factor(T_reactor, T_adapt)
    Rho_5 = k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5  # Uptake of sugars
    Rho_6 = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6  # Uptake of amino-acids
    Rho_7 = k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7  # Uptake of LCFA (long-chain fatty acids)
    Rho_8 = k_m_c4 * (S_va / (K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va+ 1e-6)) * I_8  # Uptake of valerate
    Rho_9 = k_m_c4 * (S_bu / (K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va+ 1e-6)) * I_9  # Uptake of butyrate
    Rho_10 = k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10  # Uptake of propionate
    Rho_12 = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12 * F_K  # Uptake of hydrogen
    p_gas_h2 = S_gas_h2 * R * T_ad / 16
    Rho_T_8 = k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2)
    S_h2delta = q_ad / V_liq * (S_h2_in - S_h2) + (1 - Y_su) * f_h2_su * Rho_5 + (1 - Y_aa) * f_h2_aa * Rho_6 + (1 - Y_fa) * 0.3 * Rho_7 + (1 - Y_c4) * 0.15 * Rho_8 + (1 - Y_c4) * 0.2 * Rho_9 + (1 - Y_pro) * 0.43 * Rho_10 - Rho_12 - Rho_T_8
    S_h2gradeq = - 1.0 / V_liq * q_ad - 3.0 / 10.0 * (1 - Y_fa) * k_m_fa * S_fa / (K_S_fa + S_fa) * X_fa * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_fa) * (1 + S_h2 / K_I_h2_fa)) / K_I_h2_fa - 3.0 / 20.0 * (1 - Y_c4) * k_m_c4 * S_va * S_va / (K_S_c4 + S_va) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4 ) * (1 + S_h2 / K_I_h2_c4 )) / K_I_h2_c4 - 1.0 / 5.0 * (1 - Y_c4) * k_m_c4 * S_bu * S_bu / (K_S_c4 + S_bu) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4 ) * (1 + S_h2 / K_I_h2_c4 )) / K_I_h2_c4 - 43.0 / 100.0 * (1 - Y_pro) * k_m_pro * S_pro / (K_S_pro + S_pro) * X_pro * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_pro ) * (1 + S_h2 / K_I_h2_pro )) / K_I_h2_pro - F_K * k_m_h2 / (K_S_h2 + S_h2) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) + F_K * k_m_h2 * S_h2 / ((K_S_h2 + S_h2) * (K_S_h2 + S_h2)) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) - k_L_a
    S_h2 = S_h2 - S_h2delta / S_h2gradeq
    if S_h2 <= 0:
        S_h2 = tol
    j+=1

## time array definition
t = influent_state['time']
# ADDED: optional short-run switch for smoke tests only.
# Reason: allow quick verification of the appended thermal state without
# changing the source CSV or the original dynamic-simulation loop.
# Role: truncate the existing time vector when max_sim_days is set above.
# Reference: verification convenience; not part of original ADM1 equations.
if max_sim_days is not None:
  t = t[t <= max_sim_days]


# Initiate the cache data frame for storing simulation results
simulate_results = pd.DataFrame([state_zero])
# ADDED: T_reactor_K is appended to the original output columns.
# Reason: preserve original output order and expose the new thermal/shock states.
# Role: keep all original output columns in place, with T_reactor_K/T_adapt_K last.
# Reference: original PyADM1 simulate_results column order; Yuki thesis Eq. 4.55.
columns = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC", "S_IN", "S_I", "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I", "S_cation", "S_anion", "pH", "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion", "S_hco3_ion", "S_co2", "S_nh3", "S_nh4_ion", "S_gas_h2", "S_gas_ch4", "S_gas_co2", "T_reactor_K", "T_adapt_K"]
simulate_results.columns = columns

# Setting the solver method for the simulate function
solvermethod = 'DOP853'
t0=0
n=0
total_steps = max(len(t) - 1, 0)
progress_interval = max(1, total_steps // 20) if total_steps else 1
progress_start = time.perf_counter()
print(f"Starting ADM1 thermal simulation ({output_label}): {total_steps} steps", flush=True)
print(
  f"Open-loop q_ch4_heater={q_ch4_heater_base:.3f} m3/d | "
  f"step at {step_start:g} d, end={step_end:g} d, fraction={step_fraction:g}",
  flush=True
)

# Initiate cache data frame for storing gasflow values
initial_heat = thermal_terms(T_reactor_initial, T_in_initial, 0.0, q_ad_initial)
# ADDED: thermal diagnostic log.
# Reason: keep q_ch4 (ADM1 methane production) separate from q_ch4_heater
# (heater methane consumption MV), and expose textbook heat-balance terms.
# Role: write a side CSV with MV, gas-use fraction, and heat-load components.
# Reference: uploaded textbook photos, examples 19.4-19.6.
initflow = {
  'time': [0.0],
  'q_gas': [0.0],
  'q_ch4': [0.0],
  'q_ch4_heater': [initial_heat["q_ch4_heater"]],
  'q_biogas_heater': [initial_heat["q_biogas_heater"]],
  'methane_use_fraction': [initial_heat["methane_use_fraction"]],
  'q_ad': [initial_heat["q_ad"]],
  'wet_sludge_mass': [initial_heat["wet_sludge_mass"]],
  'T_in_C': [T_in - 273.15],
  'T_reactor_C': [T_reactor - 273.15],
  'T_adapt_C': [T_adapt - 273.15],
  'methanogenesis_shock_factor': [methanogenesis_temp_shock_factor(T_reactor, T_adapt)],
  'methanogenesis_temp_mismatch_K': [T_reactor - T_adapt],
  'feed_heat_MJ_d': [initial_heat["feed_heat"]],
  'cover_heat_loss_MJ_d': [initial_heat["cover_heat_loss"]],
  'air_wall_heat_loss_MJ_d': [initial_heat["air_wall_heat_loss"]],
  'soil_wall_heat_loss_MJ_d': [initial_heat["soil_wall_heat_loss"]],
  'wall_heat_loss_MJ_d': [initial_heat["wall_heat_loss"]],
  'heater_heat_MJ_d': [initial_heat["heater_heat"]],
}
gasflow = pd.DataFrame(initflow)
total_ch4 = 0

# Initiate cache data frame for storing feedflow values
initq = {'q_ad' : [q_ad_initial]}
feedflow = pd.DataFrame(initq)


## Dynamic simulation
# Loop for simlating at each time step and feeding the results to the next time step
for u in t[1:]:
  n+=1
  setInfluent(n)
  
  # ADDED: append T_in after the original influent state_input values.
  # Reason: original ADM1 influent vector is unchanged; T_in is used only by
  # the added heat-balance equation.
  # Role: feed the inlet sludge temperature into ADM1_ODE as state_input[26].
  # Reference: uploaded textbook photo, example 19.4.
  state_input = [S_su_in,S_aa_in,S_fa_in,S_va_in,S_bu_in,S_pro_in,S_ac_in,S_h2_in,S_ch4_in,S_IC_in,S_IN_in,S_I_in,X_xc_in,X_ch_in,X_pr_in,X_li_in,X_su_in,X_aa_in,X_fa_in,X_c4_in,X_pro_in,X_ac_in,X_h2_in,X_I_in,S_cation_in,S_anion_in,T_in]
  
  # Span for next time step
  tstep = [t0,u]

  # Solve and store ODE results for next step
  # ADDED: sim_T_reactor is appended after the original 38 result arrays.
  # Reason: solve_ivp returns one row per state; the added state needs one
  # extra result array without changing existing result-array order.
  # Role: receive integrated T_reactor for this time step.
  # Reference: original PyADM1 simulate() return shape plus appended state.
  sim_S_su, sim_S_aa, sim_S_fa, sim_S_va, sim_S_bu, sim_S_pro, sim_S_ac, sim_S_h2, sim_S_ch4, sim_S_IC, sim_S_IN, sim_S_I, sim_X_xc, sim_X_ch, sim_X_pr, sim_X_li, sim_X_su, sim_X_aa, sim_X_fa, sim_X_c4, sim_X_pro, sim_X_ac, sim_X_h2, sim_X_I, sim_S_cation, sim_S_anion, sim_S_H_ion, sim_S_va_ion, sim_S_bu_ion, sim_S_pro_ion, sim_S_ac_ion, sim_S_hco3_ion, sim_S_co2, sim_S_nh3, sim_S_nh4_ion, sim_S_gas_h2, sim_S_gas_ch4, sim_S_gas_co2, sim_T_reactor, sim_T_adapt = simulate(tstep, solvermethod)

  # Store ODE simulation result states
  # ADDED: T_reactor is unpacked after original ADM1 states.
  # Reason: carry the new thermal state forward while leaving ADM1 assignment order intact.
  # Role: update T_reactor with the last integrated value for this step.
  # Reference: original PyADM1 state transfer pattern.
  S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2, T_reactor, T_adapt =   sim_S_su[-1], sim_S_aa[-1], sim_S_fa[-1], sim_S_va[-1], sim_S_bu[-1], sim_S_pro[-1], sim_S_ac[-1], sim_S_h2[-1], sim_S_ch4[-1], sim_S_IC[-1], sim_S_IN[-1], sim_S_I[-1], sim_X_xc[-1], sim_X_ch[-1], sim_X_pr[-1], sim_X_li[-1], sim_X_su[-1], sim_X_aa[-1], sim_X_fa[-1], sim_X_c4[-1], sim_X_pro[-1], sim_X_ac[-1], sim_X_h2[-1], sim_X_I[-1], sim_S_cation[-1], sim_S_anion[-1], sim_S_H_ion[-1], sim_S_va_ion[-1], sim_S_bu_ion[-1], sim_S_pro_ion[-1], sim_S_ac_ion[-1], sim_S_hco3_ion[-1], sim_S_co2[-1], sim_S_nh3[-1], sim_S_nh4_ion[-1], sim_S_gas_h2[-1], sim_S_gas_ch4[-1], sim_S_gas_co2[-1], sim_T_reactor[-1], sim_T_adapt[-1]
  # ADDED: keep legacy T_op/T_ad variables synchronized for downstream
  # algebraic expressions that still reference those original names.
  # Reason: minimize edits to original variable names.
  # Role: make later gas/DAE code see the current temperature.
  # Reference: original PyADM1 constant T_op/T_ad usage.
  T_op = T_reactor
  T_ad = T_reactor
  
  # Solve DAE states
  DAESolve()

  # Algebraic equations
  # ADDED: use dynamic T_reactor instead of constant T_op in gas-law terms.
  # Reason: gas pressure calculations depend on operating temperature.
  # Role: preserve original gas algebraic form with a dynamic temperature.
  # Reference: original PyADM1 gas equations and ADM1/BSM2 gas law.
  p_gas_h2 =  (S_gas_h2 * R * T_reactor / 16)
  p_gas_ch4 =  (S_gas_ch4 * R * T_reactor / 64)
  p_gas_co2 =  (S_gas_co2 * R * T_reactor)
  p_gas=  (p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o)
  q_gas =  (k_p * (p_gas- p_atm))
  if q_gas < 0:    
    q_gas = 0
  
  q_ch4 = q_gas * (p_gas_ch4/p_gas) # methane flow
  if q_ch4 < 0:
    q_ch4 = 0

  # ADDED: log heat-balance terms and heater MV for identification.
  # Reason: step response needs the actual MV and heat components at each time.
  # Role: populate thermal_inputs_<label>.csv.
  # Reference: uploaded textbook photos, examples 19.4-19.6.
  heat = thermal_terms(T_reactor, T_in, u, q_ad, q_ch4)
  flowtemp = {
    'time': float(u),
    'q_gas' : q_gas,
    'q_ch4' : q_ch4,
    'q_ch4_heater' : heat["q_ch4_heater"],
    'q_biogas_heater' : heat["q_biogas_heater"],
    'methane_use_fraction' : heat["methane_use_fraction"],
    'q_ad': heat["q_ad"],
    'wet_sludge_mass': heat["wet_sludge_mass"],
    'T_in_C': T_in - 273.15,
    'T_reactor_C': T_reactor - 273.15,
    'T_adapt_C': T_adapt - 273.15,
    'methanogenesis_shock_factor': methanogenesis_temp_shock_factor(T_reactor, T_adapt),
    'methanogenesis_temp_mismatch_K': T_reactor - T_adapt,
    'feed_heat_MJ_d': heat["feed_heat"],
    'cover_heat_loss_MJ_d': heat["cover_heat_loss"],
    'air_wall_heat_loss_MJ_d': heat["air_wall_heat_loss"],
    'soil_wall_heat_loss_MJ_d': heat["soil_wall_heat_loss"],
    'wall_heat_loss_MJ_d': heat["wall_heat_loss"],
    'heater_heat_MJ_d': heat["heater_heat"],
  }
  gasflow = pd.concat([gasflow, pd.DataFrame([flowtemp])], ignore_index=True)

  S_nh4_ion =  (S_IN - S_nh3)
  S_co2 =  (S_IC - S_hco3_ion)
  total_ch4 = total_ch4 + q_ch4 


  #state transfer
  # ADDED: append T_reactor/T_adapt after original 38 ADM1 states for next step.
  # Reason: the next solve_ivp call must start from the updated thermal and
  # adaptation temperatures.
  # Role: keep state_zero length aligned with ADM1_ODE return length.
  # Reference: original PyADM1 state transfer pattern; Yuki thesis Eq. 4.55.
  state_zero = [S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2, T_reactor, T_adapt]
  
  dfstate_zero = pd.DataFrame([state_zero], columns = columns)
  simulate_results = pd.concat([simulate_results, dfstate_zero], ignore_index=True)
  t0 = u

  if n == 1 or n % progress_interval == 0 or n == total_steps:
    elapsed = time.perf_counter() - progress_start
    pct = (100.0 * n / total_steps) if total_steps else 100.0
    pH_now = -np.log10(max(S_H_ion, 1e-30))
    print(
      f"Progress {n}/{total_steps} ({pct:5.1f}%) | t={float(u):.3f} d | "
      f"T={T_reactor - 273.15:.3f} C | pH={pH_now:.3f} | "
      f"q_ch4={q_ch4:.3f} | q_ch4_heater={heat['q_ch4_heater']:.3f} | elapsed={elapsed:.1f}s",
      flush=True
    )
      

# Write the dynamic simulation resutls to csv
phlogarray = -1 * np.log10(simulate_results['pH'])
simulate_results['pH'] = phlogarray
# ADDED: include time and Celsius temperature in the CSV output.
# Reason: original output can be compared by columns, while step-response
# identification can read time/T directly from the generated file.
# Role: add convenience columns without modifying integrated state values.
# Reference: step-response identification output requirement.
simulate_results.insert(0, "time", list(t)[:len(simulate_results)])
simulate_results["T_reactor_C"] = simulate_results["T_reactor_K"] - 273.15
simulate_results["T_adapt_C"] = simulate_results["T_adapt_K"] - 273.15
simulate_results["methanogenesis_shock_factor"] = [
  methanogenesis_temp_shock_factor(t_reactor, t_adapt)
  for t_reactor, t_adapt in zip(simulate_results["T_reactor_K"], simulate_results["T_adapt_K"])
]
simulate_results["methanogenesis_temp_mismatch_K"] = (
  simulate_results["T_reactor_K"] - simulate_results["T_adapt_K"]
)
output_path = BASE_DIR / f"dynamic_out_thermal_{output_label}.csv"
simulate_results.to_csv(output_path, index = False)
thermal_input_path = BASE_DIR / f"thermal_inputs_{output_label}.csv"
gasflow.to_csv(thermal_input_path, index=False)
parameter_path = BASE_DIR / f"thermal_parameters_{output_label}.csv"
# ADDED: save thermal assumptions and initial heat-balance check.
# Reason: make the textbook assumptions traceable and verify values such as
# sludge heating load, digester heat loss, and methane-use fraction.
# Reference: uploaded textbook photos, examples 19.4-19.6.
initial_heat_check = thermal_terms(T_reactor_initial, T_in_initial, 0.0, q_ad_initial)
baseline_gas = gasflow[(gasflow["time"] < step_start) & (gasflow["q_ch4"] > 0)]
pyadm_q_ch4_base = baseline_gas["q_ch4"].mean() if len(baseline_gas) else np.nan
pd.DataFrame([{
  "initial_T_reactor_C": T_reactor_initial - 273.15,
  "initial_T_adapt_C": T_adapt_initial - 273.15,
  "initial_T_in_C": T_in_initial - 273.15,
  "air_temp_C": air_temp,
  "soil_temp_C": soil_temp,
  "heat_loss_time_factor": heat_loss_time_factor,
  "reactor_vol_heat_capacity_MJ_m3_K": liquid_heat_capacity,
  "sludge_specific_heat_MJ_kg_K": sludge_heat_capacity,
  "methane_LHV_MJ_m3": methane_lhv,
  "heater_thermal_efficiency": heater_efficiency,
  "use_methanogenesis_temp_shock": use_methanogenesis_temp_shock,
  "methanogenesis_adaptation_tau_d": methanogenesis_adaptation_tau,
  "methanogenesis_shock_s_hg_K": methanogenesis_shock_s_hg,
  "methanogenesis_shock_sigma_K": methanogenesis_shock_sigma,
  "design_population_cap": design_population,
  "raw_sludge_solids_kg_cap_d": raw_sludge_solids_per_capita,
  "raw_sludge_solids_fraction": raw_sludge_solids_fraction,
  "raw_sludge_solids_kg_d": raw_sludge_solids,
  "sludge_density_kg_m3": sludge_density,
  "book_wet_sludge_mass_kg_d": book_wet_sludge_mass,
  "initial_wet_sludge_mass_kg_d": wet_sludge_mass_flow(q_ad_initial),
  "final_wet_sludge_mass_kg_d": wet_sludge_mass_flow(q_ad),
  "digester_diameter_m": digester_diameter,
  "digester_cover_area_m2": digester_cover_area,
  "digester_air_wall_area_m2": digester_air_wall_area,
  "digester_soil_wall_area_m2": digester_soil_wall_area,
  "cover_U_J_m2_h_K": cover_u,
  "air_wall_U_J_m2_h_K": air_wall_u,
  "soil_wall_U_J_m2_h_K": soil_wall_u,
  "volatile_solids_fraction": volatile_solids_fraction,
  "volatile_solids_destruction": volatile_solids_destruction,
  "biogas_yield_m3_kg_VS_destroyed": biogas_yield,
  "biogas_methane_fraction": biogas_methane_fraction,
  "produced_biogas_m3_d": produced_biogas,
  "produced_methane_m3_d": produced_methane,
  "initial_feed_heat_MJ_d": initial_heat_check["feed_heat"],
  "initial_cover_heat_loss_MJ_d": initial_heat_check["cover_heat_loss"],
  "initial_air_wall_heat_loss_MJ_d": initial_heat_check["air_wall_heat_loss"],
  "initial_soil_wall_heat_loss_MJ_d": initial_heat_check["soil_wall_heat_loss"],
  "initial_wall_heat_loss_MJ_d": initial_heat_check["wall_heat_loss"],
  "initial_heater_heat_MJ_d": initial_heat_check["heater_heat"],
  "q_ad_m3_d": q_ad_initial,
  "initial_q_ad_m3_d": q_ad_initial,
  "final_q_ad_m3_d": q_ad,
  "pyadm_q_ch4_base_m3_d": pyadm_q_ch4_base,
  "q_ch4_heater_base_m3_d": q_ch4_heater_base,
  "q_biogas_heater_base_m3_d": q_ch4_heater_base / biogas_methane_fraction,
  "methane_use_fraction_base": methane_use_fraction_from_pyadm(q_ch4_heater_base, pyadm_q_ch4_base),
  "textbook_methane_use_fraction_base": q_ch4_heater_base / produced_methane,
  "step_start_d": step_start,
  "step_end_d": step_end,
  "step_fraction": step_fraction,
  "step_absolute_m3_d": step_absolute,
}]).to_csv(parameter_path, index=False)
print(
  f"Simulation complete. Saved: {output_path}, {thermal_input_path}, {parameter_path} | "
  f"elapsed={time.perf_counter() - progress_start:.1f}s",
  flush=True
)

## ring test begin
# to compare the resutls with the dynamic simulation data from the BSM2 Matlab implementation
# pyOut = pd.read_csv("dynamic_out.csv")
# pyIn = pd.read_csv("digester_influent.csv")
# MatlabOut = pd.read_csv("Matlabout_dyn.csv")

# pyOut.time = pyIn.time
# pyOut.Q = pyIn.Q
# MatlabOut.Q = pyOut.Q
# mvalue = pvalue = 0
# ringtest = pd.DataFrame(columns=["state", "Matlab", "Python", "error"])

# n = 0
# for i in pyOut.columns:
#   Matlabinteg = integrate.trapz(MatlabOut[i] , MatlabOut.time)
#   pyinteg = integrate.trapz(pyOut[i] , pyOut.time)
#   results =pd.DataFrame([[MatlabOut[i].name, Matlabinteg/280, pyinteg/280, abs(pyinteg-Matlabinteg)/280]], columns=["state", "Matlab", "Python", "error"])
#   ringtest = ringtest.append(results)
#   print("Matlab " + MatlabOut[i].name + " average = " + str(Matlabinteg/280) + " Python " +  pyOut[i].name + " average = " + str(pyinteg/280) + " Error =" + str(abs(pyinteg-Matlabinteg)/280))
  
# ringtest.to_csv("ringtest.csv", index = False)


# for i in pyOut.columns:
#   plt.figure(figsize=(32, 8))
#   plt.plot(MatlabOut.time, MatlabOut[i], label = MatlabOut[i].name, linestyle="-", color = "red")
#   plt.plot(pyOut.time, pyOut[i], label = pyOut[i].name, linestyle="--", color = "blue")
#   plt.legend()
#   plt.show()

## ring test end
