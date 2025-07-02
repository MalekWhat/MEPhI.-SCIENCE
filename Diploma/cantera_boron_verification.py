import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn

#model 1: Shomate (NIST)
yaml_s_shomate = """
phases:
- name: shomate_solid_phase
  thermo: fixed-stoichiometry
  elements: [B]
  species: [B(s)_shomate]
species:
- name: B(s)_shomate
  composition: {B: 1}
  thermo:
    model: Shomate
    temperature-ranges: [298.0, 1800.0, 2350.0]
    data:
    # T(low) = 298, T(mid) = 1800, T(high) = 2350
    # Cp = A + B*t + C*t^2 + D*t^3 + E/t^2, t = T/1000
    # H - H(298.15) = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H
    # Coeffs: A, B, C, D, E, F, G, H=H(298.15)
    # Range 298-1800 K
    - [10.18574, 29.24415, -18.02137, 4.212326, -0.550999, -6.036299, 7.089077]
    # Range 1800-2350 K
    - [25.12664, 1.975493, 0.338395, -0.040032, -2.635578, -14.43597, 25.59930]
"""

yaml_l_shomate = """
phases:
- name: shomate_liquid_phase
  thermo: fixed-stoichiometry
  elements: [B]
  species: [B(l)_shomate]
species:
- name: B(l)_shomate
  composition: {B: 1}
  thermo:
    model: Shomate
    temperature-ranges: [2350.0, 5000.0]
    data:
    # T(low) = 2350, T(high) = 5000
    # Range 2350-5000 K
    - [31.38000, 0.0, 0.0, 0.0, 0.0, -18.75100, 31.86500]
"""

#model 2: McBride (JANAF 83)
yaml_s_mcbride = """
phases:
- name: mcbride_solid_phase
  thermo: fixed-stoichiometry
  elements: [B]
  species: [B(s)_mcbride]
species:
- name: B(s)_mcbride
  composition: {B: 1}
  thermo:
    model: NASA7
    temperature-ranges: [200.0, 1000.0, 2350.0]
    data:
    # Range 200-1000 K
    - [-1.15931693E+00, 1.13777145E-02, -1.06985988E-05, 2.76106443E-09, 7.31746996E-13, -7.13339210E+01, 4.36439895E+00]
    # Range 1000-2350 K
    - [1.83494094E+00, 1.79198702E-03, -7.97879498E-07, 2.02764512E-10, -1.92028345E-14, -7.83202899E+02, -1.06433298E+01]
"""

yaml_l_mcbride = """
phases:
- name: mcbride_liquid_phase
  thermo: fixed-stoichiometry
  elements: [B]
  species: [B(l)_mcbride]
species:
- name: B(l)_mcbride
  composition: {B: 1}
  thermo:
    model: NASA7
    temperature-ranges: [2350.0, 6000.0]
    data:
    # Range 2350-6000 K
    - [3.81862551E+00, 0.0, 0.0, 0.0, 0.0, 3.36060019E+03, -2.07322599E+01]
"""

#model 3: JANAF 1998
yaml_s_janaf98 = """
phases:
- name: janaf98_solid_phase
  thermo: fixed-stoichiometry
  elements: [B]
  species: [B(s)_janaf98]
species:
- name: B(s)_janaf98
  composition: {B: 1}
  thermo:
    model: NASA7
    temperature-ranges: [300.0, 1000.0, 2450.0]
    data:
    # Range 300-1000 K
    - [-1.3181931e+00, 1.1950484e-02, -1.0999163e-05, 2.1567584e-09, 1.2019863e-12, -4.5597194e+01, 5.1212201e+00]
    # Range 1000-2450 K
    - [2.1353842e+00, 6.2384826e-04, 5.2269843e-07, -3.4412816e-10, 5.4070294e-14, -8.2167211e+02, -1.2048934e+01]
"""

yaml_l_janaf98 = """
phases:
- name: janaf98_liquid_phase
  thermo: fixed-stoichiometry
  elements: [B]
  species: [B(l)_janaf98]
species:
- name: B(l)_janaf98
  composition: {B: 1}
  thermo:
    model: NASA7
    temperature-ranges: [2450.0, 5000.0]
    data:
    # Range 2450-5000 K
    - [3.6735752e+00, 0.0, 0.0, 0.0, 0.0, 4.1164170e+02, -2.1048345e+01]
"""

def calculate_properties(sol_obj, T_range):
    """Calculates Cp and H for a given Cantera solution object over a T range."""
    cp_values_kmol = []
    h_values_kmol = []
    for T in T_range:
        sol_obj.TP = T, ct.one_atm
        cp_values_kmol.append(sol_obj.cp_mole)
        h_values_kmol.append(sol_obj.enthalpy_mole)

    # Cp: J/kmol-K -> J/mol-K
    cp_values_mol = np.array(cp_values_kmol) / 1000
    # H: J/kmol -> kJ/mol
    h_values_kj_mol = np.array(h_values_kmol) / 1000000

    return cp_values_mol, h_values_kj_mol

def main():
    sol_s_shomate = ct.Solution(yaml=yaml_s_shomate)
    sol_l_shomate = ct.Solution(yaml=yaml_l_shomate)
    sol_s_mcbride = ct.Solution(yaml=yaml_s_mcbride)
    sol_l_mcbride = ct.Solution(yaml=yaml_l_mcbride)
    sol_s_janaf98 = ct.Solution(yaml=yaml_s_janaf98)
    sol_l_janaf98 = ct.Solution(yaml=yaml_l_janaf98)

    T_solid = np.linspace(298.15, 2350, 500)
    T_liquid = np.linspace(2350, 5000, 500)
    
    H_FUSION = 48.93

    cp_s_shomate, h_s_shomate = calculate_properties(sol_s_shomate, T_solid)
    cp_l_shomate, h_l_shomate = calculate_properties(sol_l_shomate, T_liquid)
    
    cp_s_mcbride, h_s_mcbride = calculate_properties(sol_s_mcbride, T_solid)
    cp_l_mcbride, h_l_mcbride = calculate_properties(sol_l_mcbride, T_liquid)

    cp_s_janaf98, h_s_janaf98 = calculate_properties(sol_s_janaf98, T_solid)
    cp_l_janaf98, h_l_janaf98 = calculate_properties(sol_l_janaf98, T_liquid)

    sol_s_shomate.TP = 298.15, ct.one_atm
    H298_shomate = sol_s_shomate.enthalpy_mole / 1000000
    sol_s_mcbride.TP = 298.15, ct.one_atm
    H298_mcbride = sol_s_mcbride.enthalpy_mole / 1000000
    sol_s_janaf98.TP = 298.15, ct.one_atm
    H298_janaf98 = sol_s_janaf98.enthalpy_mole / 1000000
    
    plt.rcParams.update({'font.size': 17})

    fig1, ax1 = plt.subplots(figsize=(14, 9))
    T_shomate = np.concatenate((T_solid, T_liquid))
    cp_shomate = np.concatenate((cp_s_shomate, cp_l_shomate))
    
    T_mcbride_solid = np.linspace(200, 2350, 500)
    T_mcbride_liquid = np.linspace(2350, 6000, 500)
    cp_s_mcbride, _ = calculate_properties(sol_s_mcbride, T_mcbride_solid)
    cp_l_mcbride, _ = calculate_properties(sol_l_mcbride, T_mcbride_liquid)
    T_mcbride = np.concatenate((T_mcbride_solid, T_mcbride_liquid))
    cp_mcbride = np.concatenate((cp_s_mcbride, cp_l_mcbride))

    T_janaf_solid = np.linspace(300, 2450, 500)
    T_janaf_liquid = np.linspace(2450, 5000, 500)
    cp_s_janaf98, _ = calculate_properties(sol_s_janaf98, T_janaf_solid)
    cp_l_janaf98, _ = calculate_properties(sol_l_janaf98, T_janaf_liquid)
    T_janaf = np.concatenate((T_janaf_solid, T_janaf_liquid))
    cp_janaf = np.concatenate((cp_s_janaf98, cp_l_janaf98))

    ax1.plot(T_shomate, cp_shomate, 'r-', label='Shomate (NIST)')
    ax1.plot(T_mcbride, cp_mcbride, 'g--', label='McBride (JANAF 83)')
    ax1.plot(T_janaf, cp_janaf, 'b-', label='JANAF 1998')
    
    ax1.set_title('Сравнение теплоемкости бора (Cantera)', fontsize=20)
    ax1.set_xlabel('Температура (K)')
    ax1.set_ylabel('Теплоемкость Cp (Дж/моль·K)')
    ax1.grid(True, which='both', linestyle='--')
    
    ax1.axvline(x=2350, color='gray', linestyle=':', label='T плавл. (Shomate, McBride)')
    ax1.axvline(x=2450, color='black', linestyle=':', label='T плавл. (JANAF 98)')
    
    ax1.legend()
    ax1.set_xlim(0, 5000)
    ax1.set_ylim(0, 40)
    fig1.tight_layout()
    fig1.savefig('cantera_cp_verification.png')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 9))
    
    #NIST experimental data points
    nist_H_T = np.array([
        298., 300., 400., 500., 600., 700., 800., 900., 1000., 1100., 1200., 1300., 1400., 1500., 1600., 1700., 1800., 1900., 2000., 2100., 2200., 2300.,
        2350., 2400., 2500., 2600., 2700., 2800., 2900., 3000., 3100., 3200., 3300., 3400., 3500., 3600., 3700., 3800., 3900., 4000., 4100.
    ])
    nist_H_minus_H298 = np.array([
        -0.00, 0.02, 1.40, 3.13, 5.10, 7.24, 9.51, 11.90, 14.37, 16.91, 19.50, 22.15, 24.84, 27.57, 30.33, 33.14, 36.01, 38.90, 41.83, 44.79, 47.79, 50.82,
        53.63, 55.22, 58.39, 61.57, 64.74, 67.92, 71.09, 74.27, 77.44, 80.62, 83.79, 86.97, 90.14, 93.32, 96.49, 99.67, 102.8, 106.0, 109.2
    ])
    nist_H_T_liquid_mask = nist_H_T >= 2350
    nist_H_minus_H298[nist_H_T_liquid_mask] += H_FUSION
    ax2.plot(nist_H_T, nist_H_minus_H298, 'o', label='NIST Data (таблицы)', markersize=6)

    _, h_s_shomate = calculate_properties(sol_s_shomate, T_solid)
    _, h_l_shomate = calculate_properties(sol_l_shomate, T_liquid)
    
    _, h_s_mcbride = calculate_properties(sol_s_mcbride, T_mcbride_solid)
    _, h_l_mcbride = calculate_properties(sol_l_mcbride, T_mcbride_liquid)
    
    _, h_s_janaf98 = calculate_properties(sol_s_janaf98, T_janaf_solid)
    _, h_l_janaf98 = calculate_properties(sol_l_janaf98, T_janaf_liquid)


    h_l_shomate_adjusted = h_l_shomate + H_FUSION
    h_l_mcbride_adjusted = h_l_mcbride + H_FUSION
    h_l_janaf98_adjusted = h_l_janaf98 + H_FUSION
        
    h_shomate = np.concatenate((h_s_shomate - H298_shomate, h_l_shomate_adjusted - H298_shomate))
    h_mcbride = np.concatenate((h_s_mcbride - H298_mcbride, h_l_mcbride_adjusted - H298_mcbride))
    h_janaf = np.concatenate((h_s_janaf98 - H298_janaf98, h_l_janaf98_adjusted - H298_janaf98))

    ax2.plot(T_shomate, h_shomate, 'r-', label='Shomate (NIST)')
    ax2.plot(T_mcbride, h_mcbride, 'g--', label='McBride (JANAF 83)')
    ax2.plot(T_janaf, h_janaf, 'b-', label='JANAF 1998')

    ax2.set_title('Сравнение энтальпии бора H(T) с учетом теплоты плавления (Cantera)', fontsize=20)
    ax2.set_xlabel('Температура (K)')
    ax2.set_ylabel('Энтальпия (кДж/моль)')
    ax2.grid(True, which='both', linestyle='--')

    ax2.axvline(x=2350, color='gray', linestyle=':', label='T плавл. (Shomate, McBride)')
    ax2.axvline(x=2450, color='black', linestyle=':', label='T плавл. (JANAF 98)')

    ax2.legend()
    ax2.set_xlim(0, 5200)
    ax2.set_ylim(-5, 250)
    fig2.tight_layout()
    fig2.savefig('cantera_enthalpy_verification.png')
    plt.close(fig2)
    
    print("Verification plots 'cantera_cp_verification.png' and 'cantera_enthalpy_verification.png' have been generated.")


if __name__ == '__main__':
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_cwd = os.getcwd()
    os.chdir(output_dir)

    try:
        main()
    finally:
        os.chdir(original_cwd) 