import numpy as np
import matplotlib.pyplot as plt
import os

#J/(mol*K)
R = 8.31446261815324


#(300 K to 1000 K): a1, a2, a3, a4, a5, a6, a7
janaf98_solid_coeffs_low_T = [-1.3181931e+00, 1.1950484e-02, -1.0999163e-05, 2.1567584e-09, 1.2019863e-12, -4.5597194e+01, 5.1212201e+00]
#(1000 K to 2450 K)
janaf98_solid_coeffs_high_T = [2.1353842e+00, 6.2384826e-04, 5.2269843e-07, -3.4412816e-10, 5.4070294e-14, -8.2167211e+02, -1.2048934e+01]

#JANAF 1998 Liquid Boron Coefficients
#(2450 K to 5000 K)
janaf98_liquid_coeffs = [3.6735752e+00, 0.0, 0.0, 0.0, 0.0, 4.1164170e+02, -2.1048345e+01]

# McBride (from JANAF 83) Solid Boron Coefficients
#(200 K to 1000 K)
mcbride_solid_coeffs_low_T = [-1.15931693E+00, 1.13777145E-02, -1.06985988E-05, 2.76106443E-09, 7.31746996E-13, -7.13339210E+01, 4.36439895E+00]
#(1000 K to 2350 K)
mcbride_solid_coeffs_high_T = [1.83494094E+00, 1.79198702E-03, -7.97879498E-07, 2.02764512E-10, -1.92028345E-14, -7.83202899E+02, -1.06433298E+01]

# McBride (from JANAF 83) Liquid Boron Coefficients
#(2350 K to 6000 K)
mcbride_liquid_coeffs = [3.81862551E+00, 0.0, 0.0, 0.0, 0.0, 3.36060019E+03, -2.07322599E+01]

# NIST data for Enthalpy
nist_H_T = np.array([
    #solid phase
    298., 300., 400., 500., 600., 700., 800., 900., 1000., 1100., 1200., 1300., 1400., 1500., 1600., 1700., 1800., 1900., 2000., 2100., 2200., 2300.,
    #liquid phase
    2350., 2400., 2500., 2600., 2700., 2800., 2900., 3000., 3100., 3200., 3300., 3400., 3500., 3600., 3700., 3800., 3900., 4000., 4100.
])
nist_H_minus_H298 = np.array([
    #solid phase
    -0.00, 0.02, 1.40, 3.13, 5.10, 7.24, 9.51, 11.90, 14.37, 16.91, 19.50, 22.15, 24.84, 27.57, 30.33, 33.14, 36.01, 38.90, 41.83, 44.79, 47.79, 50.82,
    #liquid phase
    53.63, 55.22, 58.39, 61.57, 64.74, 67.92, 71.09, 74.27, 77.44, 80.62, 83.79, 86.97, 90.14, 93.32, 96.49, 99.67, 102.8, 106.0, 109.2
])


H_FUSION = 48.93  #kJ/mol

def calculate_h(T, coeffs):
    """
    Calculates enthalpy H from NASA 7-parameter polynomial coefficients.
    H = R * (a1*T + a2*T^2/2 + a3*T^3/3 + a4*T^4/4 + a5*T^5/5 + a6)
    Returns H in J/mol.
    """
    a1, a2, a3, a4, a5, a6, _ = coeffs
    T = np.asarray(T)
    H = R * (a1*T + (a2/2)*T**2 + (a3/3)*T**3 + (a4/4)*T**4 + (a5/5)*T**5 + a6)
    return H

def get_h_minus_h298(T, h298_ref, low_T_coeffs, high_T_coeffs, liquid_coeffs, temp_ranges):
    """Generic function to calculate H(T) - H(298.15) in kJ/mol."""
    T = np.asarray(T)
    h = np.zeros_like(T, dtype=float)

    t_min, t_mid, t_liquid_start, t_max = temp_ranges

    #solid phase
    low_T_mask = (T >= t_min) & (T < t_mid)
    h[low_T_mask] = calculate_h(T[low_T_mask], low_T_coeffs)

    high_T_mask = (T >= t_mid) & (T < t_liquid_start)
    h[high_T_mask] = calculate_h(T[high_T_mask], high_T_coeffs)

    #liquid phase
    liquid_mask = (T >= t_liquid_start) & (T <= t_max)
    h[liquid_mask] = calculate_h(T[liquid_mask], liquid_coeffs)

    #invalidate ranges outside the model's scope
    invalid_mask = (T < t_min) | (T > t_max)
    h[invalid_mask] = np.nan
    
    return (h - h298_ref) / 1000.0


def plot_enthalpy_comparison():
    """Generates and saves a plot for boron enthalpy comparison."""
    T_range = np.linspace(200, 6000, 1000)
    plt.figure(figsize=(14, 9))

    #Nist
    nist_H_T_liquid_mask = nist_H_T >= 2350
    nist_H_minus_H298[nist_H_T_liquid_mask] += H_FUSION
    plt.plot(nist_H_T, nist_H_minus_H298, 'o', label='NIST Data', markersize=6)

    #JANAF 1998
    h298_janaf98 = calculate_h(298.15, janaf98_solid_coeffs_low_T)
    temp_ranges_janaf = (300, 1000, 2450, 5000)
    enthalpy_janaf = get_h_minus_h298(T_range, h298_janaf98, janaf98_solid_coeffs_low_T, janaf98_solid_coeffs_high_T, janaf98_liquid_coeffs, temp_ranges_janaf)
    enthalpy_janaf[T_range >= temp_ranges_janaf[2]] += H_FUSION
    plt.plot(T_range, enthalpy_janaf, label='JANAF 1998')

    #McBride (JANAF 83)
    h298_mcbride = calculate_h(298.15, mcbride_solid_coeffs_low_T)
    temp_ranges_mcbride = (200, 1000, 2350, 6000)
    enthalpy_mcbride = get_h_minus_h298(T_range, h298_mcbride, mcbride_solid_coeffs_low_T, mcbride_solid_coeffs_high_T, mcbride_liquid_coeffs, temp_ranges_mcbride)
    enthalpy_mcbride[T_range >= temp_ranges_mcbride[2]] += H_FUSION
    plt.plot(T_range, enthalpy_mcbride, label='McBride (from JANAF 83)', linestyle='--')
    
    plt.title('Сравнение энтальпии бора H(T) с учетом энтальпии образования')
    plt.xlabel('Температура (K)')
    plt.ylabel('Энтальпия (кДж/моль)')
    plt.grid(True, which='both', linestyle='--')
    
    plt.axvline(x=2350, color='gray', linestyle=':', label='T melt (McBride)')
    plt.axvline(x=2450, color='black', linestyle=':', label='T melt (JANAF 98)')
    plt.legend()

    filename = "boron_enthalpy_comparison.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

if __name__ == '__main__':
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    original_cwd = os.getcwd()
    os.chdir(output_dir)

    try:
        plot_enthalpy_comparison()
        print("\nСравнение энтальпии завершено. График сохранен в папке 'output'.")
    finally:
        os.chdir(original_cwd) 