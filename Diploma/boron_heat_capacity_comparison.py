import numpy as np
import matplotlib.pyplot as plt
import os

R = 8.31446261815324 #J/(mol*K)


#JANAF 1998 Solid Boron Coefficients
#300 K to 1000 K
janaf98_solid_coeffs_low_T = [-1.3181931e+00, 1.1950484e-02, -1.0999163e-05, 2.1567584e-09, 1.2019863e-12]
# Range 2: 1000 K to 2450 K
janaf98_solid_coeffs_high_T = [2.1353842e+00, 6.2384826e-04, 5.2269843e-07, -3.4412816e-10, 5.4070294e-14]

#JANAF 1998 Liquid Boron Coefficients
#2450 K to 5000 K
janaf98_liquid_coeffs = [3.6735752e+00, 0.0, 0.0, 0.0, 0.0]

#McBride (from JANAF 83) Solid Boron Coefficients
#200 K to 1000 K
mcbride_solid_coeffs_low_T = [-1.15931693E+00, 1.13777145E-02, -1.06985988E-05, 2.76106443E-09, 7.31746996E-13]
#1000 K to 2350 K
mcbride_solid_coeffs_high_T = [1.83494094E+00, 1.79198702E-03, -7.97879498E-07, 2.02764512E-10, -1.92028345E-14]

#McBride (from JANAF 83) Liquid Boron Coefficients
#2350 K to 6000 K
mcbride_liquid_coeffs = [3.81862551E+00, 0.0, 0.0, 0.0, 0.0]

#NIST data for Solid Boron
nist_solid_T = np.array([298., 300., 400., 500., 600., 700., 800., 900., 1000., 1100., 1200., 1300., 1400., 1500., 1600., 1700., 1800., 1900., 2000., 2100., 2200., 2300.])
nist_solid_Cp = np.array([11.21, 11.33, 15.83, 18.63, 20.62, 22.15, 23.34, 24.30, 25.07, 25.70, 26.22, 26.68, 27.08, 27.48, 27.88, 28.32, 28.73, 29.10, 29.45, 29.80, 30.14, 30.48])

#NIST data for Liquid Boron
nist_liquid_T = np.array([2350., 2400., 2500., 2600., 2700., 2800., 2900., 3000., 3100., 3200., 3300., 3400., 3500., 3600., 3700., 3800., 3900., 4000., 4100.])
nist_liquid_Cp = np.array([31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75, 31.75])

def calculate_cp(T, coeffs):
    """
    Calculates heat capacity (Cp) from NASA polynomial coefficients.
    Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    """
    T_powers = np.array([np.ones_like(T), T, T**2, T**3, T**4])
    if T_powers.ndim == 1:
        T_powers = T_powers[:, np.newaxis]
    cp_over_r = np.dot(coeffs, T_powers)
    return R * cp_over_r

def get_janaf98_solid_cp(T):
    """Calculates Cp for JANAF 1998 solid boron across temperature ranges."""
    T = np.asarray(T)
    cp = np.zeros_like(T, dtype=float)

    # (300 <= T < 1000)
    low_T_mask = (T >= 300) & (T < 1000)
    cp[low_T_mask] = calculate_cp(T[low_T_mask], janaf98_solid_coeffs_low_T)

    #(1000 <= T <= 2450)
    high_T_mask = (T >= 1000) & (T <= 2450)
    cp[high_T_mask] = calculate_cp(T[high_T_mask], janaf98_solid_coeffs_high_T)
    
    #outside
    invalid_mask = ~ (low_T_mask | high_T_mask)
    cp[invalid_mask] = np.nan
    
    return cp

def get_mcbride_solid_cp(T):
    """Calculates Cp for McBride (JANAF 83) solid boron across temperature ranges."""
    T = np.asarray(T)
    cp = np.zeros_like(T, dtype=float)
    
    #(200 <= T < 1000)
    low_T_mask = (T >= 200) & (T < 1000)
    cp[low_T_mask] = calculate_cp(T[low_T_mask], mcbride_solid_coeffs_low_T)

    #(1000 <= T <= 2350)
    high_T_mask = (T >= 1000) & (T <= 2350)
    cp[high_T_mask] = calculate_cp(T[high_T_mask], mcbride_solid_coeffs_high_T)

    invalid_mask = ~ (low_T_mask | high_T_mask)
    cp[invalid_mask] = np.nan
    
    return cp

def get_janaf98_liquid_cp(T):
    """Calculates Cp for JANAF 1998 liquid boron."""
    T = np.asarray(T)
    cp = calculate_cp(T, janaf98_liquid_coeffs)
    cp[(T < 2450) | (T > 5000)] = np.nan
    return cp
    
def get_mcbride_liquid_cp(T):
    """Calculates Cp for McBride (JANAF 83) liquid boron."""
    T = np.asarray(T)
    cp = calculate_cp(T, mcbride_liquid_coeffs)
    cp[(T < 2350) | (T > 6000)] = np.nan
    return cp
    
def plot_combined_cp_comparison():
    """Generates and saves a combined plot for solid and liquid boron heat capacity."""
    T_range = np.linspace(200, 5000, 1000)
    
    plt.figure(figsize=(14, 9))
    
    #NIST Data
    plt.plot(nist_solid_T, nist_solid_Cp, 'o', label='NIST Data', markersize=5, color='C0')
    plt.plot(nist_liquid_T, nist_liquid_Cp, 'o', markersize=5, color='C0') # No label for the second part

    #JANAF 1998
    cp_janaf_solid = get_janaf98_solid_cp(T_range)
    cp_janaf_liquid = get_janaf98_liquid_cp(T_range)
    cp_janaf_combined = np.where(np.isnan(cp_janaf_solid), cp_janaf_liquid, cp_janaf_solid)
    plt.plot(T_range, cp_janaf_combined, label='JANAF 1998', color='C1')

    # McBride(JANAF 83)
    cp_mcbride_solid = get_mcbride_solid_cp(T_range)
    cp_mcbride_liquid = get_mcbride_liquid_cp(T_range)
    cp_mcbride_combined = np.where(np.isnan(cp_mcbride_solid), cp_mcbride_liquid, cp_mcbride_solid)
    plt.plot(T_range, cp_mcbride_combined, label='McBride (from JANAF 83)', linestyle='--', color='C2')

    plt.title('Сравнение теплоемкости бора (твердая и жидкая фазы)')
    plt.xlabel('Температура (K)')
    plt.ylabel('Теплоемкость Cp (Дж/моль·K)')
    plt.grid(True, which='both', linestyle='--')
    
    plt.axvline(x=2350, color='gray', linestyle=':', label='T melt (McBride)')
    plt.axvline(x=2450, color='black', linestyle=':', label='T melt (JANAF 98)')
    
    plt.legend()
    plt.xlim(0, 5000)
    plt.ylim(0, 40)

    filename = "boron_cp_comparison_combined.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

if __name__ == '__main__':
    if not os.path.exists('output'):
        os.makedirs('output')
    
    os.chdir('output')
    
    plot_combined_cp_comparison()
    print("\nСравнение завершено. График сохранен в папке 'output'.") 