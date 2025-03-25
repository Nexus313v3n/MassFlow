import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
c = 3e8  # m/s
H0 = 70e3 / 3.086e22  # 2.27e-18 s^-1
to_km_s_Mpc = c / 3.086e22 * 1e6  # ~9.715e14

# Load Union2.1 data (user must provide this file)
# data = np.loadtxt('Union2.1.txt', usecols=(1, 2, 3))
# z_data = data[:, 0]
# mu_data = data[:, 1]
# mu_err = data[:, 2]

# MassFlow model: H(z) = 1.45e-18 [(1 + z)^1.1 + 0.25] + 0.25e-18 (1 + z)^1.75
def H_your(z):
    return 1.45e-18 * ((1 + z)**1.1 + 0.25) + 0.25e-18 * ((1 + z)**1.75)

# LambdaCDM for comparison
Omega_m = 0.3
Omega_L = 0.7
def H_lcdm(z):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

# H(z) in km/s/Mpc
def H_your_km_s_Mpc(z):
    return H_your(z) * to_km_s_Mpc

def H_lcdm_km_s_Mpc(z):
    return H_lcdm(z) * to_km_s_Mpc

# Luminosity distance
def dL_your(z):
    integrand = lambda z_prime: c / H_your(z_prime)
    integral, _ = quad(integrand, 0, z)
    return (1 + z) * integral / 3.086e22

def dL_lcdm(z):
    integrand = lambda z_prime: c / H_lcdm(z_prime)
    integral, _ = quad(integrand, 0, z)
    return (1 + z) * integral / 3.086e22

# Angular diameter distance
def dA_your(z):
    integrand = lambda z_prime: c / H_your(z_prime)
    integral, _ = quad(integrand, 0, z)
    return integral / (1 + z) / 3.086e22

# Distance modulus
def mu_your(z):
    return 5 * np.log10(dL_your(z) * 1e6) - 5

def mu_lcdm(z):
    return 5 * np.log10(dL_lcdm(z) * 1e6) - 5

# Compute models
z_range = np.linspace(0.01, 1.4, 100)
mu_your_vals = [mu_your(z) for z in z_range]
mu_lcdm_vals = [mu_lcdm(z) for z in z_range]

# Supernova plot (for X Post 4)
plt.figure(figsize=(10, 6))
# plt.errorbar(z_data, mu_data, yerr=mu_err, fmt='o', color='gray', alpha=0.5, label='Union2.1 Data')
plt.plot(z_range, mu_your_vals, 'b-', label='MassFlow')
plt.plot(z_range, mu_lcdm_vals, 'r--', label='ΛCDM (Ω_m=0.3, Ω_Λ=0.7)')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance Modulus (μ)')
plt.title('MassFlow vs. ΛCDM')
plt.legend()
plt.grid(True)
plt.savefig('massflow_supernova_plot.png')  # Save for X
plt.show()

# Key outputs
print("H(z=0):", H_your_km_s_Mpc(0), "km/s/Mpc")
print("H(z=0.35):", H_your_km_s_Mpc(0.35), "km/s/Mpc")
print("H(z=2.34):", H_your_km_s_Mpc(2.34), "km/s/Mpc")
print("H(z=1100):", H_your_km_s_Mpc(1100), "km/s/Mpc")
print("mu(z=0.2):", mu_your(0.2))
print("mu(z=1.4):", mu_your(1.4))
print("d_A(z=1100):", dA_your(1100), "Mpc")
