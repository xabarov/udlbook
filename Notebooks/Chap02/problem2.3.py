import numpy as np

# three data points
x = np.array([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90])
y = np.array([0.67, 0.85, 1.05, 1.0, 1.40, 1.5, 1.3, 1.54, 1.55, 1.68, 1.73, 1.6 ])

# --- Discriminative: y = a + b x ---
X = np.vstack([np.ones_like(x), x]).T
phi_disc = np.linalg.lstsq(X, y, rcond=None)[0]

# --- Generative: x = c + d y ---
Y = np.vstack([np.ones_like(y), y]).T
phi_gen = np.linalg.lstsq(Y, x, rcond=None)[0]

# Invert generative model: y = (x - c) / d
def y_from_gen(x, phi):
    return (x - phi[0]) / phi[1]

print("Discriminative parameters (phi0, phi1):", phi_disc)
print("Generative parameters (phi0, phi1):", phi_gen)

# Compare predictions
x_test = np.array([1.5, 2.5])
y_disc_pred = phi_disc[0] + phi_disc[1] * x_test
y_gen_pred = y_from_gen(x_test, phi_gen)

print("Discriminative predictions:", y_disc_pred)
print("Generative predictions:", y_gen_pred)