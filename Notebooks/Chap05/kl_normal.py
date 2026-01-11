import numpy as np

def kl_normal(mu1, sigma1, mu2, sigma2):
    """KL( N(mu1,sigma1^2) || N(mu2,sigma2^2) )"""
    mu1, sigma1, mu2, sigma2 = map(float, (mu1, sigma1, mu2, sigma2))
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)

def integrate_trapz(y, x):
    # NumPy 2.x: np.trapezoid, старые: np.trapz
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

def kl_numeric(mu1, sigma1, mu2, sigma2, x_min=-30, x_max=30, n=400_000):
    x = np.linspace(x_min, x_max, n)
    p = normal_pdf(x, mu1, sigma1)
    q = normal_pdf(x, mu2, sigma2)
    integrand = p * (np.log(p) - np.log(q))
    return integrate_trapz(integrand, x)

mu1, s1 = 0.0, 1.0
mu2, s2 = 1.0, 2.0

kl_f = kl_normal(mu1, s1, mu2, s2)
kl_n = kl_numeric(mu1, s1, mu2, s2)

print("KL(formula) =", kl_f)
print("KL(numeric) =", kl_n)
print("abs diff    =", abs(kl_f - kl_n))

print("\nАсимметрия:")
print("KL(N1||N2) =", kl_normal(mu1, s1, mu2, s2))
print("KL(N2||N1) =", kl_normal(mu2, s2, mu1, s1))