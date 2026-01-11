import numpy as np

def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)

def kl_discrete(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))

def empirical_kl_hist_vs_model(data, model_pdf, bins=40, data_range=None, eps=1e-12):
    """
    1) Оцениваем P по гистограмме: p_i ~ частоты в бинах
    2) Оцениваем Q как массу модели на тех же бинах: q_i ≈ pdf(center_i)*width
       (потом нормируем q, чтобы сумма была 1)
    3) Возвращаем KL(P||Q) по дискретным вероятностям
    """
    hist_counts, edges = np.histogram(data, bins=bins, range=data_range)
    p = hist_counts.astype(float)
    p = p / p.sum()

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    q_unnorm = model_pdf(centers) * widths
    q = q_unnorm / q_unnorm.sum()

    return kl_discrete(p, q, eps=eps), p, q, edges

# --- Сгенерируем данные: смесь двух нормалей (она НЕ нормальная) ---
rng = np.random.default_rng(0)
n = 50_000
data = np.concatenate([
    rng.normal(loc=-2.0, scale=0.8, size=int(0.6*n)),
    rng.normal(loc= 2.0, scale=0.5, size=int(0.4*n)),
])

# --- Модель Q1: "одна нормаль", подогнанная по mean/std данных ---
mu_hat = data.mean()
sigma_hat = data.std(ddof=0)
Q1 = lambda x: normal_pdf(x, mu_hat, sigma_hat)

# --- Модель Q2: заведомо плохая нормаль ---
Q2 = lambda x: normal_pdf(x, 0.0, 1.0)

kl1, p, q1, edges = empirical_kl_hist_vs_model(data, Q1, bins=60, data_range=(-6, 6))
kl2, _, q2, _     = empirical_kl_hist_vs_model(data, Q2, bins=60, data_range=(-6, 6))

print("Данные: смесь двух нормалей")
print("KL(P_hat || Q1_fitNormal) =", kl1)
print("KL(P_hat || Q2_N(0,1))    =", kl2)