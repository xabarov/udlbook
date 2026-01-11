# demo_newton_nn_scaling.py
import time
import torch


def make_data(n=128, d=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    true_w = torch.randn(d, 1, generator=g)
    y = X @ true_w + 0.1 * torch.randn(n, 1, generator=g)
    return X, y
def make_data_teacher_mlp(n=256, d=20, h_teacher=20, noise=0.05, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)

    X = torch.randn(n, d, generator=g, dtype=dtype)

    P_teacher = param_count(d, h_teacher)
    theta_teacher = 0.8 * torch.randn(P_teacher, generator=g, dtype=dtype)

    with torch.no_grad():
        y_clean = mlp_forward_from_theta(X, theta_teacher, d, h_teacher)

        eps = torch.randn(
            y_clean.shape,
            generator=g,
            dtype=y_clean.dtype,
            device=y_clean.device
        )
        y = y_clean + noise * eps

        # опциональная нормировка таргета
        y = (y - y.mean()) / (y.std() + 1e-8)

    return X, y

def param_count(d, h):
    # W1: h*d, b1: h, W2: 1*h, b2: 1
    return h * d + h + h * 1 + 1


def unpack_theta(theta, d, h):
    """
    theta: shape (P,)
    returns W1 (h,d), b1 (h,), W2 (1,h), b2 (1,)
    """
    i = 0
    W1 = theta[i : i + h * d].view(h, d); i += h * d
    b1 = theta[i : i + h].view(h); i += h
    W2 = theta[i : i + h].view(1, h); i += h
    b2 = theta[i : i + 1].view(1); i += 1
    return W1, b1, W2, b2


def mlp_forward_from_theta(X, theta, d, h):
    W1, b1, W2, b2 = unpack_theta(theta, d, h)
    hidden = torch.tanh(X @ W1.T + b1)     # (n,h)
    out = hidden @ W2.T + b2               # (n,1)
    return out


def loss_from_theta(theta, X, y, d, h):
    pred = mlp_forward_from_theta(X, theta, d, h)
    return torch.mean((pred - y) ** 2)


def dense_hessian_and_grad(theta, X, y, d, h):
    theta = theta.detach().clone().requires_grad_(True)

    def f(t):
        return loss_from_theta(t, X, y, d, h)

    loss = f(theta)
    grad = torch.autograd.grad(loss, theta, create_graph=True)[0]
    H = torch.autograd.functional.hessian(f, theta)

    return loss.detach(), grad.detach(), H.detach()

def newton_steps(theta0, X, y, d, h, steps=5, damping=1e-3, growth=10.0, max_tries=10):
    theta = theta0.detach().clone()

    for k in range(steps):
        # loss, grad, Hessian at current theta
        loss0 = loss_from_theta(theta, X, y, d, h).detach()
        _, g, H = dense_hessian_and_grad(theta, X, y, d, h)

        P = theta.numel()
        I = torch.eye(P, dtype=theta.dtype, device=theta.device)

        lam = damping
        accepted = False

        for _ in range(max_tries):
            delta = torch.linalg.solve(H + lam * I, g)
            theta_try = theta - delta
            loss_try = loss_from_theta(theta_try, X, y, d, h).detach()

            if loss_try <= loss0:
                theta = theta_try
                damping = max(lam / growth, 1e-12)
                accepted = True
                break

            lam *= growth

        if not accepted:
            # если совсем не получилось — делаем маленький градиентный шаг
            theta = theta - 1e-4 * g
            damping = lam

        print(
            f"  step {k+1:02d} | loss={loss0.item():.6e} -> {loss_from_theta(theta, X, y, d, h).item():.6e} "
            f"| ||g||={torch.linalg.norm(g).item():.3e} | damping={damping:.1e}"
        )

    return theta


def format_mem(bytes_):
    b = float(bytes_)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def main():
    torch.set_default_dtype(torch.float64)
    X, y = make_data_teacher_mlp(n=256, d=20, h_teacher=40, noise=0.03, seed=0, dtype=torch.get_default_dtype())
    d = X.shape[1]

    for h in [5, 10, 20, 40, 80]:
        P = param_count(d, h)
        bytes_hess = (P * P) * torch.tensor([], dtype=torch.get_default_dtype()).element_size()
        print(f"\nh={h:3d} | P={P:6d} | dense Hessian ~ {format_mem(bytes_hess)}")

        theta0 = torch.randn(P, dtype=torch.get_default_dtype()) * 0.1

        t0 = time.perf_counter()
        try:
            loss, g, H = dense_hessian_and_grad(theta0, X, y, d, h)
            t1 = time.perf_counter()
            print(f"  build grad+H: {t1 - t0:.3f}s | loss={loss.item():.3e} | H shape={tuple(H.shape)}")

            if P <= 5000:
                print("  Newton iterations:")
                _ = newton_steps(theta0, X, y, d, h, steps=3, damping=1e-3)
            else:
                print("  skip Newton steps (P too large for comfortable demo)")

        except RuntimeError as e:
            print(f"  failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()