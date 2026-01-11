import matplotlib.pyplot as plt
import numpy as np
from optimizers import SGD, Adam, Nadam, RMSProp, SGDMomentum, SGDNesterov


def f_and_grad(xy):
    """
    xy: np.ndarray shape (2,)
    f(x,y) = x^2 + 100*y^2  (анизотропная квадратичная форма)
    grad = [2x, 200y]
    """
    x, y = xy[0], xy[1]
    f = x * x + 100.0 * y * y
    g = np.array([2.0 * x, 200.0 * y], dtype=float)
    return f, g


def run(OptClass, init_xy, steps=150, **opt_kwargs):
    xy = init_xy.astype(float).copy()          # shape (2,)
    p = xy.view()                              # будем обновлять "параметр" в месте
    params = [p]                               # один параметр-вектор из 2 координат

    opt = OptClass(params, **opt_kwargs)

    path = [p.copy()]
    losses = []

    for _ in range(steps):
        loss, grad = f_and_grad(p)
        losses.append(loss)

        grads = [grad]                         # градиент для единственного параметра
        opt.step(grads)

        path.append(p.copy())

    return np.array(path), np.array(losses)


def plot_contours(ax, xlim=(-2.5, 2.5), ylim=(-0.5, 0.5)):
    xs = np.linspace(*xlim, 400)
    ys = np.linspace(*ylim, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = X**2 + 50.0 * Y**2

    levels = np.geomspace(Z.min() + 1e-6, Z.max(), 25)
    ax.contour(X, Y, Z, levels=levels, linewidths=0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Contours of $f(x,y)=x^2+50y^2$ + trajectories")


def main():
    init = np.array([2.0, 0.4])  # старт: y заметно меньше по диапазону, но "жёстче" по кривизне
    steps = 120

    # Чтобы картинка была наглядной, шаги можно подбирать отдельно.
    # (Если хотите "честно одинаковый lr" — выставьте всем одинаковый и посмотрите, кто развалится.)
    
    lr1 = 0.005
    lr2 = 0.3
    lr3 = 0.03

    configs = {
        "SGD": (SGD, dict(lr=lr1)),
        "Momentum": (SGDMomentum, dict(lr=lr1, momentum=0.9)),
        "Nesterov": (SGDNesterov, dict(lr=lr1, momentum=0.9)),
        "Adam": (Adam, dict(lr=lr2, betas=(0.9, 0.999), eps=1e-8)),
        "Nadam": (Nadam, dict(lr=lr2, betas=(0.9, 0.999), eps=1e-8)),
    }

    configs["RMSProp"] = (RMSProp, dict(lr=lr3, rho=0.9, eps=1e-8))

    results = {}
    for name, (cls, kwargs) in configs.items():
        path, losses = run(cls, init, steps=steps, **kwargs)
        results[name] = (path, losses)

    # --- Figure 1: contours + trajectories ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_contours(ax1)

    for name, (path, _) in results.items():
        ax1.plot(path[:, 0], path[:, 1], linewidth=2, label=name)
        ax1.scatter(path[0, 0], path[0, 1], s=30)        # start
        ax1.scatter(path[-1, 0], path[-1, 1], s=30)      # end

    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # --- Figure 2: loss curves ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for name, (_, losses) in results.items():
        ax2.plot(losses, label=name, linewidth=2)

    ax2.set_title(r"Loss vs iterations ($f(x,y)$)")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("loss")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    main()