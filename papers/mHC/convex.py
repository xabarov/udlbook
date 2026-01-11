import numpy as np
import matplotlib.pyplot as plt

def setup_ax(ax, title, xlim=(-6, 10), ylim=(-6, 10)):
    ax.axhline(0, color="k", lw=1, alpha=0.3)
    ax.axvline(0, color="k", lw=1, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

def plot_points_labels(ax, O, v1, v2):
    ax.scatter([O[0]], [O[1]], color="black", s=40, zorder=10)
    ax.text(O[0] + 0.08, O[1] + 0.08, "0", fontsize=11)

    ax.scatter([v1[0]], [v1[1]], color="black", s=40, zorder=10)
    ax.text(v1[0] + 0.08, v1[1] + 0.08, r"$v_1$", fontsize=11)

    ax.scatter([v2[0]], [v2[1]], color="black", s=40, zorder=10)
    ax.text(v2[0] + 0.08, v2[1] + 0.08, r"$v_2$", fontsize=11)

def main():
    v1 = np.array([2.0, 0.5])
    v2 = np.array([0.5, 2.0])
    O  = np.array([0.0, 0.0])

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()

    # 1) span{v1, v2}
    ax = axs[0]
    setup_ax(ax, r"$\mathrm{span}\{v_1,v_2\}$ (linear span)")
    rng = 3.0
    a = np.linspace(-rng, rng, 121)
    b = np.linspace(-rng, rng, 121)
    A, B = np.meshgrid(a, b)
    span_pts = (A[..., None] * v1 + B[..., None] * v2).reshape(-1, 2)
    ax.scatter(span_pts[:, 0], span_pts[:, 1], s=3, alpha=0.08, color="gray")
    plot_points_labels(ax, O, v1, v2)

    # 2) cone{v1, v2}
    ax = axs[1]
    setup_ax(ax, r"$\mathrm{cone}\{v_1,v_2\}$ (conic hull)")
    lam_max = 4.0
    lam = np.linspace(0, lam_max, 160)
    L1, L2 = np.meshgrid(lam, lam)
    cone_pts = (L1[..., None] * v1 + L2[..., None] * v2).reshape(-1, 2)
    ax.scatter(cone_pts[:, 0], cone_pts[:, 1], s=3, alpha=0.10, color="#ff7f0e")
    ray = np.linspace(0, lam_max, 200)
    ax.plot((ray[:, None] * v1)[:, 0], (ray[:, None] * v1)[:, 1], color="#ff7f0e", lw=2)
    ax.plot((ray[:, None] * v2)[:, 0], (ray[:, None] * v2)[:, 1], color="#ff7f0e", lw=2)
    plot_points_labels(ax, O, v1, v2)

    # 3) conv{0, v1, v2}
    ax = axs[2]
    setup_ax(ax, r"$\mathrm{conv}\{0,v_1,v_2\}$ (convex hull)")
    tri = np.vstack([O, v1, v2, O])  # замкнём контур
    ax.fill(tri[:, 0], tri[:, 1], color="#1f77b4", alpha=0.25)
    ax.plot(tri[:, 0], tri[:, 1], color="#1f77b4", lw=2)  # контур, чтобы точно было видно
    plot_points_labels(ax, O, v1, v2)

    # 4) aff{v1, v2}
    ax = axs[3]
    setup_ax(ax, r"$\mathrm{aff}\{v_1,v_2\}$ (affine hull)")
    t = np.linspace(-2.0, 3.0, 200)
    aff_line = v1 + t[:, None] * (v2 - v1)
    ax.plot(aff_line[:, 0], aff_line[:, 1], color="#2ca02c", lw=2)
    plot_points_labels(ax, O, v1, v2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()