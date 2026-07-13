from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from plot_airfoil import plot_airfoil

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# table 1 of the paper (alpha written as theta), coordinates x = [h/b, theta, beta]
omega_h, omega_theta, omega_beta = 50.0, 100.0, 300.0   # uncoupled [rad/s]
mu = 40.0
a, b, c = -0.4, 1.0, 0.6
x_theta, x_beta = 0.2, -0.025
r_theta_sq, r_beta_sq = 0.25, 0.00625
rho = 1.225                                             # not from the paper
m_s = mu * np.pi * rho * b**2

M_s = m_s * b**2 * np.array([
    [1.0,      x_theta,                              x_beta],
    [x_theta,  r_theta_sq,                           r_beta_sq + x_beta * (c - a)],
    [x_beta,   r_beta_sq + x_beta * (c - a),         r_beta_sq],
])
K_s = m_s * b**2 * np.diag([omega_h**2,
                            r_theta_sq * omega_theta**2,
                            r_beta_sq * omega_beta**2])


def coupled_modes():
    """Return the coupled natural frequencies (ascending) and unit eigenvectors, dominant DOF positive."""
    lam, vec = np.linalg.eig(np.linalg.inv(M_s) @ K_s)
    order = np.argsort(lam.real)
    freqs, vec = np.sqrt(lam.real[order]), vec[:, order].real
    vec *= np.sign(vec[np.abs(vec).argmax(axis=0), range(3)])
    return freqs, vec


if __name__ == "__main__":
    mode_names = ["heave-dominated", "theta-dominated", "beta-dominated"]
    mode_colors = ["tab:red", "tab:green", "tab:blue"]
    uncoupled_freqs = [omega_h, omega_theta, omega_beta]
    coupled_freqs, coupled_vecs = coupled_modes()

    print("mode | omega_uncoupled | omega_coupled | eigenvector (h/b, theta, beta)")
    for j, name in enumerate(mode_names):
        v = coupled_vecs[:, j]
        print(f"{name:16s} | {uncoupled_freqs[j]:7.2f} | {coupled_freqs[j]:7.2f} | "
              f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})")

    # coupled vs uncoupled shape of each mode; eigenvector component 0 is h/b,
    # not h, so it is multiplied by b when plotting
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(7, 7.5))
    scaling = 0.5
    for j, (ax, name) in enumerate(zip(axes, mode_names)):
        h_u, theta_u, beta_u = np.eye(3)[j] * scaling
        plot_airfoil(b, a, c, x_theta, x_beta,
                     h=h_u * b, theta=theta_u, beta=beta_u,
                     ax=ax, color="0.45", label="uncoupled", show_markers=False)
        h_c, theta_c, beta_c = coupled_vecs[:, j] * scaling
        plot_airfoil(b, a, c, x_theta, x_beta,
                     h=h_c * b, theta=theta_c, beta=beta_c,
                     ax=ax, show_undeformed=False, color=mode_colors[j], label="coupled")
        ax.set_title(name, fontsize=10)

    # shared legend, de-duplicated (marker labels repeat), shapes before points
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    points = ["EA", "hinge", "CG", "flap CG"]
    ordered = [l for l in unique if l not in points] + [p for p in points if p in unique]
    axes[0].legend([unique[l] for l in ordered], ordered, loc="center left",
                   bbox_to_anchor=(1.02, 0.5), fontsize=8)

    fig.suptitle(f"coupled vs uncoupled mode shapes ({scaling} * unit eigenvector)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coupled_uncoupled_modes.pdf")
    plt.show()
