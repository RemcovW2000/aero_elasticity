"""
Parameters from table 1 of paper.
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from plot_airfoil import plot_airfoil

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

omega_h = 50 # rad/s
omega_theta = 100 # rad/s
omega_beta = 300 # rad/s
mu = 40

a = -0.4
b = 1.0
c = 0.6

x_theta = 0.2
x_beta = -0.025

r_theta_sq = 0.25
r_beta_sq = 0.00625

# not from paper:
rho = 1.225
m_s = mu*np.pi*rho*b**2

M_matrix = m_s * b**2 * np.array(
    [
        [1, x_theta, x_beta],
        [x_theta, r_theta_sq, r_beta_sq + x_beta*(c-a)],
        [x_beta, r_beta_sq + x_beta*(c-a), r_beta_sq],
    ]
)

K_matrix = m_s * b**2 * np.diag([omega_h**2, r_theta_sq * omega_theta**2, r_beta_sq * omega_beta**2])

# find eigenfrequencies:
# coupled:
eigenvalues_sq, eigenvectors = np.linalg.eig(np.linalg.inv(M_matrix) @ K_matrix)
eigenvalues = np.sqrt(eigenvalues_sq)
print("coupled eigenvalues:")

h_index = np.argmin(eigenvalues)
h_freq = eigenvalues[h_index]

beta_index = np.argmax(eigenvalues)
beta_freq = eigenvalues[beta_index]

theta_index = next(i for i in range(len(eigenvalues)) if i not in (h_index, beta_index))
theta_freq = eigenvalues[theta_index]
print('coupled: [omega_h, omega_theta, omega_beta]:')
print(np.real(h_freq), np.real(theta_freq), np.real(beta_freq))

print("coupled eigenvectors: ")
coupled_heave_mode = eigenvectors[:, h_index]
coupled_theta_mode = eigenvectors[:, theta_index]
coupled_beta_mode = -eigenvectors[:, beta_index]
print("coupled eigenvectors: heave_mode, theta_mode, and beta_mode: ")
print(np.real(coupled_heave_mode), np.real(coupled_theta_mode), np.real(coupled_beta_mode))

# --- table data: uncoupled vs coupled frequencies and coupled eigenvectors ---
uncoupled_freqs = {"heave-dominated": omega_h,
                   "theta-dominated": omega_theta,
                   "beta-dominated": omega_beta}
coupled_freqs = {"heave-dominated": np.real(h_freq),
                 "theta-dominated": np.real(theta_freq),
                 "beta-dominated": np.real(beta_freq)}

modes = {
    "heave-dominated": coupled_heave_mode,
    "theta-dominated": coupled_theta_mode,
    "beta-dominated": coupled_beta_mode,
}

print("\ntable: mode | omega_uncoupled | omega_coupled | eigenvector (h/b, theta, beta)")
for name in modes:
    v = np.real(modes[name])
    print(f"{name:16s} | {uncoupled_freqs[name]:7.2f} | {coupled_freqs[name]:7.2f} | "
          f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})")

# --- plot the coupled vs uncoupled shape of each mode (eigenvector * scaling) ---
# note: the state vector is (h/b, theta, beta) per Eq. (2.9) of the paper,
# so eigenvector component 0 is h/b, not h -> multiply by b to get h.
uncoupled_modes = {
    "heave-dominated": np.array([1.0, 0.0, 0.0]),
    "theta-dominated": np.array([0.0, 1.0, 0.0]),
    "beta-dominated": np.array([0.0, 0.0, 1.0]),
}

mode_colors = {
    "heave-dominated": "tab:red",
    "theta-dominated": "tab:green",
    "beta-dominated": "tab:blue",
}

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(7, 7.5))
scaling = 0.5
for ax, name in zip(axes, modes):
    # uncoupled shape (the first call also draws the dashed reference outline)
    h_u, theta_u, beta_u = uncoupled_modes[name] * scaling
    plot_airfoil(b, a, c, x_theta, x_beta,
                 h=h_u * b, theta=theta_u, beta=beta_u,
                 ax=ax, color="0.45", label="uncoupled", show_markers=False)
    # coupled shape
    h_c, theta_c, beta_c = np.real(modes[name]) * scaling
    plot_airfoil(b, a, c, x_theta, x_beta,
                 h=h_c * b, theta=theta_c, beta=beta_c,
                 ax=ax, show_undeformed=False, color=mode_colors[name], label="coupled")
    ax.set_title(name, fontsize=10)

# shared legend on the first subplot (marker entries EA/hinge/CG/flap CG appear
# once, on the coupled shape), shapes ordered before points
handles, labels = axes[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))
point_names = ["EA", "hinge", "CG", "flap CG"]
shape_names = [l for l in unique if l not in point_names]
ordered = shape_names + [p for p in point_names if p in unique]
axes[0].legend([unique[l] for l in ordered], ordered, loc="center left",
               bbox_to_anchor=(1.02, 0.5), fontsize=8)

fig.suptitle(f"coupled vs uncoupled mode shapes ({scaling} * unit eigenvector)")
fig.tight_layout()
fig.savefig(FIG_DIR / "coupled_uncoupled_modes.pdf")
plt.show()