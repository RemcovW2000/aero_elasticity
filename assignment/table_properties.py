"""
Parameters from table 1 of paper.
"""
import numpy as np
from matplotlib import pyplot as plt

from plot_airfoil import plot_airfoil

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
coupled_beta_mode = eigenvectors[:, beta_index]
print("coupled eigenvectors: heave_mode, theta_mode, and beta_mode: ")
print(np.real(coupled_heave_mode), np.real(coupled_theta_mode), np.real(coupled_beta_mode))

# --- plot the unit deformation of each coupled mode (eigenvector * 1) ---
# note: the state vector is (h/b, theta, beta) per Eq. (2.9) of the paper,
# so eigenvector component 0 is h/b, not h -> multiply by b to get h.
modes = {
    "heave-dominated": coupled_heave_mode,
    "theta-dominated": coupled_theta_mode,
    "beta-dominated": coupled_beta_mode,
}

mode_colors = {
    "heave-dominated": "tab:red",
    "theta-dominated": "tab:green",
    "beta-dominated": "tab:blue",
}

fig, ax = plt.subplots(figsize=(9, 4.5))
scaling = 0.5
plot_airfoil(b, a, c, x_theta, x_beta, ax=ax, color="0.6", label="reference",
             show_markers=False)
for name, mode in modes.items():
    h_over_b, theta, beta = np.real(mode)
    plot_airfoil(b, a, c, x_theta, x_beta,
                 h=h_over_b * b * scaling, theta=theta * scaling, beta=beta * scaling,
                 ax=ax, show_undeformed=False, color=mode_colors[name], label=name)

# de-duplicate legend entries (EA/hinge/CG/flap CG markers repeat once per mode),
# then order shapes (reference + modes) before points (EA/hinge/CG/flap CG)
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
point_names = ["EA", "hinge", "CG", "flap CG"]
shape_names = [l for l in unique if l not in point_names]
ordered = shape_names + [p for p in point_names if p in unique]
ax.legend([unique[l] for l in ordered], ordered, loc="center left",
          bbox_to_anchor=(1.02, 0.5), fontsize=8)

ax.set_title(f"coupled mode shapes ({scaling} * unit eigenvector)")
fig.tight_layout()
plt.show()