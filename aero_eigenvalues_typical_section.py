import numpy as np
from matplotlib import pyplot as plt

from typical_section import TypicalSectionDynamicParams, TypicalSectionAeroParams, \
TypicalSection

params = TypicalSectionDynamicParams(
    m_airfoil=1.567,
    m_flap=0.0,
    I_airfoil=1.0,
    I_flap=0.01,
    b=0.127,
    a=-0.5,
    c=0.5,
    x_theta=-0.5,
    x_beta=0.1,
    K_h=2818.8,
    K_theta=37.3,
    K_beta=0.0,
)

aero_params = TypicalSectionAeroParams(
    Cl_alpha=2 * np.pi,
    Cl_beta=0.2,
    Cm_ac_beta=0.0,
    Cm_beta=0.0,
    S=1.0
)

section = TypicalSection(params, aero_params)
freqs0, modes0 = section.calculate_natural_frequencies()
print("natural frequencies:", freqs0)
print("eigenmodes:\n", modes0)

rho = 1.225
vs = np.linspace(0, 100, 50)

# sweep velocity, tracking each mode by eigenvector continuity so the two
# curves stay smooth instead of flipping (np.linalg.eig does not order its
# output). freqs[i, j] / vecs[i, :, j] belong to the same tracked mode j.
freqs = np.zeros((len(vs), 2))
vecs = np.zeros((len(vs), 2, 2))
prev = None
for i, v in enumerate(vs):
    f, V = section.calculate_aero_natural_frequencies(0.5 * rho * v**2)
    V = V.real / np.linalg.norm(V.real, axis=0)  # unit-normalise columns

    if prev is None:
        order = np.argsort(f)  # start ascending in frequency
    else:
        overlap = np.abs(prev.T @ V)             # [prev_mode, cur_mode]
        order = [int(np.argmax(overlap[j])) for j in range(2)]
        if order[0] == order[1]:
            order = list(np.argsort(f))
    f, V = f[order], V[:, order]

    if prev is not None:                         # keep sign continuous
        for j in range(2):
            if np.dot(prev[:, j], V[:, j]) < 0:
                V[:, j] *= -1

    freqs[i], vecs[i], prev = f, V, V.copy()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
colors = ["tab:blue", "tab:red"]
sc = None
for j in range(2):
    omega = freqs[:, j]     # x
    h = vecs[:, 0, j]       # y
    theta = vecs[:, 1, j]   # z
    ax.plot(omega, h, theta, color=colors[j], alpha=0.4, lw=1,
            label=f"mode {j + 1}")
    sc = ax.scatter(omega, h, theta, c=vs, cmap="viridis", s=10)

ax.set_xlabel("natural frequency $\\omega$ [rad/s]")
ax.set_ylabel("eigenvector h-component")
ax.set_zlabel(r"eigenvector $\theta$-component")
fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="velocity [m/s]")
ax.legend()
fig.tight_layout()
plt.show()