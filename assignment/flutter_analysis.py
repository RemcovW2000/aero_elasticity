"""
Questions 5 and 6 - Flutter diagram via the p-method, and the flutter mode.

Sweeps the free-stream speed V, computes the eigenvalues of the 12x12
state-space matrix A(V) from state_space.py at every speed, and plots the
frequency and damping of the three physical (oscillatory) branches.

Because eig() returns eigenvalues in an arbitrary order that changes with
speed, the branches are tracked through the sweep: each branch keeps the
eigenpair whose eigenvector correlates best (modal assurance criterion)
with its eigenvector at the previous speed. The lag-state eigenvalues are
real and their eigenvectors have little displacement content, so they are
never selected.
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from state_space import assemble_A, flutter_speed
from aero_matrices import PARAMS, a, b, c
from plot_airfoil import plot_airfoil

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# state ordering of z = [x_dot, x, w]: displacements are states 3..5
DISP = slice(3, 6)

mode_names = ["heave-dominated", "theta-dominated", "beta-dominated"]
mode_colors = ["tab:red", "tab:green", "tab:blue"]


# ----------------------------------------------------------------------
# Eigensolution helpers
# ----------------------------------------------------------------------
def upper_eigenpairs(A):
    """Eigenpairs with Im(lambda) >= 0 (one per complex-conjugate pair)."""
    lam, vec = np.linalg.eig(A)
    keep = lam.imag >= 0.0
    return lam[keep], vec[:, keep]


def mac(u, v):
    """Modal assurance criterion between two complex eigenvectors."""
    return np.abs(u.conj() @ v) ** 2 / ((np.abs(u.conj() @ u)) * np.abs(v.conj() @ v))


def initial_branches(A):
    """Identify the three structural branches at the first (low) speed.

    At low V the physical eigenvectors are dominated by a single
    displacement DOF, so each branch is seeded with the oscillatory
    eigenpair whose displacement content is concentrated in that DOF.
    """
    lam, vec = upper_eigenpairs(A)
    osc = np.where(lam.imag > 1.0)[0]        # oscillatory candidates only
    picks = []
    for dof in range(3):
        score = [np.abs(vec[DISP, i][dof]) / np.linalg.norm(vec[DISP, i])
                 for i in osc]
        order = np.argsort(score)[::-1]
        pick = next(osc[j] for j in order if osc[j] not in picks)
        picks.append(pick)
    return lam[picks], [vec[:, i] for i in picks]


def track_branches(lam_prev_vecs, A):
    """Continue each branch with the best-MAC eigenpair of the new A."""
    lam, vec = upper_eigenpairs(A)
    lam_out, vec_out, taken = [], [], []
    for v_prev in lam_prev_vecs:
        score = [mac(v_prev, vec[:, i]) if i not in taken else -1.0
                 for i in range(len(lam))]
        pick = int(np.argmax(score))
        taken.append(pick)
        lam_out.append(lam[pick])
        vec_out.append(vec[:, pick])
    return np.array(lam_out), vec_out


# ----------------------------------------------------------------------
# Velocity sweep (p-method)
# ----------------------------------------------------------------------
def sweep(V_range):
    """Return the tracked eigenvalues (len(V_range) x 3) over the sweep."""
    lams = np.zeros((len(V_range), 3), dtype=complex)
    lams[0], vecs = initial_branches(assemble_A(V_range[0]))
    for i, V in enumerate(V_range[1:], start=1):
        lams[i], vecs = track_branches(vecs, assemble_A(V))
    return lams


# ----------------------------------------------------------------------
# Flutter mode
# ----------------------------------------------------------------------
def flutter_mode(V_f):
    """Displacement part of the flutter eigenvector at the flutter speed.

    At V_f the flutter eigenvalue lies on the imaginary axis, lambda = i
    omega_f, so the mode is purely harmonic. The eigenvector is complex:
    the DOFs oscillate at the same frequency but with phase differences,
    x(t) = Re(v exp(i omega_f t)). The vector is normalised to unit norm
    with the phase of the dominant DOF rotated to zero.
    """
    lam, vec = upper_eigenpairs(assemble_A(V_f))
    osc = np.where(lam.imag > 1.0)[0]
    pick = osc[np.argmin(np.abs(lam[osc].real))]     # closest to the imag axis
    v = vec[DISP, pick]
    v = v / v[np.argmax(np.abs(v))] * np.max(np.abs(v))   # dominant DOF real
    return lam[pick], v / np.linalg.norm(v)


if __name__ == "__main__":
    V_f = flutter_speed()
    print(f"flutter speed ~ {V_f:.1f} m/s")

    V_range = np.arange(2.0, 341.0, 2.0)
    lams = sweep(V_range)

    freq = lams.imag                          # [rad/s]
    zeta = -lams.real / np.abs(lams)          # damping ratio [-]

    # flutter diagram: frequency and damping in one figure
    fig, (ax_f, ax_d) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    for j in range(3):
        ax_f.plot(V_range, freq[:, j], color=mode_colors[j], label=mode_names[j])
        ax_d.plot(V_range, zeta[:, j], color=mode_colors[j], label=mode_names[j])
    for ax in (ax_f, ax_d):
        ax.axvline(V_f, color="k", linestyle="--", lw=1, label=f"$V_f$ = {V_f:.0f} m/s")
        ax.grid(alpha=0.3)
    ax_d.axhline(0.0, color="k", lw=0.8)
    ax_f.set_ylabel(r"$\omega = \mathrm{Im}(\lambda)$ [rad/s]")
    ax_f.set_ylim(bottom=0)
    ax_f.legend()
    ax_d.set_ylabel(r"$\zeta = -\mathrm{Re}(\lambda)/|\lambda|$ [-]")
    ax_d.set_xlabel("V [m/s]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "flutter_diagram.pdf")

    # flutter mode shape, plotted at several phase instants of the cycle
    # (instants half a cycle apart are mirror images, so 0..2pi/3 suffices)
    lam_f, v_f = flutter_mode(V_f)
    print(f"flutter eigenvalue: {lam_f:.3f}  (omega_f = {lam_f.imag:.1f} rad/s)")
    print("flutter eigenvector (h/b, theta, beta):", np.round(v_f, 3))

    b_val, a_val, c_val = float(PARAMS[b]), float(PARAMS[a]), float(PARAMS[c])
    x_theta, x_beta = 0.2, -0.025
    scaling = 0.5

    fig, ax = plt.subplots(figsize=(9, 4.5))
    phases = {r"$\omega_f t = 0$": (0.0, "tab:green"),
              r"$\omega_f t = \pi/3$": (np.pi / 3, "tab:purple"),
              r"$\omega_f t = 2\pi/3$": (2 * np.pi / 3, "tab:orange")}
    show_ref = True
    for label, (phase, color) in phases.items():
        h_p, theta_p, beta_p = np.real(v_f * np.exp(1j * phase)) * scaling
        plot_airfoil(b_val, a_val, c_val, x_theta, x_beta,
                     h=h_p * b_val, theta=theta_p, beta=beta_p,
                     ax=ax, show_undeformed=show_ref, show_markers=True,
                     color=color, label=label)
        show_ref = False

    # de-duplicate legend entries (EA/hinge/CG/flap CG markers repeat once per
    # phase instant), shapes ordered before points
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    point_names = ["EA", "hinge", "CG", "flap CG"]
    shape_names = [l for l in unique if l not in point_names]
    ordered = shape_names + [p for p in point_names if p in unique]
    ax.legend([unique[l] for l in ordered], ordered, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_title(f"flutter mode at $V_f$ = {V_f:.0f} m/s ({scaling} * unit eigenvector)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "flutter_mode.pdf")

    plt.show()
