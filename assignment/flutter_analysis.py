from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from state_space import assemble_A, flutter_speed
from table_properties import a, b, c, x_theta, x_beta
from plot_airfoil import plot_airfoil

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

DISP = slice(3, 6)          # displacement states within z = [x_dot, x, w]
mode_names = ["heave-dominated", "theta-dominated", "beta-dominated"]
mode_colors = ["tab:red", "tab:green", "tab:blue"]


def upper_eigenpairs(A):
    """Return the eigenpairs with Im(lambda) >= 0, one per complex-conjugate pair."""
    lam, vec = np.linalg.eig(A)
    keep = lam.imag >= 0.0
    return lam[keep], vec[:, keep]


def mac(u, v):
    """Return the modal assurance criterion between two complex eigenvectors."""
    return np.abs(u.conj() @ v) ** 2 / (np.abs(u.conj() @ u) * np.abs(v.conj() @ v))


def initial_branches(A):
    """Seed each branch with the oscillatory eigenpair dominated by its DOF."""
    lam, vec = upper_eigenpairs(A)
    osc = np.where(lam.imag > 1.0)[0]
    picks = []
    for dof in range(3):
        score = [np.abs(vec[DISP, i][dof]) / np.linalg.norm(vec[DISP, i]) for i in osc]
        picks.append(next(osc[j] for j in np.argsort(score)[::-1] if osc[j] not in picks))
    return lam[picks], [vec[:, i] for i in picks]


def track_branches(prev_vecs, A):
    """Continue each branch with the best-MAC eigenpair of the new A."""
    lam, vec = upper_eigenpairs(A)
    lam_out, vec_out, taken = [], [], []
    for v_prev in prev_vecs:
        score = [mac(v_prev, vec[:, i]) if i not in taken else -1.0
                 for i in range(len(lam))]
        taken.append(int(np.argmax(score)))
        lam_out.append(lam[taken[-1]])
        vec_out.append(vec[:, taken[-1]])
    return np.array(lam_out), vec_out


def sweep(V_range):
    """Return the tracked eigenvalues (len(V_range) x 3) over the velocity sweep."""
    lams = np.zeros((len(V_range), 3), dtype=complex)
    lams[0], vecs = initial_branches(assemble_A(V_range[0]))
    for i, V in enumerate(V_range[1:], start=1):
        lams[i], vecs = track_branches(vecs, assemble_A(V))
    return lams


def flutter_mode(V_f):
    """Return the flutter eigenvalue and the unit-norm displacement eigenvector at V_f."""
    lam, vec = upper_eigenpairs(assemble_A(V_f))
    osc = np.where(lam.imag > 1.0)[0]
    pick = osc[np.argmin(np.abs(lam[osc].real))]        # closest to the imaginary axis
    v = vec[DISP, pick]
    v *= np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))  # dominant DOF phase to zero
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

    # root locus: the same eigenvalues in the complex plane, sweep ends marked
    i_f = np.argmin(np.abs(V_range - V_f))
    j_f = np.argmax(lams[i_f].real)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for j in range(3):
        ax.plot(lams[:, j].real, lams[:, j].imag, color=mode_colors[j],
                lw=1.5, label=mode_names[j])
        ax.plot(lams[0, j].real, lams[0, j].imag, "o", color=mode_colors[j])
        ax.plot(lams[-1, j].real, lams[-1, j].imag, "s", mfc="none",
                color=mode_colors[j])
    ax.plot(lams[i_f, j_f].real, lams[i_f, j_f].imag, "*", color="k", ms=12)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$ [1/s]")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$ [rad/s]")
    ax.grid(alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", ls="", color="k",
                       label=f"$V$ = {V_range[0]:.0f} m/s"),
                Line2D([], [], marker="s", ls="", mfc="none", color="k",
                       label=f"$V$ = {V_range[-1]:.0f} m/s"),
                Line2D([], [], marker="*", ls="", color="k", ms=10,
                       label=f"$V_f$ = {V_f:.0f} m/s")]
    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "root_locus.pdf")

    # flutter mode shape at three phase instants (half a cycle apart mirrors)
    lam_f, v_f = flutter_mode(V_f)
    print(f"flutter eigenvalue: {lam_f:.3f}  (omega_f = {lam_f.imag:.1f} rad/s)")
    print("flutter eigenvector (h/b, theta, beta):", np.round(v_f, 3))

    scaling = 0.5
    fig, ax = plt.subplots(figsize=(9, 4.5))
    phases = {r"$\omega_f t = 0$": (0.0, "tab:green"),
              r"$\omega_f t = \pi/3$": (np.pi / 3, "tab:purple"),
              r"$\omega_f t = 2\pi/3$": (2 * np.pi / 3, "tab:orange")}
    show_ref = True
    for label, (phase, color) in phases.items():
        h_p, theta_p, beta_p = np.real(v_f * np.exp(1j * phase)) * scaling
        plot_airfoil(b, a, c, x_theta, x_beta,
                     h=h_p * b, theta=theta_p, beta=beta_p,
                     ax=ax, show_undeformed=show_ref, show_markers=True,
                     color=color, label=label)
        show_ref = False

    # de-duplicated legend (marker labels repeat per instant), shapes before points
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    points = ["EA", "hinge", "CG", "flap CG"]
    ordered = [l for l in unique if l not in points] + [p for p in points if p in unique]
    ax.legend([unique[l] for l in ordered], ordered, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_title(f"flutter mode at $V_f$ = {V_f:.0f} m/s ({scaling} * unit eigenvector)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "flutter_mode.pdf")

    plt.show()
