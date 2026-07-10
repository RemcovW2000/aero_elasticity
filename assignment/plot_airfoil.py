"""
Plotting utilities for the aeroelastic "typical section" airfoil.

Standard typical-section notation (Theodorsen / Bisplinghoff-Ashley-Halfman
convention). All chordwise positions are nondimensionalized by the
semichord b, measured from mid-chord:

    LE(-b) ------ EA(a*b) ------ mid-chord(0) ------ hinge(c*b) ------ TE(+b)

    b         semichord (full chord = 2b)
    a         elastic axis (EA) location, in semichords from mid-chord
              (a = -1 at the LE, a = +1 at the TE)
    c         flap hinge location, in semichords from mid-chord
    x_theta   CG offset of the whole section from the EA, in semichords
              (x_cg = (a + x_theta) * b)
    x_beta    CG offset of the flap from the hinge, in semichords
              (x_cg,flap = (c + x_beta) * b)

Degrees of freedom:
    h         plunge of the EA, positive DOWN
    theta     pitch about the EA [rad], positive NOSE UP
    beta      flap deflection about the hinge [rad], positive TRAILING EDGE
              DOWN, relative to the main airfoil

These sign conventions follow the classic typical-section literature (e.g.
Bisplinghoff, Ashley & Halfman, "Aeroelasticity"; Fung, "An Introduction to
the Theory of Aeroelasticity"). If a source defines h/theta/beta with the
opposite sign, negate the value before passing it in.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _rotate(points: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an (N, 2) array of points about `center` by `angle` [rad], CCW-positive."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return (points - center) @ R.T + center


def deform_points(points: np.ndarray, is_flap: np.ndarray,
                   a: float, b: float, c: float,
                   h: float = 0.0, theta: float = 0.0, beta: float = 0.0) -> np.ndarray:
    """
    Apply a rigid-body typical-section deformation to a set of undeformed
    points (given in the same units as `a`, `b`, `c`, i.e. actual chordwise
    position, not divided by b).

    `is_flap` is a boolean mask, True for points that belong to the flap
    (aft of the hinge) and therefore also rotate by `beta` about the hinge.

    Order of operations mirrors the physical DOF definitions: beta is
    relative to the main airfoil, so it is applied first; theta then
    rotates the whole assembly (main body + deflected flap) about the EA;
    h then translates everything down.
    """
    points = np.atleast_2d(points).astype(float).copy()
    is_flap = np.atleast_1d(is_flap)
    ea = np.array([a * b, 0.0])
    hinge = np.array([c * b, 0.0])

    if np.any(is_flap):
        points[is_flap] = _rotate(points[is_flap], hinge, -beta)
    points = _rotate(points, ea, -theta)
    points[:, 1] -= h
    return points


def _naca00xx_half_thickness(x_over_chord: np.ndarray, t: float = 0.10) -> np.ndarray:
    """Symmetric NACA 00xx half-thickness distribution, x_over_chord in [0, 1]."""
    xc = np.clip(x_over_chord, 0.0, 1.0)
    return 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                     + 0.2843 * xc ** 3 - 0.1015 * xc ** 4)


def _outline(b: float, c: float, thickness: float, n_main: int = 50, n_flap: int = 20):
    """Undeformed (upper, lower) surface points for the main body and the flap."""
    chord = 2 * b

    x_main = np.linspace(-b, c * b, n_main)
    x_flap = np.linspace(c * b, b, n_flap)

    t_main = _naca00xx_half_thickness((x_main + b) / chord, thickness) * chord / 2
    t_flap = _naca00xx_half_thickness((x_flap + b) / chord, thickness) * chord / 2

    main_upper = np.column_stack([x_main, t_main])
    main_lower = np.column_stack([x_main, -t_main])
    flap_upper = np.column_stack([x_flap, t_flap])
    flap_lower = np.column_stack([x_flap, -t_flap])
    return main_upper, main_lower, flap_upper, flap_lower


def plot_airfoil(b: float, a: float, c: float, x_theta: float | None = None,
                  x_beta: float | None = None, h: float = 0.0, theta: float = 0.0,
                  beta: float = 0.0, thickness: float = 0.10, ax: plt.Axes | None = None,
                  show_undeformed: bool = True, color: str = "tab:blue",
                  label: str | None = None) -> plt.Axes:
    """
    Plot a typical-section airfoil (thin main body + hinged flap) with its
    key reference points (LE, TE, EA, hinge, CG, flap CG), in a deformed
    state given by (h, theta, beta). Pass h=theta=beta=0 for the reference
    (undeformed) shape.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    main_upper, main_lower, flap_upper, flap_lower = _outline(b, c, thickness)

    if show_undeformed:
        for seg in (main_upper, main_lower, flap_upper, flap_lower):
            ax.plot(seg[:, 0], seg[:, 1], color="0.75", lw=1, ls="--", zorder=1)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)

    def deform(seg, is_flap_value):
        is_flap = np.full(len(seg), is_flap_value)
        return deform_points(seg, is_flap, a, b, c, h, theta, beta)

    main_upper_d = deform(main_upper, False)
    main_lower_d = deform(main_lower, False)
    flap_upper_d = deform(flap_upper, True)
    flap_lower_d = deform(flap_lower, True)

    main_outline = np.vstack([main_upper_d, main_lower_d[::-1]])
    flap_outline = np.vstack([flap_upper_d, flap_lower_d[::-1]])
    ax.fill(main_outline[:, 0], main_outline[:, 1], color=color, alpha=0.35, zorder=2)
    ax.fill(flap_outline[:, 0], flap_outline[:, 1], color=color, alpha=0.6, zorder=2)
    ax.plot(main_outline[:, 0], main_outline[:, 1], color=color, lw=1.5, zorder=3)
    ax.plot(flap_outline[:, 0], flap_outline[:, 1], color=color, lw=1.5, zorder=3,
            label=label)

    # reference points, each deformed the same way as the surface it belongs to
    points = {"LE": (np.array([-b, 0.0]), False),
              "TE": (np.array([b, 0.0]), True),
              "EA": (np.array([a * b, 0.0]), False),
              "hinge": (np.array([c * b, 0.0]), False)}
    if x_theta is not None:
        points["CG"] = (np.array([(a + x_theta) * b, 0.0]), False)
    if x_beta is not None:
        points["flap CG"] = (np.array([(c + x_beta) * b, 0.0]), True)

    markers = {"LE": "<", "TE": ">", "EA": "^", "hinge": "v", "CG": "o", "flap CG": "s"}
    for name, (pt, is_flap) in points.items():
        pt_d = deform_points(pt, np.array([is_flap]), a, b, c, h, theta, beta)[0]
        ax.scatter(*pt_d, marker=markers[name], color="k", s=45, zorder=5)
        ax.annotate(name, pt_d, textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


if __name__ == "__main__":
    # quick self-test: reference shape vs. an exaggerated deformed shape
    b, a, c, x_theta, x_beta = 1.0, -0.4, 0.6, 0.2, -0.025
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_airfoil(b, a, c, x_theta, x_beta, ax=ax, color="0.5", label="reference")
    plot_airfoil(b, a, c, x_theta, x_beta, h=0.2, theta=0.3, beta=-0.4,
                 ax=ax, show_undeformed=False, color="tab:red", label="deformed")
    ax.legend()
    ax.set_title("typical-section airfoil: reference vs. deformed")
    plt.show()
