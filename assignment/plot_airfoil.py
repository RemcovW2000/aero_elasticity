from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# typical-section conventions: chordwise positions in semichords from mid-chord
# (LE at -b, EA at a*b, hinge at c*b, TE at +b); h is plunge of the EA positive
# down, theta is pitch about the EA positive nose-up, beta is flap deflection
# about the hinge positive trailing-edge down, relative to the main airfoil.


def _rotate(points: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an (N, 2) array of points about `center` by `angle` [rad], CCW-positive."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return (points - center) @ R.T + center


def deform_points(points: np.ndarray, is_flap: np.ndarray,
                  a: float, b: float, c: float,
                  h: float = 0.0, theta: float = 0.0, beta: float = 0.0) -> np.ndarray:
    """Apply the rigid-body deformation (beta about the hinge, then theta about the EA, then plunge h)."""
    points = np.atleast_2d(points).astype(float).copy()
    is_flap = np.atleast_1d(is_flap)
    ea, hinge = np.array([a * b, 0.0]), np.array([c * b, 0.0])

    if np.any(is_flap):
        points[is_flap] = _rotate(points[is_flap], hinge, -beta)
    points = _rotate(points, ea, -theta)
    points[:, 1] -= h
    return points


def _naca00xx_half_thickness(x_over_chord: np.ndarray, t: float = 0.10) -> np.ndarray:
    """Return the symmetric NACA 00xx half-thickness at x_over_chord in [0, 1]."""
    xc = np.clip(x_over_chord, 0.0, 1.0)
    return 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                    + 0.2843 * xc ** 3 - 0.1015 * xc ** 4)


def _outline(b: float, c: float, thickness: float, n_main: int = 50, n_flap: int = 20):
    """Return the undeformed (upper, lower) surface points of the main body and flap."""
    chord = 2 * b
    x_main = np.linspace(-b, c * b, n_main)
    x_flap = np.linspace(c * b, b, n_flap)
    t_main = _naca00xx_half_thickness((x_main + b) / chord, thickness) * chord / 2
    t_flap = _naca00xx_half_thickness((x_flap + b) / chord, thickness) * chord / 2
    return (np.column_stack([x_main, t_main]), np.column_stack([x_main, -t_main]),
            np.column_stack([x_flap, t_flap]), np.column_stack([x_flap, -t_flap]))


def plot_airfoil(b: float, a: float, c: float, x_theta: float | None = None,
                 x_beta: float | None = None, h: float = 0.0, theta: float = 0.0,
                 beta: float = 0.0, thickness: float = 0.10, ax: plt.Axes | None = None,
                 show_undeformed: bool = True, show_markers: bool = True,
                 color: str = "tab:blue", label: str | None = None) -> plt.Axes:
    """Plot the typical-section airfoil deformed by (h, theta, beta), optionally with reference outline and EA/hinge/CG markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    main_upper, main_lower, flap_upper, flap_lower = _outline(b, c, thickness)

    if show_undeformed:
        for seg in (main_upper, main_lower, flap_upper, flap_lower):
            ax.plot(seg[:, 0], seg[:, 1], color="0.75", lw=1, ls="--", zorder=1)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)

    def deform(seg, is_flap_value):
        return deform_points(seg, np.full(len(seg), is_flap_value), a, b, c, h, theta, beta)

    main_outline = np.vstack([deform(main_upper, False), deform(main_lower, False)[::-1]])
    flap_outline = np.vstack([deform(flap_upper, True), deform(flap_lower, True)[::-1]])
    ax.fill(main_outline[:, 0], main_outline[:, 1], color=color, alpha=0.35, zorder=2)
    ax.fill(flap_outline[:, 0], flap_outline[:, 1], color=color, alpha=0.6, zorder=2)
    ax.plot(main_outline[:, 0], main_outline[:, 1], color=color, lw=1.5, zorder=3)
    ax.plot(flap_outline[:, 0], flap_outline[:, 1], color=color, lw=1.5, zorder=3,
            label=label)

    if show_markers:
        points = {"EA": (np.array([a * b, 0.0]), False),
                  "hinge": (np.array([c * b, 0.0]), False)}
        if x_theta is not None:
            points["CG"] = (np.array([(a + x_theta) * b, 0.0]), False)
        if x_beta is not None:
            points["flap CG"] = (np.array([(c + x_beta) * b, 0.0]), True)

        markers = {"EA": "^", "hinge": "v", "CG": "o", "flap CG": "s"}
        for name, (pt, is_flap) in points.items():
            pt_d = deform_points(pt, np.array([is_flap]), a, b, c, h, theta, beta)[0]
            ax.scatter(*pt_d, marker=markers[name], color="k", s=45, zorder=5, label=name)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


if __name__ == "__main__":
    # self-test: reference shape vs an exaggerated deformed shape
    b, a, c, x_theta, x_beta = 1.0, -0.4, 0.6, 0.2, -0.025
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_airfoil(b, a, c, x_theta, x_beta, ax=ax, color="0.5", label="reference",
                 show_markers=False)
    plot_airfoil(b, a, c, x_theta, x_beta, h=0.2, theta=0.3, beta=-0.4,
                 ax=ax, show_undeformed=False, color="tab:red", label="deformed")
    ax.legend()
    plt.show()
