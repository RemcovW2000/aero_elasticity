# animation of the purely harmonic flutter motion x(t) = Re(v_f exp(i omega_f t));
# not part of the report
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from state_space import flutter_speed
from flutter_analysis import flutter_mode, FIG_DIR
from table_properties import a, b, c, x_theta, x_beta
from plot_airfoil import plot_airfoil

scaling = 0.5
V_f = flutter_speed()
lam_f, v_f = flutter_mode(V_f)
omega_f = lam_f.imag

n_frames = 60
fig, ax = plt.subplots(figsize=(8, 4.5))


def draw(frame):
    phase = 2 * np.pi * frame / n_frames
    ax.clear()
    h_p, theta_p, beta_p = np.real(v_f * np.exp(1j * phase)) * scaling
    plot_airfoil(b, a, c, x_theta, x_beta,
                 h=h_p * b, theta=theta_p, beta=beta_p,
                 ax=ax, show_undeformed=True, show_markers=True,
                 color="tab:green")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.9, 0.9)
    ax.set_title(f"flutter mode at $V_f$ = {V_f:.0f} m/s, "
                 f"$\\omega_f$ = {omega_f:.1f} rad/s, "
                 f"$\\omega_f t$ = {np.degrees(phase):5.0f}$^\\circ$")


anim = FuncAnimation(fig, draw, frames=n_frames)
anim.save(FIG_DIR / "flutter_mode.gif", writer=PillowWriter(fps=20), dpi=90)
print("saved flutter_mode.gif")
