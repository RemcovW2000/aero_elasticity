"""
Question 4 - Monolithic aeroelastic state-space A matrix.

Assembles the 12x12 system matrix A of  z_dot = A z, with
    z = [ x_dot ; x ; w ],   x = [h/b, theta, beta],   w = 6 lag states.

The aerodynamics are built symbolically in aero_matrices.py (Roger's
approximation of the Theodorsen model) and evaluated numerically here. The
structural matrices follow table 1 of the paper.
"""
import numpy as np

from aero_matrices import (
    a, b, c, rho, PARAMS,
    M_a, C_a, gamma, d0, d1, A0, A1, A2, A3, A4, numeric,
)

# numeric constants
b_val = float(PARAMS[b])
rho_val = float(PARAMS[rho])
a_val = float(PARAMS[a])
c_val = float(PARAMS[c])
eps_val = [0.0455, 0.3]
psi_val = [0.165, 0.335]

# ----------------------------------------------------------------------
# Structural matrices (table 1 of the paper), coordinate x = [h/b, theta, beta]
# ----------------------------------------------------------------------
omega_h, omega_theta, omega_beta = 50.0, 100.0, 300.0   # [rad/s]
mu = 40.0
x_theta, x_beta = 0.2, -0.025
r_theta_sq, r_beta_sq = 0.25, 0.00625

m_s = mu * np.pi * rho_val * b_val**2

M_s = m_s * b_val**2 * np.array([
    [1.0,      x_theta,                            x_beta],
    [x_theta,  r_theta_sq,                         r_beta_sq + x_beta * (c_val - a_val)],
    [x_beta,   r_beta_sq + x_beta * (c_val - a_val), r_beta_sq],
])
K_s = m_s * b_val**2 * np.diag([omega_h**2,
                                r_theta_sq * omega_theta**2,
                                r_beta_sq * omega_beta**2])


# ----------------------------------------------------------------------
# State-space assembly
# ----------------------------------------------------------------------
def assemble_A(V):
    """Return the 12x12 aeroelastic system matrix A at free-stream speed V.

    Lag-state convention: x_a,l = pbar/(pbar+eps_l) x, driven by the velocity,
        x_dot_a,l = x_dot - eps_l (V/b) x_a,l,
    so the lag force is A_{2+l} x_a,l and the aero stiffness only removes A0.
    """
    Ma = numeric(M_a, V)
    A0n, A1n, A3n, A4n = numeric(A0, V), numeric(A1, V), numeric(A3, V), numeric(A4, V)
    Alag = {0: A3n, 1: A4n}

    # effective structural + aero matrices
    M_ae = M_s - Ma
    C_ae = -(b_val / V) * A1n          # = -(C_a + pi rho V b^2 gamma d1^T)
    K_ae = K_s - A0n

    # lag-state dynamics: w_dot = N x_dot + W0 w   (driven by velocity)
    N = np.array([[1, 0, 0], [1, 0, 0],
                  [0, 1, 0], [0, 1, 0],
                  [0, 0, 1], [0, 0, 1]], dtype=float)
    W0 = -(V / b_val) * np.diag([eps_val[0], eps_val[1],
                                 eps_val[0], eps_val[1],
                                 eps_val[0], eps_val[1]])

    # lag force L_f (3x6): column (dof j, pole l) = column j of A_{2+l}
    order = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]  # (dof, pole)
    L_f = np.zeros((3, 6))
    for col, (dof, pl) in enumerate(order):
        L_f[:, col] = Alag[pl][:, dof]

    Minv = np.linalg.inv(M_ae)
    top = np.hstack([-Minv @ C_ae, -Minv @ K_ae, Minv @ L_f])
    mid = np.hstack([np.eye(3), np.zeros((3, 3)), np.zeros((3, 6))])
    bot = np.hstack([N, np.zeros((6, 3)), W0])
    return np.vstack([top, mid, bot])


def flutter_speed(V_lo=1.0, V_hi=600.0, tol=1e-3):
    """Bisection on the lowest speed where max(Re(eig(A))) crosses zero."""
    f = lambda V: np.max(np.linalg.eigvals(assemble_A(V)).real)
    lo, hi = V_lo, V_hi
    if f(lo) > 0:
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Cross-checks
# ----------------------------------------------------------------------
def _cross_checks():
    print("=== cross-checks ===")
    V = 137.0
    Ma = numeric(M_a, V)
    print("M_a symmetric:", np.allclose(Ma, Ma.T))

    from aero_matrices import K_a as K_a_sym
    g = numeric(gamma, V).ravel()
    d0n, d1n = numeric(d0, V).ravel(), numeric(d1, V).ravel()
    S0 = 2 * np.pi * rho_val * V**2 * b_val * np.outer(g, d0n)
    S1 = 2 * np.pi * rho_val * V**2 * b_val * np.outer(g, d1n)
    Ca, Ka = numeric(C_a, V), numeric(K_a_sym, V)
    print("A0 == K_a + S0            :", np.allclose(numeric(A0, V), Ka + S0))
    print("A1 == (V/b)C_a + 0.5 S1   :", np.allclose(numeric(A1, V), (V / b_val) * Ca + 0.5 * S1))
    print("A2 == (V/b)^2 M_a         :", np.allclose(numeric(A2, V), (V / b_val)**2 * Ma))
    print("A3 == psi1(eps1 S1 - S0)  :", np.allclose(numeric(A3, V), psi_val[0] * (eps_val[0] * S1 - S0)))
    print("A4 == psi2(eps2 S1 - S0)  :", np.allclose(numeric(A4, V), psi_val[1] * (eps_val[1] * S1 - S0)))

    A_lowV = assemble_A(1e-4)
    w_ae = np.sort(np.abs(np.linalg.eigvals(A_lowV).imag))[::-1][:6:2]
    w_struct = np.sort(np.sqrt(np.linalg.eigvals(np.linalg.inv(M_s) @ K_s).real))[::-1]
    print("V->0 aeroelastic freqs [rad/s]:", np.round(np.sort(w_ae), 2))
    print("bare structural  freqs [rad/s]:", np.round(np.sort(w_struct), 2))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)

    _cross_checks()

    V_f = flutter_speed()
    print(f"\nflutter speed ~ {V_f:.1f} m/s")
