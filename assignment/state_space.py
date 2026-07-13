import numpy as np

from table_properties import M_s, K_s
from aero_matrices import M_a, b, eps, roger_matrices


def assemble_A(V):
    """Return the 12x12 aeroelastic system matrix of z_dot = A z, z = [x_dot, x, w]."""
    A0, A1, _, A3, A4 = roger_matrices(V)
    M_ae = M_s - M_a
    C_a = (b / V) * A1                      # total aerodynamic damping
    K_ae = K_s - A0
    W = np.hstack([A3, A4])
    N = np.vstack([np.eye(3), np.eye(3)])
    W0 = (V / b) * np.diag(np.repeat(eps, 3))
    M_inv = np.linalg.inv(M_ae)
    return np.block([
        [M_inv @ C_a,  -M_inv @ K_ae,    M_inv @ W],
        [np.eye(3),    np.zeros((3, 3)), np.zeros((3, 6))],
        [N,            np.zeros((6, 3)), -W0],
    ])


def flutter_speed(V_lo=1.0, V_hi=600.0, tol=1e-3):
    """Return the lowest speed where an eigenvalue of A crosses into the right half-plane."""
    unstable = lambda V: np.linalg.eigvals(assemble_A(V)).real.max() > 0
    while V_hi - V_lo > tol:
        V_mid = (V_lo + V_hi) / 2
        if unstable(V_mid):
            V_hi = V_mid
        else:
            V_lo = V_mid
    return (V_lo + V_hi) / 2


if __name__ == "__main__":
    print(f"flutter speed ~ {flutter_speed():.1f} m/s")
