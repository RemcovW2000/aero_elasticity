import numpy as np

from table_properties import a, b, c, rho

# Jones' approximation coefficients and poles
psi = np.array([0.165, 0.335])
eps = np.array([0.0455, 0.3])


def theodorsen_constants(a, c):
    """Return the geometric constants T_n(a, c) of the Theodorsen model."""
    ac, s = np.arccos(c), np.sqrt(1.0 - c**2)
    T = {
        1:  -(2 + c**2) * s / 3 + c * ac,
        3:  -(1/8 + c**2) * ac**2 + c * s * ac * (7 + 2*c**2) / 4
            - (1 - c**2) * (5*c**2 + 4) / 8,
        4:  -ac + c * s,
        5:  -(1 - c**2) - ac**2 + 2 * c * s * ac,
        7:  c * (7 + 2*c**2) * s / 8 - (c**2 + 1/8) * ac,
        8:  -(1 + 2*c**2) * s / 3 + c * ac,
        9:  (s**3 / 3 + a * (-ac + c * s)) / 2,
        10: s + ac,
        11: (2 - c) * s + (1 - 2*c) * ac,
        12: (2 + c) * s - (2*c + 1) * ac,
    }
    T[13] = -(T[7] + (c - a) * T[1]) / 2
    return T


T = theodorsen_constants(a, c)
pi = np.pi

# non-circulatory matrices; displacement x = [h/b, theta, beta], force [-Lb, M_theta, M_beta]
M_a = rho * b**2 * np.array([
    [-pi * b**2,     pi * a * b**2,                T[1] * b**2],
    [pi * a * b**2,  -pi * b**2 * (1/8 + a**2),    b**2 * (T[7] + (c - a) * T[1])],
    [T[1] * b**2,    -2 * T[13] * b**2,            T[3] * b**2 / pi],
])
C_a_unit = rho * b**2 * np.array([                 # C_a = V * C_a_unit
    [0, -pi * b,                 T[4] * b],
    [0, -pi * b * (1/2 - a),     -b * (T[1] - T[8] - (c - a) * T[4] + T[11] / 2)],
    [0, -b * (-2*T[9] - T[1] + T[4] * (a - 1/2)),  b * T[4] * T[11] / (2 * pi)],
])
K_a_unit = rho * b**2 * np.array([                 # K_a = V^2 * K_a_unit
    [0, 0, 0],
    [0, 0, -(T[4] + T[10])],
    [0, 0, -(T[5] - T[4] * T[10]) / pi],
])

# circulatory load distribution and downwash vectors
gamma = np.array([-b, b * (a + 1/2), -b * T[12] / (2 * pi)])
d0 = np.array([0.0, 1.0, T[10] / pi])
d1 = np.array([1.0, 1/2 - a, T[11] / (2 * pi)])


def roger_matrices(V):
    """Return the Roger matrices A0..A4 of the aerodynamic load at speed V."""
    qc = 2 * pi * rho * V**2 * b
    A0 = V**2 * K_a_unit + qc * np.outer(gamma, d0)
    A1 = (V / b) * V * C_a_unit + (1 - psi.sum()) * qc * np.outer(gamma, d1)
    A2 = (V / b)**2 * M_a
    A3 = qc * psi[0] * np.outer(gamma, eps[0] * d1 - d0)
    A4 = qc * psi[1] * np.outer(gamma, eps[1] * d1 - d0)
    return A0, A1, A2, A3, A4
