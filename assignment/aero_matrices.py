"""
Question 3 - Roger's approximation of the Theodorsen unsteady aerodynamics.

Symbolic (SymPy) build of the aerodynamic sub-matrices and the Roger matrices
A0..A4, plus a helper to evaluate any of them numerically.

Conventions follow the report (NASA / Karpel):
    displacement  x = [ h/b, theta, beta ]^T
    load          F = [ -L b, M_theta, M_beta ]^T
The plunge coordinate is normalised by the semi-chord b and the vertical force
is scaled by b, which makes M_a symmetric. The free-stream speed V is carried in
place, so C_a scales with V and K_a with V^2.
"""
import numpy as np
import sympy as sp

# ----------------------------------------------------------------------
# Symbols
# ----------------------------------------------------------------------
a, b, c, rho, V = sp.symbols('a b c rho V', real=True)
psi1, psi2, eps1, eps2 = sp.symbols('psi_1 psi_2 varepsilon_1 varepsilon_2', real=True)
T = {n: sp.Symbol(f'T_{n}') for n in (1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13)}
pi = sp.pi
half = sp.Rational(1, 2)

# ----------------------------------------------------------------------
# Theodorsen sub-matrices (V carried in place)
# ----------------------------------------------------------------------
M_a = rho * b**2 * sp.Matrix([
    [-pi * b**2,      pi * a * b**2,              T[1] * b**2],
    [ pi * a * b**2, -pi * b**2 * (sp.Rational(1, 8) + a**2), b**2 * (T[7] + (c - a) * T[1])],
    [ T[1] * b**2,   -2 * T[13] * b**2,           T[3] * b**2 / pi],
])

C_a = rho * b**2 * V * sp.Matrix([
    [0, -pi * b,                T[4] * b],
    [0, -pi * b * (half - a),  -b * (T[1] - T[8] - (c - a) * T[4] + half * T[11])],
    [0, -b * (-2 * T[9] - T[1] + T[4] * (a - half)),  b * T[4] * T[11] / (2 * pi)],
])

K_a = rho * b**2 * V**2 * sp.Matrix([
    [0, 0, 0],
    [0, 0, -(T[4] + T[10])],
    [0, 0, -(T[5] - T[4] * T[10]) / pi],
])

gamma = sp.Matrix([-b, b * (a + half), -b * T[12] / (2 * pi)])
d0 = sp.Matrix([0, 1, T[10] / pi])
d1 = sp.Matrix([1, half - a, T[11] / (2 * pi)])

# ----------------------------------------------------------------------
# Jones' approximation + Roger matrices
# ----------------------------------------------------------------------
qc = 2 * pi * rho * V**2 * b                     # circulatory scale factor
A0 = K_a + qc * gamma * d0.T
A1 = (V / b) * C_a + (1 - psi1 - psi2) * qc * gamma * d1.T
A2 = (V / b)**2 * M_a
A3 = qc * psi1 * gamma * (eps1 * d1 - d0).T
A4 = qc * psi2 * gamma * (eps2 * d1 - d0).T

# ----------------------------------------------------------------------
# Numerical evaluation
# ----------------------------------------------------------------------
PARAMS = {a: -0.4, b: 1.0, c: 0.6, rho: 1.225}
JONES = {psi1: 0.165, psi2: 0.335, eps1: 0.0455, eps2: 0.3}


def theodorsen_values(a_val, c_val):
    """Numeric T_n(a, c) as a substitution dict."""
    ac = np.arccos(c_val)
    s = np.sqrt(1.0 - c_val**2)
    val = {
        1:  -1/3 * (2 + c_val**2) * s + c_val * ac,
        3:  -(1/8 + c_val**2) * ac**2 + 1/4 * c_val * s * ac * (7 + 2*c_val**2)
            - 1/8 * (1 - c_val**2) * (5*c_val**2 + 4),
        4:  -ac + c_val * s,
        5:  -(1 - c_val**2) - ac**2 + 2 * c_val * s * ac,
        7:  1/8 * c_val * (7 + 2*c_val**2) * s - (c_val**2 + 1/8) * ac,
        8:  -1/3 * (1 + 2*c_val**2) * s + c_val * ac,
        9:  1/2 * (1/3 * s**3 + a_val * (-ac + c_val * s)),
        10: s + ac,
        11: (2 - c_val) * s + (1 - 2*c_val) * ac,
        12: (2 + c_val) * s - (2*c_val + 1) * ac,
    }
    val[13] = -1/2 * (val[7] + (c_val - a_val) * val[1])
    return {T[n]: val[n] for n in T}


def subs_dict(V_val, params=None):
    """Full substitution dict at speed V_val."""
    p = dict(PARAMS) if params is None else {**PARAMS, **params}
    a_val, c_val = float(p[a]), float(p[c])
    return {**p, V: V_val, **JONES, **theodorsen_values(a_val, c_val)}


def numeric(mat, V_val, params=None):
    """Evaluate a symbolic matrix/vector to a float numpy array at speed V_val."""
    return np.array(mat.subs(subs_dict(V_val, params)), dtype=np.float64)
