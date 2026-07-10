"""
Parameters from table 1 of paper.
"""
import numpy as np

omega_h = 50 # rad/s
omega_theta = 100 # rad/s
omega_beta = 300 # rad/s
mu = 40

a = -0.4
b = 1.0
c = 0.6

x_theta = 0.2
x_beta = 0.025

r_theta_sq = 0.25
r_beta_sq = 0.00625

# not from paper:
m_s = 1

M_matrix = m_s * b**2 * np.array(
    [
        [1, x_theta, x_beta],
        [x_theta, r_theta_sq, r_beta_sq + x_beta*(c-a)],
        [x_beta, r_beta_sq + x_beta*(c-a), r_beta_sq],
    ]
)

K_matrix = m_s * b**2 * np.diag([omega_h**2, omega_theta**2, omega_beta**2])

# find eigenfrequencies:
# Solve the generalized eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M_matrix) @ K_matrix)
print("eigenvalues:", eigenvalues)