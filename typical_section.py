import dataclasses
from socket import send_fds
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


@dataclasses.dataclass(frozen=True)
class TypicalSectionDynamicParams:
    a: float
    x_theta: float
    x_beta: float
    c: float
    b: float

    m_airfoil: float
    m_flap: float

    I_airfoil: float
    I_flap: float

    K_h: float
    K_theta: float
    K_beta: float

@dataclasses.dataclass(frozen=True)
class TypicalSectionAeroParams:
    Cl_alpha: float
    Cl_beta: float
    Cm_ac_beta: float
    Cm_beta: float

    S: float
    Cm_ac: float = 0.0


class TypicalSection:
    def __init__(self, dynamic_params: TypicalSectionDynamicParams, aero_params: TypicalSectionAeroParams | None = None):
        self.dynamic_params = dynamic_params

        self.aero_params = aero_params

    def get_mass_matrix(self)-> np.ndarray:
        """Calculate mass matrix for a typical section."""
        p = self.dynamic_params
        mass = p.m_airfoil + p.m_flap

        I_t_1 = p.m_airfoil * p.x_theta**2 * p.b**2
        I_t_2 = (p.c-p.a + p.x_beta)**2 * p.b**2 * p.m_flap
        I_t_3 = p.I_flap + p.I_airfoil

        I_theta = I_t_1 + I_t_2 + I_t_3

        # I_flap is about the flap CG -> shift to the hinge (parallel axis)
        I_beta = p.I_flap + p.m_flap * p.x_beta**2 * p.b**2

        S_theta = p.x_theta * p.b * p.m_airfoil + (p.c-p.a + p.x_beta) * p.b * p.m_flap
        S_beta = p.m_flap * p.x_beta * p.b * p.b

        m_2_3 = I_beta + (p.c-p.a) * p.b * S_beta
        return np.array(
            [
                [mass, S_theta, S_beta],
                [S_theta, I_theta, m_2_3],
                [S_beta, m_2_3, I_beta],
            ]
        )

    def get_stifness_matrix(self)-> np.ndarray:
        """Calculate stifness matrix for a typical section."""
        p=self.dynamic_params
        return np.diag([p.K_h, p.K_theta, p.K_beta])

    def calculate_natural_frequencies(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate natural frequencies and eigenmodes for a typical section.

        Returns
        -------
        natural_frequencies : (n,) array, natural_frequencies[j] for mode j
        eigenvectors : (n, n) array, eigenvectors[:, j] is the mode shape for
            natural_frequencies[j]
        """
        M = self.get_mass_matrix()
        K = self.get_stifness_matrix()

        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)

        # Natural frequencies are the square roots of the eigenvalues
        natural_frequencies = np.sqrt(np.abs(eigenvalues))
        return natural_frequencies, eigenvectors

    def calculate_uncoupled_natural_frequencies(self)-> np.ndarray:
        """Calculate uncoupled natural frequencies for a typical section."""
        p = self.dynamic_params
        uncoupled_natural_frequencies = np.array([
            np.sqrt(p.K_h / (p.m_airfoil + p.m_flap)),
            np.sqrt(p.K_theta / (p.I_airfoil + p.I_flap + p.m_airfoil * p.x_theta**2 * p.b**2 + (p.c-p.a + p.x_beta)**2 * p.b**2 * p.m_flap)),
            np.sqrt(p.K_beta / (p.I_flap + p.m_flap * p.x_beta**2 * p.b**2))
        ])
        return uncoupled_natural_frequencies

    def get_aero_stifness_tilde(self) -> np.ndarray:
        """
        Calculate aero stiffness matrix for a typical section.
        """
        p_a = self.aero_params
        p_d = self.dynamic_params
        return np.array(
            [
                [0, -p_a.S * p_a.Cl_alpha],
                [0, p_a.S * p_a.Cl_alpha * (0.5 + p_d.a) * p_d.b],
            ]
        )

    def calculate_aero_natural_frequencies(self, q: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate natural frequencies and eigenmodes for a typical section with
        aerodynamic loads.

        Calculate a reduced system -> only taking into account theta and h as degrees
        of freedom.

        Returns
        -------
        natural_frequencies : (2,) array, natural_frequencies[j] for mode j
        eigenvectors : (2, 2) array, eigenvectors[:, j] = [h, theta] mode shape
            for natural_frequencies[j]
        """
        M = self.get_mass_matrix()[:2, :2]
        K = self.get_stifness_matrix()[:2, :2]
        K_tilde = self.get_aero_stifness_tilde()

        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ (K - q*K_tilde))

        # Natural frequencies are the square roots of the eigenvalues
        natural_frequencies = np.sqrt(np.abs(eigenvalues))
        return natural_frequencies, eigenvectors

    def calculate_aero_loads_at_position(self, q, x_0: np.ndarray) -> np.ndarray:
        """
        Calculate (external) aerodynamic loads for a typical section at given shape
        using relations found in lecture 3.

        x_0 = np.array([[h], [theta], [beta]])
        returns external force:
        F_ext = np.array([[F_z], [M_theta]]
        """
        p_a = self.aero_params
        p_d = self.dynamic_params

        # Calculate the external aerodynamic load not dependent on theta and h
        force_for_start_deformation = q * x_0 * np.array([[-p_a.S * p_a.Cl_alpha],
                                                          [p_a.S * p_a.Cl_alpha * (
                                                                      0.5 + p_d.a) * p_d.b]])
        beta = x_0[2]
        force_due_to_beta = q * beta * np.array([[p_a.S * p_a.Cl_beta],
                                              [p_a.S * p_a.Cl_beta * (
                                                          0.5 + p_d.a) * p_d.b + p_a.S * p_a.Cm_ac_beta * 2 * p_d.b]])
        constant = q * np.array([[0], [p_a.S * p_a.Cm_ac * 2 * p_d.b]])

        return force_for_start_deformation + force_due_to_beta + constant

    def calculate_monolithic_static_solution_x_f(self, q: float, external_load: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the monolithic static solution for a typical section at applied load.

        Use relations: (K_structural - qK_tilde_aero)x = F_a_0, F_a_1 = F_a_0 + q*K_tilde_aero
        F_a_0 is external aerodynamic load not dependent on theta and h.

        Returns a tuple of (deformation, force)
        """
        # Calculate the aero stiffness matrix
        K_tilde = self.get_aero_stifness_tilde()

        # Calculate the structural stiffness matrix
        K = self.get_stifness_matrix()[:2, :2]

        # Solve for the deformation vector
        deformation_vector = np.linalg.solve(K - q*K_tilde, external_load)
        F_a_1 = external_load + q*K_tilde@deformation_vector
        return deformation_vector, F_a_1

    def calculate_elastic_deformation(self, F_ext: np.ndarray) -> np.ndarray:
        """
        Calculate the deflection for a typical section at given aerodynamic load.
        """
        K = self.get_stifness_matrix()[:2, :2]
        x = np.linalg.solve(K, F_ext)
        return x

    def calculate_iterative_static_solution_x_f(self, q: float, external_load: np.ndarray, tol: float = 1e-10, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the monolithic static solution for a typical section at applied load using an iterative approach.

        Use relations: (K_structural - qK_tilde_aero)x = F_a_0, F_a_1 = F_a_0 + q*K_tilde_aero
        F_a_0 is external aerodynamic load not dependent on theta and h.

        Returns a tuple of (deformation, force)
        """
        initial_deformation = np.linalg.solve(self.get_stifness_matrix()[:2, :2], external_load)
        deformation_vector = initial_deformation
        for i in range(max_iter):
            external_load = external_load + q*self.get_aero_stifness_tilde()@deformation_vector

            new_deformation_vector = np.linalg.solve(self.get_stifness_matrix()[:2, :2], external_load)

            # Check for convergence
            if np.linalg.norm(new_deformation_vector - deformation_vector) < tol:
                break

            deformation_vector = new_deformation_vector

        return deformation_vector, external_load