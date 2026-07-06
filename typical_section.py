import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class TypicalSectionParams:
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


class TypicalSection:
    def __init__(self, params: TypicalSectionParams):
        self.params = params

    def get_mass_matrix(self)-> np.ndarray:
        """Calculate mass matrix for a typical section."""
        p = self.params
        mass = p.m_airfoil + p.m_flap

        I_t_1 = p.m_airfoil * p.x_theta**2 * p.b**2
        I_t_2 = (p.c-p.a + p.x_beta)**2 * p.b**2 * p.m_flap
        I_t_3 = p.I_flap + p.I_airfoil

        I_theta = I_t_1 + I_t_2 + I_t_3

        I_beta = p.I_flap

        S_theta = p.x_theta * p.b * p.m_airfoil + (p.c-p.a + p.x_beta) * p.b * p.m_flap
        S_beta = p.m_flap * p.x_beta

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
        p=self.params
        return np.diag([p.K_h, p.K_theta, p.K_beta])

    def calculate_natural_frequencies(self)-> np.ndarray:
        """Calculate natural frequencies for a typical section."""
        M = self.get_mass_matrix()
        K = self.get_stifness_matrix()

        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)

        # Natural frequencies are the square roots of the eigenvalues
        natural_frequencies = np.sqrt(np.abs(eigenvalues))
        return natural_frequencies

    def calculate_uncoupled_natural_frequencies(self)-> np.ndarray:
        """Calculate uncoupled natural frequencies for a typical section."""
        p = self.params
        uncoupled_natural_frequencies = np.array([
            np.sqrt(p.K_h / (p.m_airfoil + p.m_flap)),
            np.sqrt(p.K_theta / (p.I_airfoil + p.I_flap + p.m_airfoil * p.x_theta**2 * p.b**2 + (p.c-p.a + p.x_beta)**2 * p.b**2 * p.m_flap)),
            np.sqrt(p.K_beta / (p.I_flap))
        ])
        return uncoupled_natural_frequencies

if __name__ == "__main__":
    params = TypicalSectionParams(
        m_airfoil=1.567,
        m_flap=0.0,
        I_airfoil=1.0,
        I_flap=0.01,
        b=0.127,
        a=-0.5,
        c=0.5,
        x_theta=-0.5,
        x_beta=0.1,
        K_h=2818.8,
        K_theta=37.3,
        K_beta=0.0,
    )

    section = TypicalSection(params)
    print(section.calculate_natural_frequencies())

    print(section.calculate_uncoupled_natural_frequencies())