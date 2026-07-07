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

    def monolithic_static_solution_x_f(self, q: float, beta: float, x_0: np.ndarray = np.array([0,0])) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the monolithic static solution for a typical section.

        Use relations: (K_structural - qK_tilde_aero)x = F_a_0, F_a_1 = F_a_0 + q*K_tilde_aero
        F_a_0 is external aerodynamic load not dependent on theta and h.

        Returns a tuple of (deformation, force)
        """
        p_a = self.aero_params
        p_d = self.dynamic_params

        # Calculate the external aerodynamic load not dependent on theta and h
        force_for_start_deformation = q*x_0*np.array([-p_a.S*p_a.Cl_alpha, p_a.S*p_a.Cl_alpha*(0.5 + p_d.a) * p_d.b])
        force_due_to_b = q*beta*np.array(
            [
                p_a.S*p_a.Cl_beta,
                p_a.S*p_a.Cl_beta * (0.5 + p_d.a) * p_d.b + p_a.S*p_a.Cm_ac_beta * 2 * p_d.b
            ]
        )
        constant = q*np.array([0, p_a.S * p_a.Cm_ac*2*p_d.b])

        external_aero_force = force_for_start_deformation + force_due_to_b + constant

        # Calculate the aero stiffness matrix
        K_tilde = self.get_aero_stifness_tilde()

        # Calculate the structural stiffness matrix
        K = self.get_stifness_matrix()[:2, :2]

        # Solve for the deformation vector
        deformation_vector = np.linalg.solve(K - q*K_tilde, external_aero_force)
        F_a_1 = external_aero_force + q*K_tilde*deformation_vector
        return deformation_vector, F_a_1

if __name__ == "__main__":
    params = TypicalSectionDynamicParams(
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

    aero_params = TypicalSectionAeroParams(
        Cl_alpha=2 * np.pi,
        Cl_beta=0.2,
        Cm_ac_beta=0.0,
        Cm_beta=0.0,
        S=1.0
    )

    section = TypicalSection(params, aero_params)
    freqs0, modes0 = section.calculate_natural_frequencies()
    print("natural frequencies:", freqs0)
    print("eigenmodes:\n", modes0)

    rho = 1.225
    vs = np.linspace(0, 100, 50)

    # sweep velocity, tracking each mode by eigenvector continuity so the two
    # curves stay smooth instead of flipping (np.linalg.eig does not order its
    # output). freqs[i, j] / vecs[i, :, j] belong to the same tracked mode j.
    freqs = np.zeros((len(vs), 2))
    vecs = np.zeros((len(vs), 2, 2))
    prev = None
    for i, v in enumerate(vs):
        f, V = section.calculate_aero_natural_frequencies(0.5 * rho * v**2)
        V = V.real / np.linalg.norm(V.real, axis=0)  # unit-normalise columns

        if prev is None:
            order = np.argsort(f)  # start ascending in frequency
        else:
            overlap = np.abs(prev.T @ V)             # [prev_mode, cur_mode]
            order = [int(np.argmax(overlap[j])) for j in range(2)]
            if order[0] == order[1]:
                order = list(np.argsort(f))
        f, V = f[order], V[:, order]

        if prev is not None:                         # keep sign continuous
            for j in range(2):
                if np.dot(prev[:, j], V[:, j]) < 0:
                    V[:, j] *= -1

        freqs[i], vecs[i], prev = f, V, V.copy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["tab:blue", "tab:red"]
    sc = None
    for j in range(2):
        omega = freqs[:, j]     # x
        h = vecs[:, 0, j]       # y
        theta = vecs[:, 1, j]   # z
        ax.plot(omega, h, theta, color=colors[j], alpha=0.4, lw=1,
                label=f"mode {j + 1}")
        sc = ax.scatter(omega, h, theta, c=vs, cmap="viridis", s=10)

    ax.set_xlabel("natural frequency $\\omega$ [rad/s]")
    ax.set_ylabel("eigenvector h-component")
    ax.set_zlabel(r"eigenvector $\theta$-component")
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="velocity [m/s]")
    ax.legend()
    fig.tight_layout()
    plt.show()