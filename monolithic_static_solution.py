import numpy as np

from typical_section import TypicalSectionDynamicParams, TypicalSectionAeroParams, \
    TypicalSection

params = TypicalSectionDynamicParams(
    m_airfoil=1.567,
    m_flap=0.0,
    I_airfoil=1.0,
    I_flap=0.01,
    b=0.127,
    a=-0.0,
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

q = 0.5*1.225*30**2
static_deformation, static_force = section.monolithic_static_solution_x_f(q, 0, x_0=np.array([[0],[0.01]]))