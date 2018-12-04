

import numpy as np
import clifford

layout, blades = clifford.Cl(3) # Three-dimensional clifford algebra

# You really should never actually do this.
locals().update(blades)



def generalized_rotation_matrix(phi, theta, psi):

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(psi), -np.sin(psi)],
        [0, np.sin(psi), np.cos(psi)]
    ])

    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx



def R_euler(phi, theta, psi):

    Rphi = np.exp(-phi/2 * e12)
    Rtheta = np.exp(-theta/2 * e23)
    # TODO: Should this be e12 or e13?
    Rpsi = np.exp(-psi/2 * e12)

    return Rphi * Rtheta * Rpsi

angles = np.random.uniform(0, np.pi, size=3)

R = R_euler(*angles)

A = [e1, e2, e3] # initial ortho-normal frame
B = [R*a*~R for a in A] # resultant frame after rotation

M = np.array([float(b|a) for b in B for a in A]).reshape((3, 3))




R2 = generalized_rotation_matrix(*angles)