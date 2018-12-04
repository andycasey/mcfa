

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




#R2 = generalized_rotation_matrix(*angles)
"""
layout, blades = clifford.Cl(5) # Five-dimensional clifford algebra.
locals.update(blades)

x = 2*e1 + 4*e2 + 5*e3 + 3*e4 + 6*e5
y = 6*e1 + 2*e2 + 0*e3 + 1*e4 + 7*e5

"""

A = np.array([2, 4, 5, 3, 6])
B = np.array([6, 2, 1, 1, 7])


def reflection(u, n):
    return u - 2 * n * ((n.T @ u) / (n.T @ n))


def nd_rotation(a, b):

    u = np.array(a) / np.linalg.norm(a)
    v = np.array(b) / np.linalg.norm(b)

    N = u.size
    S = reflection(np.eye(N), v + u)
    R = reflection(S, v)

    return R


def ngram(alpha, u, v):

    c, s = np.cos(alpha), np.sin(alpha)

    u = np.atleast_2d(u)
    v = np.atleast_2d(v)

    U = u / np.sqrt(u @ u.T)
    V = v - (u * v.T) * u
    V = V / np.sqrt(v @ v.T)

    N = u.size
    R = np.eye(N) \
      + (v.T @ u - u.T @ v) * s \
      + (u.T @ u + v.T @ v) * (c - 1)

    return (R, U, V)


import numpy as np

# input vectors
v1 = np.array( [1,1,1,1,1,1] )
v2 = np.array( [2,3,4,5,6,7] )

v1, v2 = A, B

# Gram-Schmidt orthogonalization
n1 = v1 / np.linalg.norm(v1)
v2 = v2 - np.dot(n1,v2) * n1
n2 = v2 / np.linalg.norm(v2)

# rotation by pi/2
a = np.pi/2

I = np.identity(n1.size)

R = I + ( np.outer(n2,n1) - np.outer(n1,n2) ) * np.sin(a) + ( np.outer(n1,n1) + np.outer(n2,n2) ) * (np.cos(a)-1)

# check result
print( np.matmul( R, n1 ) )
print( n2 )