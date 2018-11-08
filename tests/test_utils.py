

import numpy as np

from mcfa import utils


np.random.seed(42)


def test_latent_factor_rotation(D=15, J=10, noise=0.05):

    # Generate fake factors.
    A = np.random.uniform(-1, 1, size=(D, J))

    # Randomly flip them.
    true_signs = np.sign(np.random.uniform(-1, 1, size=J)).astype(int)

    # Add a little noise to the signs
    m = np.random.normal(true_signs, noise, J)

    # Add a little bias.
    b = np.random.normal(0, noise, J)

    A_prime = m * A + b

    # Re-order them.
    true_indices = np.random.choice(J, J, replace=False)
    A_prime = A_prime[:, true_indices]

    R = utils.latent_factor_rotation_matrix(A_prime, A)

    # Check order.
    nonzero = (R != 0)
    _, inferred_indices = np.where(nonzero)

    # Check flips.
    assert np.alltrue(true_indices == inferred_indices)

    inferred_signs = R.T[nonzero.T]
    assert np.alltrue(true_signs == inferred_signs)


def test_latent_factor_rotation_many_times(N=1000, D=15, J=10, noise=0.05):
    
    for i in range(N):
        test_latent_factor_rotation(D=D, J=J, noise=noise)
