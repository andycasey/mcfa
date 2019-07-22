

# Experiment 1 missing not at random.


def non_random_missing_mask(N, D, f_total=0.1, f_per_dimension=None):

    missing = np.zeros((N, D), dtype=bool)

    if f_per_dimension is None:
        f_per_dimension = np.random.uniform(0.1, 0.7, size=D)


    M = int(f_total * N * D) # number of missing entries.
    M_per_D = int(M*f_per_dimension/np.sum(f_per_dimension))

    for d, M_d in enumerate(M_per_D):
        idx = np.random.choice(N, M_d, replace=False)
        missing[idx, d] = True

    return missing








