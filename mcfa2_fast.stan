
// Mixture of common factor analyzers

data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // dimensionality of the data 
    int<lower=1> J; // number of latent factors
    int<lower=1> K; // number of components
    vector[D] y[N]; // the data
}

transformed data {
    int<lower=1> M = J * (D - J) + choose(J, 2); // number of non-zero loadings
}

parameters {
    /*
    We use a positive ordered matrix to prevent component mixture switching.
    This is transformed to a simplex later on.
    */

    // For mixing weights
    //positive_ordered[K] lambda;
    simplex[K] theta;

    // Component means in latent space
    vector[J] xi[K];

    // Ingredients for component covariance matrices in latent space
    cholesky_factor_corr[J] OmegaCorr[K];
    vector[J] OmegaDiag[K];

    // Ingredients for latent factors
    vector[M] BetaLowerTriangular;
    positive_ordered[J] BetaDiagonal; 
    real<lower=0> LSigma;

    // Specific (unique) variance
    vector<lower=0>[D] psi;
}

transformed parameters {
    vector[D] mu[K];
    //simplex[K] theta = lambda / sum(lambda); 
    vector[K] log_theta = log(theta);

    cholesky_factor_cov[D] CholeskySigma[K];

    // TODO: Can we just have A as a cholesky_factor_cov and set a LKJ prior on
    //       the entries which would mimic the current prior we have on the 
    //       elements of the lower triangular?

    cholesky_factor_cov[D, J] A; 
    {
        /*
        We want to avoid having large variables declared in the global scope of 
        the transformed parameters block, otherwise Stan will save traces of the
        parameters for every sample.

        But if we declare these parameters here then we cannot use constrained
        types like:

            cov_matrix[D] Omega;

        Which is (probably) more computationally efficient and numerically stable.
        */

        matrix[J, J] Omega;
        matrix[D, D] eye_psi = diag_matrix(psi);
        
        int idx = 0;
        for (i in 1:D)
            for (j in (i + 1):J)
                A[i, j] = 0;
        
        for (j in 1:J) {
            A[j, j] = BetaDiagonal[J - j + 1];
            for (i in (j + 1):D) {
                idx = idx + 1;
                A[i, j] = BetaLowerTriangular[idx];
            }
        }

        for (k in 1:K) {
            mu[k] = A * xi[k];

            // This is correct:
            Omega = quad_form_diag(
                multiply_lower_tri_self_transpose(OmegaCorr[k]), 
                OmegaDiag[k]);

            //Sigma[k] = A * Omega * A' + eye_psi;
            // TODO: use cholesky decompose.
            CholeskySigma[k] = cholesky_decompose(A * Omega * A' + eye_psi);
        }

    }
}

model {
    // priors
    psi ~ normal(0, 1);
    LSigma ~ normal(0, 1);

    BetaDiagonal ~ normal(0, LSigma); // actually drawn from a half-normal
    BetaLowerTriangular ~ normal(0, LSigma);

    for (k in 1:K) {
        xi[k] ~ normal(rep_vector(0.0, J), rep_vector(1.0, J));
        OmegaCorr[k] ~ lkj_corr_cholesky(LSigma);
        OmegaDiag[k] ~ normal(rep_vector(0.0, J),
                              rep_vector(1.0, J)); // this is used to generate
    }

    // Priors for diagonal entries to remain ~orthogonal and order invariant 
    // (Leung and Drton 2016)
    //for (j in 1:J)
    //    target += (J - j) * log(BetaDiagonal[j]) - 0.5 * BetaDiagonal[j]^2 / LSigma;

    // Log mix
    for (n in 1:N) {
        vector[K] lps;

        for (k in 1:K)
            lps[k] = log_theta[k] 
                   + multi_normal_cholesky_lpdf(y[n] | mu[k], CholeskySigma[k]);

        target += log_sum_exp(lps);
    }
}
/*
generated quantities {
    cov_matrix[J] Omega[K];

    for (k in 1:K)
        Omega[k] = quad_form_diag(
                multiply_lower_tri_self_transpose(OmegaCorr[k]), 
                OmegaDiag[k]);
}
*/