
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
    // For mixing weights
    simplex[K] theta;

    // Component means in latent space
    vector[J] xi[K];

    // Ingredients for component covariance matrices in latent space
    cholesky_factor_corr[J] OmegaCorr[K];
    vector<lower=0>[J] OmegaDiag[K];

    // Ingredients for latent factors
    vector[M] BetaLowerTriangular;
    positive_ordered[J] BetaDiagonal; 
    real<lower=0> LSigma;

    // Specific (unique) variance
    vector<lower=0>[D] psi;
}

transformed parameters {
    vector[D] mu[K];
    vector[K] log_theta = log(theta);

    cholesky_factor_cov[D] CholeskySigma[K];

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

    // Log mix
    for (n in 1:N) {
        vector[K] lps;

        for (k in 1:K)
            lps[k] = log_theta[k] 
                   + multi_normal_cholesky_lpdf(y[n] | mu[k], CholeskySigma[k]);

        target += log_sum_exp(lps);
    }
}
