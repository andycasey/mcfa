
// Mixture of common factor analyzers

data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // dimensionality of the data 
    int<lower=1> J; // number of latent factors
    int<lower=1> K; // number of components
    vector[D] y[N]; // the data
}

transformed data {
    int<lower=1> M; // number of non-zero loadings
    M = J * (D - J) + choose(J, 2); 
}

parameters {
    //simplex[K] theta;
    positive_ordered[K] lambda;
    vector[J] xi[K];
    //cov_matrix[J] omega[K];
    //cholesky_factor_corr[J] Omega_corr[K];
    vector<lower=0>[J] Omega_diag[K];

    vector[M] beta_lower_triangular;
    vector<lower=0>[J] beta_diag;
    vector<lower=0>[D] psi;
    real<lower=0> sigma_L;
}

transformed parameters {
    vector[D] mu[K];
    cov_matrix[D] Sigma[K];
    matrix[J, J] Omega[K];
    simplex[K] theta = lambda / sum(lambda); 

    cholesky_factor_cov[D, J] L;
    {
        /*
        We want to avoid having Sigma and L declared in the global scope of the
        transformed parameters block, otherwise Stan will save traces of these
        parameters for every sample. 

        But if we declare these parameters here then we cannot use constrained
        types like:

            cov_matrix[D] Sigma;

        Which is (probably) more computationally efficient and numerically stable.
        */

        matrix[D, D] eye_psi = diag_matrix(psi);
        
        int idx = 0;
        for (i in 1:D) {
            for (j in (i + 1):J) {
                L[i, j] = 0;
            }
        }
        
        for (j in 1:J) {
            L[j, j] = beta_diag[j];
            for (i in (j + 1):D) {
                idx = idx + 1;
                L[i, j] = beta_lower_triangular[idx];
            }
        }

        for (k in 1:K) {
            mu[k] = L * xi[k];
            //Omega[k] = quad_form_diag(multiply_lower_tri_self_transpose(Omega_corr[k]), Omega_diag[k]);
            Omega[k] = diag_matrix(Omega_diag[k]);
            //Sigma[k] = L * omega[k] * L' + eye_psi;
            Sigma[k] = L * Omega[k] * L' + eye_psi;
        }

    }
}

model {
    vector[K] log_theta = log(theta);

    // priors
    beta_lower_triangular ~ normal(0, sigma_L);
    beta_diag ~ normal(0, 1); // this is used to generate 
    sigma_L ~ normal(0, 1);
    psi ~ normal(0, 1);

    for (k in 1:K) {
        xi[k] ~ normal(rep_vector(0.0, J), rep_vector(1.0, J));
        //Omega_corr[k] ~ lkj_corr_cholesky(1);
        Omega_diag[k] ~ normal(rep_vector(0.0, J),
                               rep_vector(1.0, J)); // this is used to generate

    }

    // Priors for diagonal entries to remain ~orthogonal and order invariant 
    // (Leung and Drton 2016)
    for (i in 1:J)
        target += (J - i) * log(beta_diag[i]) - 0.5 * beta_diag[i]^2 / sigma_L;

    for (n in 1:N) {
        vector[K] lps = log_theta;
        for (k in 1:K)
            lps[k] = lps[k] + multi_normal_lpdf(y[n] | mu[k], Sigma[k]);

        target += log_sum_exp(lps);
    }
}