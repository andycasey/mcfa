
// Latent factor model that assumes homoscedastic noise on the data

data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // dimensionality of the data 
    int<lower=1> J; // number of latent factors
    vector[D] y[N]; // the data
}

transformed data {
    vector[D] mu;   // mean of the data in each dimension
    int<lower=1> M; // number of non-zero loadings

    M = J * (D - J) + choose(J, 2); 
    mu = rep_vector(0.0, D);
}

parameters {
    vector[M] beta_lower_triangular;
    vector<lower=0>[J] beta_diag;
    vector<lower=0>[D] psi;
    real<lower=0> sigma_L;
}

transformed parameters {
    cholesky_factor_cov[D, J] L;
    cholesky_factor_cov[D] L_Sigma;
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

        matrix[D, D] Sigma;
        
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
        Sigma = multiply_lower_tri_self_transpose(L);
        for (i in 1:D)
            Sigma[i, i] = Sigma[i, i] + psi[i];
        L_Sigma = cholesky_decompose(Sigma);
    }
}

model {
    // priors
    beta_lower_triangular ~ normal(0, sigma_L);
    sigma_L ~ normal(0, 1);
    psi ~ normal(0, 1);

    // Priors for diagonal entries to remain ~orthogonal and order invariant 
    // (Leung and Drton 2016)
    for (i in 1:J)
        target += (J - i) * log(beta_diag[i]) - 0.5 * beta_diag[i]^2 / sigma_L;

    y ~ multi_normal_cholesky(mu, L_Sigma);
}