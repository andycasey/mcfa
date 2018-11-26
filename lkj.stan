
data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // dimensionality of the data 
    int<lower=1> J;
    vector[D] y[N]; // the data
    real eta;
}

parameters {
    cholesky_factor_corr[J] L;
}
model {
    L ~ lkj_corr_cholesky(eta);
    for (i in 1:N)
      y[i] ~ normal(0, 1);
}
