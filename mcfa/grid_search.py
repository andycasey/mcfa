
""" Grid search using a mixture of common factor analyzers. """

import logging
import numpy as np
from .mcfa import MCFA

logger = logging.getLogger(__name__)

def grid_search(trial_n_latent_factors, trial_n_components, X, N_inits=1,
                suppress_exceptions=True, mcfa_kwds=None, pseudo_bic_kwds=None):

    Js = np.array(trial_n_latent_factors)
    Ks = np.array(trial_n_components)

    assert np.all(Js > 0)
    assert np.all(Ks > 0)

    mcfa_kwds = mcfa_kwds or dict()
    pseudo_bic_kwds = pseudo_bic_kwds or dict()
    
    J_grid, K_grid = np.meshgrid(Js, Ks)

    shape = J_grid.shape
    converged = np.zeros(shape, dtype=bool)
    ll = np.nan * np.ones(shape)
    bic = np.nan * np.ones(shape)
    mml = np.nan * np.ones(shape)
    pseudo_bic = np.nan * np.ones(shape)

    for j, J in enumerate(Js):
        for k, K in enumerate(Ks):

            models = []
            for n in range(N_inits):

                print(f"At J = {J}, K = {K}, N = {n}")

                model = MCFA(n_latent_factors=J, n_components=K, **mcfa_kwds)

                try:
                    model.fit(X)

                except:
                    logger.exception(f"Exception occurred during grid search at "\
                                     f"J = {J}, K = {K}:")
                    if not suppress_exceptions:
                        raise
                    continue

                else:
                    models.append(model)

            if len(models) > 0:
                idx = np.nanargmax([model.log_likelihood_ for model in models])
                model = models[idx]

                ll[j, k] = model.log_likelihood_
                bic[j, k] = model.bic(X)
                mml[j, k] = model.message_length(X)

                if np.any(np.isfinite(ll)):

                    j_bic_best, k_bic_best = best(Js, Ks, bic)
                    logger.info(f"Model with minimum BIC so far has J = {j_bic_best} and K = {k_bic_best}")
                    j_mml_best, k_mml_best = best(Js, Ks, mml)
                    logger.info(f"Model with minimum I so far has J = {j_mml_best} and K = {k_mml_best}")
                    j_ll_best, k_ll_best = best(Js, Ks, -ll)
                    logger.info(f"Model with maximum log-likelihood so far has J = {j_ll_best} and K = {k_ll_best}")


    # Best of each?
    metrics = dict(ll=ll, bic=bic, pseudo_bic=pseudo_bic, message_length=mml)
    return (J_grid, K_grid, converged, metrics)#ll, bic, pseudo_bic)


def best(trial_n_latent_factors, trial_n_components, metric, 
         function=np.nanargmin):

    idx = function(metric)
    n_latent_factors = trial_n_latent_factors[idx % metric.shape[1]]
    n_components = trial_n_components[int(idx / metric.shape[1])]
    return (n_latent_factors, n_components)

