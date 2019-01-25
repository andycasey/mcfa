
""" Grid search using a mixture of common factor analyzers. """

import logging
import numpy as np
from .mcfa import MCFA

logger = logging.getLogger(__name__)

def grid_search(trial_n_latent_factors, trial_n_components, X, N_inits=1,
                mcfa_kwds=None, pseudo_bic_kwds=None):

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
                    continue

                else:
                    models.append(model)

            if len(models) > 0:
                idx = np.nanargmax([model.log_likelihood_ for model in models])
                model = models[idx]

                ll[k, j] = model.log_likelihood_
                bic[k, j] = model.bic(X)
                mml[k, j] = model.message_length(X)
                pseudo_bic[k, j] = model.pseudo_bic(X, **pseudo_bic_kwds)
                converged[k, j] = True


    # Best of each?
    metrics = dict(ll=ll, bic=bic, pseudo_bic=pseudo_bic, message_length=mml)
    return (J_grid, K_grid, converged, metrics)#ll, bic, pseudo_bic)


def best(trial_n_latent_factors, trial_n_components, metric, 
         function=np.nanargmin):

    idx = function(metric)
    n_latent_factors = trial_n_latent_factors[idx % metric.shape[1]]
    n_components = trial_n_components[int(idx / metric.shape[1])]
    return (n_latent_factors, n_components)

