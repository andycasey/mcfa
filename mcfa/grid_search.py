
""" Grid search using a mixture of common factor analyzers. """

import logging
import numpy as np
from .mcfa import MCFA

logger = logging.getLogger(__name__)

def grid_search(trial_n_latent_factors, trial_n_components, X, N_inits=25,
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

    best_models = dict(mml=None, bic=None, ll=None)

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
                lls_ = [model.log_likelihood_ for model in models]
                idx = np.nanargmax(lls_)
                model = models[idx]

                converged[k, j] = True
                ll[k, j] = model.log_likelihood_
                bic[k, j] = model.bic(X, log_likelihood=model.log_likelihood_)
                mml[k, j] = model.message_length(X, log_likelihood=model.log_likelihood_)

                logger.info(f"std. dev. (ll): {np.std(lls_):.2e}")
                logger.info(f"ptp. (ll): {np.ptp(lls_):.2e}")

                if np.any(np.isfinite(ll)):

                    j_bic_best, k_bic_best = best(Js, Ks, bic)
                    logger.info(f"Model with minimum BIC so far has J = {j_bic_best} and K = {k_bic_best}")
                    j_mml_best, k_mml_best = best(Js, Ks, mml)
                    logger.info(f"Model with minimum I so far has J = {j_mml_best} and K = {k_mml_best}")
                    j_ll_best, k_ll_best = best(Js, Ks, -ll)
                    logger.info(f"Model with maximum log-likelihood so far has J = {j_ll_best} and K = {k_ll_best}")

                    if best_models["mml"] is None \
                    or (best_models["mml"].n_latent_factors != j_mml_best \
                        or best_models["mml"].n_components != k_mml_best):
                        best_models["mml"] = model

                    if best_models["bic"] is None \
                    or (best_models["bic"].n_latent_factors != j_bic_best \
                        or best_models["bic"].n_components != k_bic_best):
                        best_models["bic"] = model

                    if best_models["ll"] is None \
                    or (best_models["ll"].n_latent_factors != j_ll_best \
                        or best_models["ll"].n_components != k_ll_best):
                        best_models["ll"] = model                    


    meta = dict(ll=ll, bic=bic, pseudo_bic=pseudo_bic, message_length=mml,
                best_models=best_models)
    return (J_grid, K_grid, converged, meta)#ll, bic, pseudo_bic)


def best(trial_n_latent_factors, trial_n_components, metric, 
         function=np.nanargmin):

    idx = function(metric)
    n_latent_factors = trial_n_latent_factors[idx % metric.shape[1]]
    n_components = trial_n_components[int(idx / metric.shape[1])]
    return (n_latent_factors, n_components)

