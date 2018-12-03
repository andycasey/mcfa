
import os
import logging
import pickle

import pystan as stan
import pystan.plots as plots

__all__ = ["load_model", "sampling_kwds", "plots"]


"""
class StanModel(stan.StanModel):

    def __init__(self, *args, **kwargs):
        super(StanModel, self).__init__(self, *args, **kwargs)

    def optimizing(self, *args, **kwargs):
        tick = time()

        print("")
"""

def load_model(path, cached_path=None, recompile=False, overwrite=True):
    r"""
    Load a Stan model from a file. If a cached file exists, use it by default.

    :param path:
        The path of the Stan model.

    :param cached_path: [optional]
        The path of the cached Stan model. By default this will be the same  as
        :path:, with a `.cached` extension appended.

    :param recompile: [optional]
        Recompile the model instead of using a cached version. If the cached
        version is different from the version in path, the model will be
        recompiled automatically.
    """

    cached_path = cached_path or "{}.cached".format(path)

    with open(path, "r") as fp:
        model_code = fp.read()

    while os.path.exists(cached_path) and not recompile:
        with open(cached_path, "rb") as fp:
            model = pickle.load(fp)

        if model.model_code != model_code:
            logging.warn("Cached model at {} differs from the code in {}; "\
                         "recompiling model".format(cached_path, path))
            recompile = True
            continue

        else:
            logging.info("Using pre-compiled model from {}".format(cached_path)) 
            break

    else:
        model = stan.StanModel(model_code=model_code)

        # Save the compiled model.
        if not os.path.exists(cached_path) or overwrite:
            with open(cached_path, "wb") as fp:
                pickle.dump(model, fp)


    return model


def sampling_kwds(**kwargs):
    r"""
    Prepare a dictionary that can be passed to Stan at the sampling stage.
    Basically this just prepares the initial positions so that they match the
    number of chains.
    """

    kwds = dict(chains=4)
    kwds.update(kwargs)

    if "init" in kwds:
        kwds["init"] = [kwds["init"]] * kwds["chains"]

    return kwds

