
import numpy as np


        
class MCFA(object):

    def __init__(self, g, q, itmax=500, nkmeans=5, nrandom=5, tol=1e-5,
        init_clust=None, init_para=None, init_method=None, conv_measure="diff",
        warn_messages=True, **kwargs):

        self.g, self.q = (int(g), int(q))
        self.itmax = int(itmax)
        self.nkmeans = int(nkmeans)
        self.nrandom = int(nrandom)
        self.tol = float(tol)

        if self.q < 1:
            raise ValueError("q must be a positive integer")

        if self.g < 1:
            raise ValueError("p must be a positive integer")

        if self.itmax < 1:
            raise ValueError("max number of iterations must be greater than one")

        if self.nkmeans < 1:
            raise ValueError("nkmeans must be a positive integer")

        if self.nrandom < 1:
            raise ValueError("nrandom must be a positive integer")

        if self.tol < 0:
            raise ValueError("tol must be greater than zero")

        self.init_clust = init_clust # TODO:
        self.init_para = init_para # TODO
        self.init_method = _validate_str_input("init_method", init_method,
            ("eigen-A", "rand-A", "gmf", None))

        self.conv_measure = _validate_str_input("conv_measure", conv_measure,
            ("diff", "ratio"))

        self.warn_messages = bool(warn_messages)

        return None



    def fit(self, Y):

        Y = _validate_data_array(Y, self.q)


        raise a




def _validate_data_array(Y, q):

    Y = np.atleast_2d(Y)
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y has non-finite entries")

    n, p = Y.shape

    if p <= q:
        raise ValueError("the number of factors (q) "\
                         "is less than the number of dimensions (p)")
    return Y


def _validate_str_input(descriptor, input_value, acceptable_inputs):

    for acceptable_input in acceptable_inputs:
        if input_value is None and input_value in acceptable_inputs:
            break

        elif input_value is not None \
        and acceptable_input.lower().startswith(input_value.lower()):
            input_value = acceptable_input
            break

    else:
        raise ValueError("{} must be in: {} (not {})".format(
            descriptor, acceptable_inputs, input_value))

    return input_value