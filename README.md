Mixture of common factor analysers
----------------------------------

This is a Python 3 implementation of an exact method for modelling multivariate 
data with orthonormal latent factors and clustering in latent space. 
The latent space clustering is performed by partial association of multivariate
normal distributions. 
Optimisation is performed by the expectation-maximization procedure outlined in
[Baek, McLachlan, & Flack (2010)](https://ieeexplore.ieee.org/document/5184847/).


Installation
------------

To install from source:

```
    git clone https://github.com/andycasey/mcfa.git
    cd mcfa/
    python setup.py install
```
