from bayes_opt import acquisition
import landscape
import numpy as np
from scipy.stats import gamma
from sampling_unifrefine import unifrefine
from scipy.stats.qmc import LatinHypercube
import sampling_randUnif

class CB(acquisition.AcquisitionFunction): # This is like UCB, but we can also set a parameter beta for mean
    def __init__(self, random_state = None, beta = 1, kappa = 1):
        super().__init__(random_state)
        self.beta = beta
        self.kappa = kappa
    
    def base_acq(self, mean, std):
        return self.beta * mean + self.kappa * std
    
class dummy_acqf(acquisition.AcquisitionFunction): # This is like UCB, but we can also set a parameter beta for mean
    def __init__(self, random_state = None):
        super().__init__(random_state)
    
    def base_acq(self, mean, std):
        return np.zeros(np.shape(mean))

class GP_UCB_2(acquisition.AcquisitionFunction): # Using Thm 2
    def __init__(self, random_state = None, delta = 0.1, a = 1, b = 0.2):
        super().__init__(random_state)
        self.delta = delta
        self.b = b
        self.a = a
    
    def base_acq(self, mean, std):
        beta = 2 * np.log2( (self.i+1)**2 * 2 * np.pi**2 / (3 * self.delta) ) + 2 * landscape.nin * np.log2( (self.i + 1)**2 ** landscape.nin * self.b * landscape.d * np.sqrt(np.log2(4 * self.a * landscape.nin / self.delta )) )
        return mean + np.sqrt(beta/5) * std
    
class RGP_UCB(acquisition.AcquisitionFunction):
    def __init__(self, random_state = None, theta = 5.0):
        super().__init__(random_state)
        self.theta = theta
        self.beta = 1.0
    
    def base_acq(self, mean, std):
        return mean + np.sqrt(self.beta) * std
    
    def suggest(self, gp, target_space, n_random = 10000, n_l_bfgs_b = 10, fit_gp = True):
        kappa = np.max([np.log(((self.i+1)**2 + 1)/np.sqrt(2 * np.pi)) / np.log(1 + self.theta/2), 1e-9])
        # print(np.log(((self.i+1)**2 + 1)/np.sqrt(2 * np.pi)))
        self.beta = gamma.rvs(kappa, self.theta)
        # print(self.beta)
        return super().suggest(gp, target_space, n_random, n_l_bfgs_b, fit_gp)
    
class thompson_sampling(acquisition.AcquisitionFunction):
    def __init__(self, random_state=None):
        super().__init__(random_state)
    
    def base_acq(self, y_mean, y_cov):
        assert y_cov.shape[0] == y_cov.shape[1], "y_cov must be a square matrix."
        return self.random_state.multivariate_normal(y_mean, y_cov)
    
    def _get_acq(self, gp, constraint=None):
        if constraint is not None:
            msg = (
                f"Received constraints, but acquisition function {type(self)} "
                + "does not support constrained optimization."
            )
            raise acquisition.ConstraintNotSupportedError(msg)

        # overwrite the base method since we require cov not std
        dim = gp.X_train_.shape[1]
        def acq(x):
            x = x.reshape(-1, dim)
            mean, cov = gp.predict(x, return_cov=True)
            return -1 * self.base_acq(mean, cov)
        return acq
    
    def suggest(self, gp, target_space, n_random=1_000, n_l_bfgs_b=0, fit_gp: bool = True):
        # reduce n_random and n_l_bfgs_b to reduce the computational load
        return super().suggest(gp, target_space, n_random, n_l_bfgs_b, fit_gp)
    
def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma
    
def presample_lh(npoints, optimizer, f): # Create a LHS and update the optimizer
    lh = LatinHypercube(landscape.nin)
    xs = lh.random(npoints)

    for x in xs:
        point = f(*x)
        optimizer.register(x,point)
        # optimizer.probe(x)

def presample_unif(npoints, optimizer): # Sample uniformly and update the optimizer
    xs = sampling_randUnif.randUnifSample(landscape.nin, npoints)
    
    for x in xs:
        optimizer.probe(x)

def presample_unifrefine(refine, optimizer): # Sample in a grid, specified by refine
    xs = np.array(unifrefine(landscape.d, landscape.nin, refine))
    
    for idx in np.ndindex(*(np.shape(xs)[1:])):
        optimizer.probe(xs[(slice(None),) + idx])

def running_unifrefine(refine, optimizer, f): # Sample in a grid, specified by refine
    xs = np.array(unifrefine(landscape.d, landscape.nin, refine))

    for idx in np.ndindex(*(np.shape(xs)[1:])):
        point = f(*xs[(slice(None),) + idx])
        
        optimizer.register(xs[(slice(None),) + idx],point)
        # optimizer.probe(xs[(slice(None),) + idx])        