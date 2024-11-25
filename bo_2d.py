from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import landscape
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import entropy, gamma
import sampling_randUnif
import test_functions
from acquisitionfunctions import *

batch_sz = 3 # batch size in LHS
landscape.nin = 2
landscape.peakedness = 100 # set the peakedness to get more extremes
mus, covs = landscape.gen_gauss(5, 2, 1) # fix an f throughout the run

def f(x,y):
    # return test_functions.trough2d(x)
    return landscape.f_sca(np.dstack((x,y)), mus, covs)

def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def presample_lh(npoints, optimizer): # Crate a LHS and update the optimizer
    lh = LatinHypercube(landscape.nin)
    xs = lh.random(npoints)

    for x in xs:
        optimizer.register(x, f(x[0], x[1]))

def presample_unif(npoints, optimizer): # Sample uniformly and update the optimizer
    xs = sampling_randUnif.randUnifSample(landscape.nin, npoints)

    for x in xs:
        optimizer.register(x, f(x[0], x[1]))

# Some acquisition functions
# acqf = acquisition.UpperConfidenceBound(kappa=10) 
# acqf = CB(beta=0, kappa=1)
# acqf = GP_UCB_2()
acqf = RGP_UCB(theta = 3)
# acqf = acquisition.ExpectedImprovement(xi = 10)

# Set opt bounds and create target
pbounds = {'x': (0,1), 'y': (0,1)}
x = np.arange(0,1,0.001).reshape(-1,1)
y = np.arange(0,1,0.001).reshape(-1,1)
z = f(x,y)
# landscape.plot2d(mus, covs)

# This is just a dummy for unif sampling
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
    verbose = 0,
    random_state=0
)

presample_unif(12, optimizer)
optimizer.maximize(init_points=0, n_iter=1) # by optimising once, we get a nice posterior

# print(f"Entropy of unif search: {entropy(y, np.abs(mu))}")

# This is the real run
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
    verbose = 0,
    random_state=0
)

presample_lh(batch_sz, optimizer)
optimizer.maximize(init_points=0, n_iter=10)

# mu = plot_gp(optimizer, x, y)
# print(f"Entropy of regression: {entropy(y, np.abs(mu))}")

# plt.show()


