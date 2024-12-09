from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import landscape
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy, gamma
import test_functions
from bo_common import *
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import random

import csv
import time

batch_sz = 3 # batch size in LHS
landscape.nin = 3
landscape.peakedness = 10 # set the peakedness to get more extremes
mus, covs = landscape.gen_gauss(5, landscape.nin, 1) # fix an f throughout the run

def f_aux(X):
    # return test_functions.trough2d(x)
    return landscape.f_sca(np.moveaxis(X,0,landscape.nin), mus, covs)

def create_function(arg_names):
    # Create a string defining the function with the required signature
    args = ", ".join(arg_names)
    func_def = f"""
def f({args}):
    return landscape.f_sca(({args}), mus, covs)
"""
    # Execute this string in the global namespace
    exec(func_def, globals())

def extract_mu(optimizer, x):
    xs = [x for _ in range(landscape.nin)]
    Xgrid = np.meshgrid(*xs)
    
    out = np.transpose(np.vstack([X.ravel() for X in Xgrid]))
    mu, sigma = posterior(optimizer, out)
    mu = np.reshape(mu, np.shape(X[0]))
    # sigma = np.reshape(sigma, np.shape(X))

    return mu

# Some acquisition functions
# acqf = acquisition.UpperConfidenceBound(kappa=10) 
# acqf = CB(beta=0, kappa=1)
# acqf = GP_UCB_2()
acqf = RGP_UCB(theta = 3)
# acqf = acquisition.ExpectedImprovement(xi = 6)
acqd = dummy_acqf()

# Set opt bounds and create target
var_names = [ f"x{i}" for i in range(0, landscape.nin) ]
pbounds = { var: (0,1) for var in var_names }
create_function(var_names)
x = np.arange(0,1,0.01).reshape(-1,1)
X = np.array(np.meshgrid(*[x for _ in range(landscape.nin)]))
Z = f_aux(X)
npts = 100
nu = 1.5
alpha = 1e-3

number_of_batch_sizes = 10
total_points = 343
runs_loop = 1
average_entropy = np.zeros(number_of_batch_sizes)
for n in range(number_of_batch_sizes):
    for m in range(runs_loop):
        
        optimizer = BayesianOptimization(
            f = None,
            pbounds=pbounds,
            acquisition_function=acqf,
            verbose = 0,
            random_state=0,
            allow_duplicate_points=True
        )
        optimizer._gp = GaussianProcessRegressor(
            kernel=Matern(nu=nu),
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=9,
            random_state=optimizer._random_state,
            )

        batches = int(np.floor(n+1/10*total_points))
        batch_size = int(np.floor(total_points/batches))
        batches = batches - 1
        # if j == 0:
        

        next_target = np.empty((batch_size,landscape.nin),dtype=dict)
        value = np.zeros(batch_size)


        # nt = optimizer.suggest()
        # point = f(**nt) # TODO: This should be a hypercube
        # optimizer.register(nt,point)

        presample_lh(batch_size,optimizer,f)

        nt = optimizer.suggest()

        comb = np.dstack((X))

        for i in range(batches):
            optimizer._gp.fit(optimizer.space.params, optimizer.space.target)
            acu = -1 * optimizer.acquisition_function._get_acq(gp = optimizer._gp)(comb)
            total_sum = np.sum(acu)
            weights = [value / total_sum for value in acu]
            # Kan vara intressant att ändra vikterna under körningen för att ha mer exploraion eller explotation
            # weights = [x**5 for x in weights]
            for j in range(batch_size):
                chosen_index = random.choices(range(len(acu)), weights=weights, k=1)[0]
                next_target[j,:] = np.unravel_index(chosen_index, X.shape[1:], order='F')
                next_target[j,:] = next_target[j,:]/np.max(X.shape) # Kan behöva dubbelkolla att x1 = x1
                value[j] = f(*next_target[j,:])
            for k in range(batch_size):
                optimizer.register(params=next_target[k],target=value[k])

        mu = extract_mu(optimizer, x)
        average_entropy[n] = average_entropy[n] + entropy(Z.flatten(), np.abs(mu).flatten())

average_entropy = average_entropy/runs_loop
print(average_entropy)