from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import math
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from bo_common import *
from scipy.stats import entropy
import landscape
from contextlib import redirect_stdout

import csv
import time

class FunctionDetails:
    def __init__(self, x,X, exact_f, exact_Z):
        self.x = x
        self.mesh_array = X
        self.f = exact_f
        self.Z = exact_Z
        
    def extract_mu(self, optimizer : BayesianOptimization):
        xs = [self.x for _ in range(landscape.nin)]
        Xgrid = np.meshgrid(*xs)
        
        out = np.transpose(np.vstack([X.ravel() for X in Xgrid]))
        mu, sigma = posterior(optimizer, out)
        mu = np.reshape(mu, np.shape(self.mesh_array[0]))
        # sigma = np.reshape(sigma, np.shape(X))

        return mu

def benchmark(
    fd : FunctionDetails,
    iter_max: int,
    aq: acquisition.AcquisitionFunction,
    pbounds,
    nu,
    alpha,
    iter_repeats:int = 1,
    init_points:int = 0,
    iter_step_size : int = 10 ):
    
    N = set([(i+1)*iter_step_size for i in range(math.floor(iter_max/iter_step_size))])
    
    avg_entropy = [0 for _ in range(math.floor(iter_max/iter_step_size))]
    
    
    for _ in range(iter_repeats):
        c = 0
        optimizer = BayesianOptimization(
            f = fd.f,
            pbounds=pbounds,
            acquisition_function=aq,
            verbose = 0,
            random_state=0
        )
        optimizer._gp = GaussianProcessRegressor(
            kernel=Matern(nu=nu),
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=9,
            random_state=optimizer._random_state,
        )

        presample_lh(init_points, optimizer)
        
        for i in range(iter_max+1):
            # optimizer.maximize()
            # {
            next_point = optimizer.suggest()
            target = fd.f(**next_point)
            optimizer.register(params=next_point, target=target)
            # }

            if i in N:
                mu = fd.extract_mu(optimizer)
                h_reg = entropy(fd.Z.flatten(), np.abs(mu).flatten())
                print(f"{i}: Entropy of regression: {h_reg}")
                avg_entropy[c] = h_reg/iter_repeats
                c+= 1
           
    
    aq_name = type(aq).__name__
    
    t = time.time()
    csv_fname = f"points_benchmark-{aq_name}-{t}.csv"
    log_fname = f"points_benchmark-{aq_name}-{t}.log"
    
    with open(log_fname, "w") as file:
        with redirect_stdout(file):
            print(f"===")
            print(f"Time: {t} ")
            print(f"Dimension: {landscape.nin}")
            print(f"# Repeats: {iter_repeats}")
            print(f"Aq: {aq_name}")
            print(f"nu: {nu}")
            print(f"alpha: {alpha}")
            print(f"===")
 
    with open(csv_fname, "w", newline='') as csvfile:
        hnames = ["n_points", "avg_entropy"]
        writer = csv.DictWriter(csvfile, fieldnames=hnames)
        
        writer.writeheader()
        
        for a,b in zip(N, avg_entropy):
            writer.writerow({"n_points":a, "avg_entropy": b})
    
    

landscape.nin = 2
landscape.peakedness = 10 # set the peakedness to get more extremes
mus, covs = landscape.gen_gauss(5, landscape.nin, 1) # fix an f throughout the ru
    
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

def main():

    x = np.arange(0,1,0.01).reshape(-1,1)
    X = np.array(np.meshgrid(*[x for _ in range(landscape.nin)]))
    Z = f_aux(X)
    var_names = [ f"x{i}" for i in range(0, landscape.nin) ]
    pbounds = { var: (0,1) for var in var_names }
    create_function(var_names)

    fd = FunctionDetails(x,X, f, Z)
    
    acqf = acquisition.ExpectedImprovement(xi = 6)
    nu = 1.5
    alpha = 1e-3

    nSamples = 100
    
    benchmark(fd,nSamples, acqf, pbounds, nu, alpha, iter_repeats=1,init_points=3)
    
    

if __name__ == '__main__':
    main()