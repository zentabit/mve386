from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import math
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from bo_common import *
from scipy.stats import entropy
import landscape

import csv
import time

# Benchmarks a specific aquisition function for some given set of parameters
# Note that this only benchmarks for the case of increasing number of sampled data points

# Den här är otroligt ineffektiv
# Eftersom jag i praktiken "kör om" den varje gång jag behvöer bryta för att räkna ut entropin
# Vill minnas att det går att manuellt göra alla stegen så bör gå att baka in den beräkningen "på vägen"
# Så den beräknar ex. 10 pktr => entropi => 10 till  => entropi => ...
# I nuläget gör den 0-10 => entropi => 0-20 => entropi => ...
# Detta är ju snabbt långsamt men fungerar tills jag löser det bättre

def benchmark(
    x,
    X,
    iter_max: int,
    aq: acquisition.AcquisitionFunction,
    pbounds,
    exact_f,
    exact_Z,
    nu,
    alpha,
    iter_repeats:int = 1,
    init_points:int = 0,
    iter_step_size : int = 10 ):
    
    def extract_mu(optimizer, x):
        xs = [x for _ in range(landscape.nin)]
        Xgrid = np.meshgrid(*xs)
        
        out = np.transpose(np.vstack([X.ravel() for X in Xgrid]))
        mu, sigma = posterior(optimizer, out)
        mu = np.reshape(mu, np.shape(X[0]))
        # sigma = np.reshape(sigma, np.shape(X))

        return mu
    
    N = [(i+1)*iter_step_size for i in range(math.floor(iter_max/iter_step_size))] 
    
    avg_entropy = [0 for _ in range(math.floor(iter_max/iter_step_size))]
    
    c = 0
    
    
    
    
    # Sekvens av punkter i steg
    for n in N:
        for i in range(iter_repeats):
            optimizer = BayesianOptimization(
                f = exact_f,
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
            optimizer.maximize(init_points=0, n_iter=n)
            mu = extract_mu(optimizer, x)
            h_reg = entropy(exact_Z.flatten(), np.abs(mu).flatten())
            
            avg_entropy[c] += h_reg
            
            
        avg_entropy[c] /= iter_repeats
        print(f"{n}: Entropy of regression: {avg_entropy[c]}")
        c += 1
    
    aq_name = type(aq).__name__
    
    fname = f"{landscape.nin}D-{aq_name}-{iter_repeats}repeats-{time.time()}.csv"
    
    with open(fname, "w", newline='') as csvfile:
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

    
    
    acqf = acquisition.ExpectedImprovement(xi = 6)
    nu = 1.5
    alpha = 1e-3

    nSamples = 300
    
    benchmark(x, X, nSamples, acqf, pbounds, f, Z, nu, alpha,iter_repeats=3, init_points=3 )
    

if __name__ == '__main__':
    main()