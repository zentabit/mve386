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
    aq_base,
    pbounds,
    n_sample,
    nu,
    alpha,
    param_dict : dict,
    iter_repeat:int = 1,
    init_points:int = 1
    ):
    
    # param_dict is intended to be a dictionary
    # "Parameter name": [start, stop, refinement]
    # Refinement anger steg från start till stop
    # Ex: [5,8,3] => [5,6,7,8] som de värden vi testar
   
    def run(aq):
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

        presample_lh(init_points, optimizer, fd.f)
        optimizer.maximize(init_points=0, n_iter=n_sample)
        mu = fd.extract_mu(optimizer)
        h_reg = entropy(fd.Z.flatten(), np.abs(mu).flatten())
        
        return h_reg
    
    # https://i.kym-cdn.com/entries/icons/original/000/041/943/1aa1blank.png
    
    d = {}
    for key in param_dict.keys():
        start, stop, refinement = param_dict[key]
        
        steps = [start + i *(stop - start)/refinement for i in range(refinement+1)]
        d[key] = steps

    arguments = []

    d_steps = [len(d[k]) for k in d.keys()]
    d_counter = [0 for _ in d.keys()]

    # Massa logik för att ta fram alla kombinationer av parametervärden
    b_loop = True
    while b_loop:
        temp = {}
        i = 0
        for k in d.keys():
            temp[k] = d[k][d_counter[i]]
            i += 1
        
        arguments.append(temp)
        
        d_counter[-1] += 1
        for i in reversed(range(len(d_counter))):
            if d_counter[i] == d_steps[i]:
                if i==0:
                    b_loop = False
                    break
                
                d_counter[i - 1] += 1
                d_counter[i] = 0
    
    # Nu när jag är klar inser jag att alla våra AQ endast har en parameter vi varierat...
    # Men om vi skulle vilja variera två eller fler så fungerar det nog

    avg_entropy = np.zeros((iter_repeat, len(arguments)))

    for j, arg in enumerate(arguments):
        
        h_avg = 0
        
        for i in range(iter_repeat):
            h_avg += run(aq_base(**arg))
            avg_entropy[i,j] = h_avg    
    
        h_avg /= iter_repeat
        
        
        
        print(f"h_reg: {h_avg}, args: {arg}")
    
    
    return arguments, avg_entropy
    

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


def benchMultipleF(n):
    x = np.arange(0,1,0.01).reshape(-1,1)
    X = np.array(np.meshgrid(*[x for _ in range(landscape.nin)]))
    var_names = [ f"x{i}" for i in range(0, landscape.nin) ]
    pbounds = { var: (0,1) for var in var_names }
    create_function(var_names)

    landscape.peakedness = 10 # set the peakedness to get more extremes
    
    
    ent = {} 
    for i in range(n):
        global mus, covs
        mus, covs = landscape.gen_gauss(5, landscape.nin, 1) # fix an f throughout the ru
         
        Z = f_aux(X)

        fd = FunctionDetails(x,X, f, Z)

        
        # Gaussian parameters
        nu = 1.5
        alpha = 1e-3

        
        aq_base = acquisition.ExpectedImprovement
        aq_arg = {"xi":[5,7,2]}
        
        
        iter_max = 30
        iter_repeats = 2
        
        arguments, avg_entropy = benchmark(fd, aq_base, pbounds, iter_max,nu,alpha,aq_arg, iter_repeat=iter_repeats)

        ent[i] = {"arguments": arguments, "entropy_list": avg_entropy}
        
    import json
    from collections import defaultdict

    
    args = ent[0]["arguments"]
    result_dict = defaultdict(dict)
    x = []
    for a in args:
        x.append(str(sum(a.values())))

    e_len = 0

    for function_nr in ent.keys():
        run_data = ent[function_nr]["entropy_list"]
       
        d = defaultdict(list)
   
        for i, entropy_list in enumerate(np.transpose(run_data)):
            d[x[i]]= list(entropy_list)
            
            e_len = len(entropy_list)
    
    
        result_dict[function_nr] = dict(d)
        
    
    result_dict = dict(result_dict)
    
    arr = np.ndarray((len(result_dict.keys()), len(x), e_len))

    for i, (fun_nr, params) in enumerate(result_dict.items()):
        for j, (key, values) in enumerate(params.items()):
            arr[i,j,:] = values
    
    
    t = time.time()
    npy_fname = f"aq_benchmark-{t}.npy"
    log_fname = f"aq_benchmark-{t}.log"
    
    with open(log_fname, "w") as file:
        with redirect_stdout(file):
            print(f"===")
            print(f"Time: {t} ")
            print(f"Dimension: {landscape.nin}")
            print(f"# Repeats: {iter_repeats}")
            print(f"# Functions: {n}")
            print(f"Aq: {aq_base.__name__}")
            print(f"nu: {nu}")
            print(f"alpha: {alpha}")
            print(f"Format: {n}x{len(x)}x{iter_repeats} matrix")
            print(f"Format: #functions x #param. x #repeats")
            print(f"Params: {x}")
            print(f"===")
    
    np.save(npy_fname, arr)
    
    

def main():
    benchMultipleF(2)
    return
    
    x = np.arange(0,1,0.01).reshape(-1,1)
    X = np.array(np.meshgrid(*[x for _ in range(landscape.nin)]))
    var_names = [ f"x{i}" for i in range(0, landscape.nin) ]
    pbounds = { var: (0,1) for var in var_names }
    
    Z = f_aux(X)
    var_names = [ f"x{i}" for i in range(0, landscape.nin) ]
    pbounds = { var: (0,1) for var in var_names }
    create_function(var_names)

    fd = FunctionDetails(x,X, f, Z)
    
    # Gaussian parameters
    nu = 1.5
    alpha = 1e-3

    
    aq_base = acquisition.ExpectedImprovement
    aq_arg = {"xi":[5,7,2]}
    
    
    iter_max = 30
    iter_repeats = 1
    
    arguments, avg_entropy = benchmark(fd, aq_base, pbounds, iter_max,nu,alpha,aq_arg, iter_repeat=iter_repeats)
    
    
    
    t = time.time()
    csv_fname = f"aq_benchmark-{t}.csv"
    log_fname = f"aq_benchmark-{t}.log"
    
    with open(log_fname, "w") as file:
        with redirect_stdout(file):
            print(f"===")
            print(f"Time: {t} ")
            print(f"Dimension: {landscape.nin}")
            print(f"# Repeats: {iter_repeats}")
            print(f"Aq: {aq_base.__name__}")
            print(f"nu: {nu}")
            print(f"alpha: {alpha}")
            print(f"===")
    
    
    with open(csv_fname, "w", newline='') as csvfile:
        hnames = list(arguments[0].keys()) + ["value"]
        writer = csv.DictWriter(csvfile, fieldnames=hnames) 
        writer.writeheader() 
        
        i=0
        for arg in arguments:
            arg["value"] = avg_entropy[i]
            writer.writerow(arg)
            i+=1
    

if __name__ == '__main__':
    main()
    
