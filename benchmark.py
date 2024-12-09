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

class Benchmark:
    
    def __init__(self,
                nu,
                alpha,
                aq_base : acquisition.AcquisitionFunction,
                n_sample,
                init_points,
                aq_params = {},
                n_sample_step_size:int = None,
                iteration_repeats:int = 1,
                function_number = 1,
                batches = 0,
                batch_size = 0
                ):
        self.nu = nu
        self.alpha = alpha
        self.aq = aq_base 
        self.n_sample = n_sample
        self.init_points = init_points
 
        self.aq_params = aq_params
        self.n_sample_step_size = n_sample_step_size
        self.iteration_repeats = iteration_repeats
        self.function_number = function_number
        
        self.batches = batches
        self.batch_size = batch_size
        
        self.benchmark_array = np.ndarray((1))

    def _computeIterationBreakpoints(self):
        return set([(i+1)*self.n_sample_step_size for i in range(math.floor(self.n_sample/self.n_sample_step_size))])

    def _computeAqParams(self):

        # https://i.kym-cdn.com/entries/icons/original/000/041/943/1aa1blank.png
        
        d = {}
        for key in self.aq_params.keys():
            start, stop, refinement = self.aq_params[key]
            
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
        
        return arguments
        
    
    def _benchmark(self,
                fd : FunctionDetails,
                aq : acquisition.AcquisitionFunction,
                index):
        
        # index is the entry that the resulting entropy should be placed at
        # in the output matrix
        
        optimizer = BayesianOptimization(
            f = fd.f,
            pbounds=self.pbounds,
            acquisition_function=aq,
            verbose = 0,
            random_state=0
        )
        optimizer._gp = GaussianProcessRegressor(
            kernel=Matern(nu=self.nu),
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=9,
            random_state=optimizer._random_state,
        )

        presample_lh(self.init_points, optimizer, fd.f)
        
        if(self.n_sample_step_size):
            N = self._computeIterationBreakpoints()
            c = 0
            for i in range(self.n_sample+1):
                # optimizer.maximize()
                # {
                next_point = optimizer.suggest()
                target = fd.f(**next_point)
                optimizer.register(params=next_point, target=target)
                # }

                if i in N:
                    mu = fd.extract_mu(optimizer)
                    h_reg = entropy(fd.Z.flatten(), np.abs(mu).flatten())


                    # TODO: kolla om logiken för indexering stämmer
                    self.benchmark_array[index + [c]] = h_reg
                    
                    c+= 1
        else:
            optimizer.maximize(init_points=0, n_iter=self.n_sample)

            mu = fd.extract_mu(optimizer)
            h_reg = entropy(fd.Z.flatten(), np.abs(mu).flatten())
            self.benchmark_array[index + [0]] = h_reg
        
        
    
    def _batchBenchmark(self):
        pass
    
    
    def _f_aux(self, X):
        # return test_functions.trough2d(x)
        return landscape.f_sca(np.moveaxis(X,0,landscape.nin), self.mus, self.covs)

    # TODO: se över hur denna ska fungera inom klassen
    def _create_function(self, arg_names):
        # Create a string defining the function with the required signature
        args = ", ".join(arg_names)
        func_def = f"""
    def f({args}):
        return landscape.f_sca(({args}), mus, covs)
    """
        # Execute this string in the global namespace
        exec(func_def, globals())   
             
    
    def run(self):
        raise Exception("Fungerar ej ännu!")
        
        x = np.arange(0,1,0.01).reshape(-1,1)
        X = np.array(np.meshgrid(*[x for _ in range(landscape.nin)]))
        var_names = [ f"x{i}" for i in range(0, landscape.nin) ]
        self.pbounds = { var: (0,1) for var in var_names }
        self._create_function(var_names)

        landscape.peakedness = 10 # set the peakedness to get more extremes
        
        arguments = self._computeAqParams()
        
        for a in range(self.function_number):
            self.mus, self.covs = landscape.gen_gauss(5, landscape.nin, 1) # fix an f throughout the ru
            Z = self._f_aux(X)

            fd = FunctionDetails(x,X, f, Z)
            
            
            for b, arg in enumerate(arguments):
                
                aq = self.aq(arg)
                
                for c in range(self.iteration_repeats):
                    
                    self._benchmark(fd, aq, [a,b,c])
        
        
        # Loop för funktioner
        
        # Loop för aq argument
        # Loop för iterationer
     

    def save(self, fname=""):
        
        if not fname:
            t = time.time()
            fname="benchmark-{t}"
        
        with open(fname+".log", "w") as file:
            with redirect_stdout(file):
                print(f"===")
                print(f"Time: {t} ")
                print(f"Dimension: {landscape.nin}")
                print(f"# Repeats: {self.iteration_repeats}")
                print(f"# Functions: {self.function_number}")
                #print(f"Aq: {aq_base.__name__}")
                print(f"nu: {self.nu}")
                print(f"alpha: {self.alpha}")
                #print(f"Format: {n}x{len(x)}x{iter_repeats} matrix")
                #print(f"Format: #functions x #param. x #repeats")
                #print(f"Params: {x}")
                print(f"===")
    
        np.save(fname, self.benchmark_array)