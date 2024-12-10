from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import math
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import entropy
from contextlib import redirect_stdout
import random 
import time

# Our files
from bo_common import *
import landscape

class FunctionDetails:
    def __init__(self, x,X, exact_f, exact_Z):
        self.x = x
        self.mesh_array = X
        self.f = exact_f
        self.Z = exact_Z
    
    def calcEntropy(self, optimizer : BayesianOptimization):
        mu = self.extract_mu(optimizer)
        return entropy(self.Z.flatten(), np.abs(mu).flatten())
    
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
                dim,
                nu,
                alpha,
                aq_base : acquisition.AcquisitionFunction,
                n_sample,
                init_points,
                aq_params = {},
                n_sample_step_size:int = None,
                iteration_repeats:int = 1,
                function_number = 1,
                batch_size = 0,
                verbose = 0
                ):
        
        landscape.nin = dim
        
        self.nu = nu
        self.alpha = alpha
        self.aq = aq_base 
        self.n_sample = n_sample
        self.init_points = init_points
 
        self.aq_params = aq_params
        self.n_sample_step_size = n_sample_step_size
        self.iteration_repeats = iteration_repeats
        self.function_number = function_number
        
        self.batch_size = batch_size
        
        self.benchmark_array = None
        self.verbose = verbose

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
            random_state=0,
            allow_duplicate_points=True
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
                if self.verbose:
                    print(f"   Step {i+1} / {self.n_sample}")
                
                # optimizer.maximize()
                # {
                next_point = optimizer.suggest()
                target = fd.f(**next_point)
                optimizer.register(params=next_point, target=target)
                # }

                if i in N:
                    mu = fd.extract_mu(optimizer)
                    h_reg = entropy(fd.Z.flatten(), np.abs(mu).flatten())

                    temp = index + [c,0]
                    self.benchmark_array[*temp] = h_reg
                    
                    c+= 1
        elif self.batch_size:
            n_batches = math.ceil(self.n_sample / self.batch_size)
            
            next_target = np.empty((self.batch_size, landscape.nin), dtype=dict)
            values = np.zeros(self.batch_size)
            
            # Behövs tydligen annars fungerar ej
            optimizer.suggest()
            
            comb = np.dstack(fd.mesh_array)
            
            for i in range(n_batches):
                if self.verbose:
                    print(f"   Batch {i+1} / {n_batches}")
                    
                optimizer._gp.fit(optimizer.space.params, optimizer.space.target)
                acu = -1 * optimizer.acquisition_function._get_acq(gp = optimizer._gp)(comb)
                total_sum = np.sum(acu)
                weights = [value / total_sum for value in acu]
                
                for j in range(self.batch_size):
                    chosen_index = random.choices(range(len(acu)), weights=weights, k=1)[0]
                    next_target[j,:] = np.unravel_index(chosen_index, fd.mesh_array.shape[1:], order='F')
                    next_target[j,:] = next_target[j,:]/np.max(fd.mesh_array.shape) # Kan behöva dubbelkolla att x1 = x1
                    values[j] = fd.f(*next_target[j,:])
                
                for k in range(self.batch_size):
                    optimizer.register(params=next_target[k],target=values[k])


                h_reg = fd.calcEntropy(optimizer)
                
                temp = index + [0, i]
                self.benchmark_array[*temp] = h_reg
        else:
            for i in range(self.n_sample+1):
                if self.verbose:
                    print(f"   Sample {i+1} / {self.n_sample}")
                    
                # optimizer.maximize()
                # {
                next_point = optimizer.suggest()
                target = fd.f(**next_point)
                optimizer.register(params=next_point, target=target)
                # }

            h_reg = fd.calcEntropy(optimizer)
            
            temp = index + [0,0]

            self.benchmark_array[*temp] = h_reg
        
        
    
    def _f_aux(self, X):
        # return test_functions.trough2d(x)
        return landscape.f_sca(np.moveaxis(X,0,landscape.nin), self.mus, self.covs)

    def _create_function(self, arg_names):
        # Create a string defining the function with the required signature
        args = ", ".join(arg_names)
        
        global mus, covs
        mus,covs = self.mus, self.covs
        
        func_def = f"""def f({args}):return landscape.f_sca(({args}), mus, covs)"""
        # Execute this string in the global namespace
        exec(func_def, globals())   
    
    def _setup(self):
        self.arguments = self._computeAqParams()
        # #funktioner, #argument, #iterationer, #steg, #batchnr
        n_steps =  math.floor(self.n_sample/self.n_sample_step_size) if self.n_sample_step_size is not None else 1
        n_batches = math.ceil(self.n_sample/self.batch_size) if self.batch_size else 1
        
        self.benchmark_array = np.zeros((self.function_number, len(self.arguments), self.iteration_repeats, n_steps, n_batches))
        
        
    
    def run(self):        
        
        self._setup()
        
        if self.batch_size and self.n_sample_step_size:
            # This is due to the fact that it makes little sense to have both step sizes
            # and batches together. As such, the implementation only considers one of the 
            # cases which would be step sizes due to ordering of if-statements.
            raise ValueError("Only one benchmark of batches and step size should be chosen.")
        
        
        x = np.arange(0,1,0.01).reshape(-1,1)
        X = np.array(np.meshgrid(*[x for _ in range(landscape.nin)]))
        var_names = [ f"x{i}" for i in range(0, landscape.nin) ]
        self.pbounds = { var: (0,1) for var in var_names }
        
        landscape.peakedness = 10 # set the peakedness to get more extremes
        
        
        for a in range(self.function_number):
            if self.verbose:
                    print(f"Function number: {a+1} / {self.function_number}")
            
            self.mus, self.covs = landscape.gen_gauss(5, landscape.nin, 1) # fix an f throughout the ru
            self._create_function(var_names)
            Z = self._f_aux(X)

            fd = FunctionDetails(x,X, f, Z)
            
            for b, arg in enumerate(self.arguments):
                if self.verbose:
                    print(f" Parameter: {arg}")
                
                aq = self.aq(**arg)
                
                for c in range(self.iteration_repeats):
                    if self.verbose:
                        print(f"  Repeat: {c} / {self.iteration_repeats}")
                   
                    self._benchmark(fd, aq, [a,b,c])
        
        
    def _funcInfo(self):
        if self.benchmark_array is None:
            self._setup()
        
        return f"""
Dimension: {landscape.nin}

==== Benchmarking config ====
# of samples: {self.n_sample}
# Functions: {self.function_number}
Args: {self.arguments}
# Repeats: {self.iteration_repeats}
# Step Size: {self.n_sample_step_size}
# Bacthes: {math.ceil(self.n_sample/self.batch_size) if self.batch_size else None} (of size {self.batch_size})
Format: {np.shape(self.benchmark_array)} matrix
Format: #funktioner, #argument, #repetitioner, #steg, #batchnr

==== Misc config ====
nu: {self.nu}
alpha: {self.alpha}
initial points: {self.init_points}
verbose: {self.verbose}
"""
    
    def save(self, fname=""):
        
        if self.benchmark_array is None:
            raise Exception("Cannot save before doing a run!")
        
        if not fname:
            t = time.time()
            fname="benchmark-{t}"
        
        with open(fname+".log", "w") as file:
            with redirect_stdout(file):
               print(self._funcInfo())
    
        np.save(fname, self.benchmark_array)
    
    def _print(self):
        print(self.benchmark_array)
    
    def __str__(self):
        return self._funcInfo()