from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import landscape
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import entropy
# from scipy.spatial.distance import jensenshannon
import sampling_randUnif

batch_sz = 3 # batch size in LHS
landscape.peakedness = 100 # set the peakedness to get more extremes
mus, covs = landscape.gen_gauss(5, 1, 1) # fix an f throughout the run

class CB(acquisition.AcquisitionFunction): # This is like UCB, but we can also set a parameter beta for mean
    def __init__(self, random_state = None, beta = 1, kappa = 1):
        super().__init__(random_state)
        self.beta = beta
        self.kappa = kappa
    
    def base_acq(self, mean, std):
        return self.beta * mean + self.kappa * std

class GP_UCB(acquisition.AcquisitionFunction): # Using Thm 2
    def __init__(self, random_state = None, delta = 0.1, a = 1, b = 0.2):
        super().__init__(random_state)
        self.delta = delta
        self.b = b
        self.a = a
    
    def base_acq(self, mean, std):
        beta = 2 * np.log2( (self.i+1)**2 * 2 * np.pi**2 / (3 * self.delta) ) + 2 * landscape.nin * np.log2( (self.i + 1)**2 ** landscape.nin * self.b * landscape.d * np.sqrt(np.log2(4 * self.a * landscape.nin / self.delta )) )
        # print(self.i)
        return mean + np.sqrt(beta/5) * std

def f(x):
    return landscape.f_sca(x, mus, covs)

def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y): # Given opt result and target function, plot result and next point to be acquired
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontsize=30
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer._space)
    mu, sigma = posterior(optimizer, x)

    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((0, 1))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})

    utility_function = optimizer.acquisition_function
    utility = -1 * utility_function._get_acq(gp=optimizer._gp)(x)
    x = x.flatten()

    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0, 1))
    #acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    return mu

def presample_lh(npoints, optimizer): # Crate a LHS and update the optimizer
    lh = LatinHypercube(landscape.nin)
    xs = lh.random(npoints)

    for x in xs:
        optimizer.register(x, f(x))

def presample_unif(npoints, optimizer): # Sample uniformly and update the optimizer
    xs = sampling_randUnif.randUnifSample(landscape.nin, npoints)

    for x in xs:
        optimizer.register(x, f(x))

# Some acquisition functions
# acqf = acquisition.UpperConfidenceBound(kappa=10) 
acqf = CB(beta=0, kappa=1)
# acqf = GP_UCB()
# acqf = acquisition.ExpectedImprovement(xi = 10)

# Set opt bounds and create target
pbounds = {'x': (0,1)}
x = np.arange(0,1,0.001).reshape(-1,1)
y = f(x)

# This is just a dummy for unif sampling
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
    verbose = 0,
    random_state=0
)

presample_unif(19, optimizer)
optimizer.maximize(init_points=0, n_iter=1) # by optimising once, we get a nice posterior
mu = plot_gp(optimizer, x, y)

print(f"Entropy of unif search: {entropy(y, np.abs(mu))}")

# This is the real run
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
    verbose = 0,
    random_state=0
)

presample_lh(batch_sz, optimizer)
optimizer.maximize(init_points=0, n_iter=17)

mu = plot_gp(optimizer, x, y)
plt.show()

print(f"Entropy of regression: {entropy(y, np.abs(mu))}")
