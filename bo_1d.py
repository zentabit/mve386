from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import landscape
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.stats import entropy, gamma
import test_functions
from bo_common import *
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import random

batch_sz = 3 # batch size in LHS
landscape.peakedness = 100 # set the peakedness to get more extremes
mus, covs = landscape.gen_gauss(3, 1, 1) # fix an f throughout the run
mus_neg, covs_neg = landscape.gen_gauss(1, 1, 1) # fix an f throughout the run for the negative hills

def f(x):
    # return test_functions.trough1d(x)
    f_positive = landscape.f_sca(x, mus, covs)
    # f_positive = np.divide(f_positive,np.max(f_positive)) + 1

    # A try to make positive hills and negative but results in entropy being inf

    f_negative = landscape.f_sca(x, mus_neg, covs_neg)
    # f_negative = np.divide(f_negative,np.max(f_negative))
    f_total = np.subtract(f_positive,f_negative)
    f_total = f_positive
    # f_total = np.divide(f_total,np.max(f_total))
    return f_total

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


# Some acquisition functions
# acqf = acquisition.UpperConfidenceBound(kappa=10) 
# acqf = CB(beta=0, kappa=1)
# acqf = GP_UCB_2()
acqf = RGP_UCB(theta = 10)
# acqf = acquisition.ExpectedImprovement(xi = 10)
# acqf = thompson_sampling(random_state=13)
acqd = dummy_acqf()

# Set opt bounds and create target
pbounds = {'x': (0,1)}
x = np.arange(0,1,0.001).reshape(-1,1)
y = f(x)
nu = 1.5

# This is just a dummy for unif sampling
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqd,
    verbose = 0,
    random_state=0
)
optimizer._gp = GaussianProcessRegressor(
    kernel=Matern(nu=nu),
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=9,
    random_state=optimizer._random_state,
    )

presample_unifrefine(19, optimizer)
optimizer.maximize(init_points=0, n_iter=1) # by optimising once, we get a nice posterior
mu = plot_gp(optimizer, x, y)

print(f"Entropy of unif search: {entropy(y, np.abs(mu))}")

# This is the real run
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
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=9,
    random_state=optimizer._random_state,
    )

# One sample at a time

# presample_lh(batch_sz, optimizer)
# optimizer.maximize(init_points=0, n_iter=100)

# Batching

batches = 4
batch_size = 10

next_target = np.empty(batch_size,dtype=dict)
value = np.zeros(batch_size)

nt = optimizer.suggest()
point = f(**nt)
optimizer.register(nt,point)

for i in range(batches):
    optimizer._gp.fit(optimizer.space.params, optimizer.space.target)
    acu = -1 * optimizer.acquisition_function._get_acq(gp = optimizer._gp)(x)
    total_sum = np.sum(acu)
    weights = [value / total_sum for value in acu]
    # Kan vara intressant att ändra vikterna under körningen för att ha mer exploraion eller explotation
    # weights = [x**2 for x in weights]
    for j in range(batch_size):
        next_target[j] = random.choices(range(len(acu)), weights=weights, k=1)[0]/1000
        value[j] = f(next_target[j])
    for k in range(batch_size):
        optimizer.register(params=next_target[k],target=value[k])

mu = plot_gp(optimizer, x, y)
print(f"Entropy of regression: {entropy(y, np.abs(mu))}")

plt.show()


