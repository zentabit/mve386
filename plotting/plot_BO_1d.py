# run with python3 -m plotting.plot_BO_1d from ..
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from lib.landscape import f_sca, gen_gauss
from lib import landscape
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

plt.rcParams.update({'font.size': 16})
# import tikzplotlib

# plt.rcParams['font.family'] = "serif"
# plt.rcParams['font.serif'] = "mathpazo"

batch_sz = 3 # batch size in LHS
landscape.peakedness = 100 # set the peakedness to get more extremes
mus, covs = gen_gauss(1, 1, 1) # fix an f throughout the run
# mus_neg, covs_neg = gen_gauss(1, 1, 1) # fix an f throughout the run for the negative hills

def f(x):
    # return test_functions.trough1d(x)

    return f_sca(x, mus, covs)

def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y): # Given opt result and target function, plot result and next point to be acquired
    fig = plt.figure(figsize=(16, 8))
    steps = len(optimizer.space)
    # fig.suptitle(
    #     'Gaussian Process and Utility Function After {} Steps'.format(steps),
    #     fontsize=30
    # )

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
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
        alpha=.3, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((0, 1))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    axis.set_xticklabels([])
    axis.set_yticklabels([])

    utility_function = optimizer.acquisition_function
    utility = -1 * utility_function._get_acq(gp=optimizer._gp)(x)
    x = x.flatten()

    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0, 1))
    #acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel("$\\alpha(x, D)$", fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    acq.set_xticklabels([])
    acq.set_yticklabels([])

    fig.tight_layout()

    # axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    # acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    return mu



# Some acquisition functions
acqf = acquisition.UpperConfidenceBound(kappa=5) 
# acqf = CB(beta=0, kappa=1)
# acqf = GP_UCB_2()
# acqf = RGP_UCB(theta = 3)
# acqf = acquisition.ExpectedImprovement(xi = 5.4)

# Set opt bounds and create target
pbounds = {'x': (0,1)}
x = np.arange(0,1,0.001).reshape(-1,1)
y = f(x)
nu = 1.5


# This is the real run
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
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

optimizer.maximize(init_points=0, n_iter=5)

mu = plot_gp(optimizer, x, y)
# print(f"Entropy of regression: {entropy(y, np.abs(mu))}")

plt.savefig("figures/bo.svg")
plt.show()

