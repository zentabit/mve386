from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import landscape
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.stats.qmc import LatinHypercube

batch_sz = 2
landscape.peakedness = 100
mus, covs = landscape.gen_gauss(5, 1, 1) # fix an f throughout the run

class CB(acquisition.AcquisitionFunction):
    def __init__(self, random_state = None, beta = 1, kappa = 1):
        super().__init__(random_state)
        self.beta = beta
        self.kappa = kappa
    
    def base_acq(self, mean, std):
        return self.beta * mean + self.kappa * std

def f(x):
    return landscape.f_sca(x, mus, covs)

def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
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

def presample_lh(npoints, optimizer):
    lh = LatinHypercube(landscape.nin)
    xs = lh.random(npoints)

    for x in xs:
        optimizer.register(x, f(x))

# acqf = acquisition.UpperConfidenceBound(kappa=1e4)
acqf = CB(beta=0.2, kappa=1)
pbounds = {'x': (0,1)}
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
    verbose = 2,
    random_state=0
)

presample_lh(batch_sz, optimizer)

x = np.arange(0,1,0.001).reshape(-1,1)
y = f(x)

optimizer.maximize(init_points=0, n_iter=10)
plot_gp(optimizer, x, y)
plt.show()
