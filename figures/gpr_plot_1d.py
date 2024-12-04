import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound
from test_functions import trough1d
import matplotlib.pyplot as plt

pbounds = {'x': (0,1)}
x = np.arange(0,1,0.001).reshape(-1,1)

optimizer = BayesianOptimization(
    f = trough1d,
    pbounds=pbounds,
    acquisition_function=UpperConfidenceBound(),
    verbose = 0,
    random_state=0
)

optimizer.probe([0.1], lazy=False)
optimizer.probe([0.25], lazy=False)
optimizer.probe([0.5], lazy=False)

def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y): # Given opt result and target function, plot result and next point to be acquired
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer._space)
    mu, sigma = posterior(optimizer, x)

    # plt.plot(x, y, linewidth=3, label='Target')
    plt.plot(x, optimizer._gp.sample_y(x))
    plt.plot(x, optimizer._gp.sample_y(x, random_state =1))
    plt.plot(x, optimizer._gp.sample_y(x, random_state = 2))
    plt.plot(x_obs.flatten(), y_obs, '+', markersize=8, label=u'Observations', color='r')
    plt.plot(x, mu, '--', color='k', label='Prediction')

    plt.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.3, fc='c', ec='None', label='95% confidence interval')

    plt.xlim((0, 1))
    plt.ylim((None, None))
    plt.ylabel('f(x)', fontdict={'size':12})
    plt.xlabel('x', fontdict={'size':12})

    return mu

plot_gp(optimizer, x, trough1d(x))
plt.show()