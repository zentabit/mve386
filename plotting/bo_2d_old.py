from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import lib.landscape
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy, gamma
import lib.test_functions
from lib.bo_common import *
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

batch_sz = 3 # batch size in LHS
landscape.nin = 2
landscape.peakedness = 10 # set the peakedness to get more extremes
mus, covs = landscape.gen_gauss(5, 2, 1) # fix an f throughout the run

def f(x,y):
    # return test_functions.trough2d(x)
    return landscape.f_sca(np.dstack((x,y)), mus, covs)

def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp_2d(optimizer, x, y, z, save = False):
    fig, axs = plt.subplots(2, 2)
    fig.set_dpi(300)
    fig.tight_layout()
    X, Y = np.meshgrid(x,y)
    z_min, z_max = z.min(), z.max()
    cmap = 'hot'
    
    ax = axs[0,0]
    # c = ax.pcolor(X, Y, z, cmap = cmap, vmin = z_min, vmax = z_max)
    c = ax.imshow(z, cmap = cmap, interpolation = 'nearest', extent = [0, 1, 0, 1], vmin = z_min, vmax = z_max)
    ax.set_title(r'$f(x)$')

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([[res["params"]["y"]] for res in optimizer.res])
    z_obs = np.array([res["target"] for res in optimizer.res])
    
    out = np.transpose(np.vstack([X.ravel(), Y.ravel()]))
    mu, sigma = posterior(optimizer, out)
    mu = np.reshape(mu, np.shape(X))
    sigma = np.reshape(sigma, np.shape(X))
    
    ax = axs[0,1]
    # c = ax.pcolormesh(X, Y, mu, cmap = cmap, vmin = z_min, vmax = z_max, linewidth = 0)
    # c.set_edgecolor('face')
    c = ax.imshow(mu, cmap = cmap, interpolation = 'nearest', extent = [0, 1, 0, 1], vmin = z_min, vmax = z_max)
    ax.scatter(x_obs, y_obs, marker = 'x', c='green')
    ax.set_title(r'$\mu_D(x)$')
    ax.set_xlim([0, 1])
    ax.set_ylim([0,1])

    ax = axs[1,0]
    # c = ax.pcolor(X, Y, sigma, cmap = cmap, vmin = z_min, vmax = z_max)
    c = ax.imshow(sigma, cmap = cmap, interpolation = 'nearest', extent = [0, 1, 0, 1], vmin = z_min, vmax = z_max)
    ax.set_title(r'$\sigma_D(x)$')

    ax = axs[1,1]
    c = ax.imshow(np.abs(z - mu), cmap = cmap, interpolation = 'nearest', extent = [0, 1, 0, 1], vmin = z_min, vmax = z_max)
    # c = ax.pcolor(X, Y, np.abs(z - mu), cmap = cmap, vmin = z_min, vmax = z_max)
    ax.set_title(r'$|f(x)-\mu_D(x)|$')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(c, cax=cbar_ax)

    if save:
        plt.savefig('figures/bo_2d.svg')

    return mu

# Some acquisition functions
# acqf = acquisition.UpperConfidenceBound(kappa=10) 
# acqf = CB(beta=0, kappa=1)
# acqf = GP_UCB_2()
# acqf = RGP_UCB(theta = 3)
acqf = acquisition.ExpectedImprovement(xi = 8)
acqd = dummy_acqf()

# Set opt bounds and create target
pbounds = {'x': (0,1), 'y': (0,1)}
x = np.arange(0,1,0.01).reshape(-1,1)
y = np.arange(0,1,0.01).reshape(-1,1)
X, Y = np.meshgrid(x,y)
Z = f(X,Y)
npts = 49
nu = 1.5
alpha = 1e-3
# landscape.plot2d(mus, covs)

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
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=9,
    random_state=optimizer._random_state,
    )

# presample_unif(npts - 1, optimizer)
presample_unifspacing(4, optimizer)

optimizer.maximize(init_points=0, n_iter=1) # by optimising once, we get a nice posterior
mu = plot_gp_2d(optimizer, x, y, Z)
h_unif = entropy(Z.flatten(), np.abs(mu).flatten())
print(f"Entropy of unif search: {h_unif}")

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
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=9,
    random_state=optimizer._random_state,
    )

# presample_lh(npts, optimizer)
# presample_unif(npts, optimizer)
optimizer.maximize(init_points=1, n_iter=15)

mu = plot_gp_2d(optimizer, x, y, Z, save=True)
h_reg = entropy(Z.flatten(), np.abs(mu).flatten())
print(f"Entropy of regression: {h_reg}")
print(f"h_unif/h_reg = {h_unif/h_reg}")

plt.show()


