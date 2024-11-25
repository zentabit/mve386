from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import landscape
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import entropy, gamma
import sampling_randUnif
import test_functions
from acquisitionfunctions import *
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


def plot_gp_2d(optimizer, x, y, z):
    fig, axs = plt.subplots(2, 2)
    X, Y = np.meshgrid(x,y)
    z_min, z_max = -abs(z).max(), abs(z).max()
    
    ax = axs[0,0]
    c = ax.pcolor(X, Y, z)
    # print(np.shape(z))
    ax.set_title('Målfunktion')

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([[res["params"]["y"]] for res in optimizer.res])
    z_obs = np.array([res["target"] for res in optimizer.res])
    
    out = np.transpose(np.vstack([X.ravel(), Y.ravel()]))
    # print(out)
    mu, sigma = posterior(optimizer, out)
    print(mu)
    mu = np.reshape(mu, np.shape(X))
    sigma = np.reshape(sigma, np.shape(X))
    
    # print(np.shape(np.reshape(mu, np.shape(X))))
    ax = axs[0,1]
    c = ax.pcolor(X, Y, mu)
    ax.scatter(x_obs, y_obs, marker = 'x', c='black')
    ax.set_title('Posterior mean')
    ax.set_xlim([0, 1])
    ax.set_ylim([0,1])

    ax = axs[1,0]
    c = ax.pcolor(X, Y, sigma)
    ax.set_title('Covariance')

    ax = axs[1,1]
    c = ax.pcolor(X, Y, z - mu)
    ax.set_title('Mål - mean')

    return mu

def presample_lh(npoints, optimizer): # Crate a LHS and update the optimizer
    lh = LatinHypercube(landscape.nin)
    xs = lh.random(npoints)

    for x in xs:
        optimizer.probe(x)

def presample_unif(npoints, optimizer): # Sample uniformly and update the optimizer
    xs = sampling_randUnif.randUnifSample(landscape.nin, npoints)

    for x in xs:
        optimizer.probe(x)

# Some acquisition functions
# acqf = acquisition.UpperConfidenceBound(kappa=10) 
# acqf = CB(beta=0, kappa=1)
# acqf = GP_UCB_2()
# acqf = RGP_UCB(theta = 3)
acqf = acquisition.ExpectedImprovement(xi = 6)

# Set opt bounds and create target
pbounds = {'x': (0,1), 'y': (0,1)}
x = np.arange(0,1,0.01).reshape(-1,1)
y = np.arange(0,1,0.01).reshape(-1,1)
X, Y = np.meshgrid(x,y)
Z = f(X,Y)
npts = 50
# landscape.plot2d(mus, covs)

# This is just a dummy for unif sampling
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
    verbose = 0,
    random_state=0
)
optimizer._gp = GaussianProcessRegressor(
    kernel=Matern(nu=1.5),
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=9,
    random_state=optimizer._random_state,
    )

presample_unif(npts - 1, optimizer)
optimizer.maximize(init_points=0, n_iter=1) # by optimising once, we get a nice posterior
mu = plot_gp_2d(optimizer, x, y, Z)

print(f"Entropy of unif search: {entropy(Z.flatten(), np.abs(mu).flatten())}")

# This is the real run
optimizer = BayesianOptimization(
    f = f,
    pbounds=pbounds,
    acquisition_function=acqf,
    verbose = 0,
    random_state=0
)
optimizer._gp = GaussianProcessRegressor(
    kernel=Matern(nu=1.5),
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=9,
    random_state=optimizer._random_state,
    )

presample_lh(batch_sz, optimizer)
optimizer.maximize(init_points=0, n_iter=npts)

# print(z)
mu = plot_gp_2d(optimizer, x, y, Z)
# mu = plot_gp(optimizer, x, y)
print(f"Entropy of regression: {entropy(Z.flatten(), np.abs(mu).flatten())}")

plt.show()


