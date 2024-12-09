from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import landscape
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy, gamma
import test_functions
from bo_common import *
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import random

batch_sz = 3 # batch size in LHS
landscape.nin = 2
landscape.peakedness = 10 # set the peakedness to get more extremes
mus, covs = landscape.gen_gauss(5, 2, 1) # fix an f throughout the run

def f(x,y):
    # return test_functions.trough2d(x)
    return landscape.f_sca(np.dstack((x,y)), mus, covs)

def f2(X):
    # return test_functions.trough2d(x)
    return landscape.f_sca(np.moveaxis(X,0,landscape.nin), mus, covs)

def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp_2d(optimizer, x, y, z):
    fig, axs = plt.subplots(2, 2)
    X, Y = np.meshgrid(x,y)
    z_min, z_max = z.min(), z.max()
    cmap = 'hot'
    
    ax = axs[0,0]
    c = ax.pcolor(X, Y, z, cmap = cmap, vmin = z_min, vmax = z_max)
    ax.set_title('Målfunktion')

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([[res["params"]["y"]] for res in optimizer.res])
    z_obs = np.array([res["target"] for res in optimizer.res])
    
    out = np.transpose(np.vstack([X.ravel(), Y.ravel()]))
    mu, sigma = posterior(optimizer, out)
    mu = np.reshape(mu, np.shape(X))
    sigma = np.reshape(sigma, np.shape(X))
    
    ax = axs[0,1]
    c = ax.pcolor(X, Y, mu, cmap = cmap, vmin = z_min, vmax = z_max)
    ax.scatter(x_obs, y_obs, marker = 'x', c='green')
    ax.set_title('Posterior mean')
    ax.set_xlim([0, 1])
    ax.set_ylim([0,1])

    ax = axs[1,0]
    c = ax.pcolor(X, Y, sigma, cmap = cmap, vmin = z_min, vmax = z_max)
    ax.set_title('Covariance')

    ax = axs[1,1]
    c = ax.pcolor(X, Y, np.abs(z - mu), cmap = cmap, vmin = z_min, vmax = z_max)
    ax.set_title('Mål - mean')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(c, cax=cbar_ax)

    return mu

# Some acquisition functions
# acqf = acquisition.UpperConfidenceBound(kappa=10) 
# acqf = CB(beta=0, kappa=1)
# acqf = GP_UCB_2()
acqf = RGP_UCB(theta = 3)
# acqf = acquisition.ExpectedImprovement(xi = 6)
acqd = dummy_acqf()

# Set opt bounds and create target
pbounds = {'x': (0,1), 'y': (0,1)}
x = np.arange(0,1,0.01).reshape(-1,1)
y = np.arange(0,1,0.01).reshape(-1,1)
X, Y = np.meshgrid(x,y)
Z = f(X,Y)

x2 = np.arange(0,1,0.01).reshape(-1,1)
X2 = np.array(np.meshgrid(*[x2 for _ in range(landscape.nin)]))

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
presample_unifrefine(2, optimizer)

optimizer.maximize(init_points=0, n_iter=1) # by optimising once, we get a nice posterior
mu = plot_gp_2d(optimizer, x, y, Z)
h_unif = entropy(Z.flatten(), np.abs(mu).flatten())
print(f"Entropy of unif search: {h_unif}")

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
    alpha=alpha,
    normalize_y=True,
    n_restarts_optimizer=9,
    random_state=optimizer._random_state,
    )

# One sample at a time

# presample_lh(npts, optimizer)
# presample_unif(npts, optimizer)
# optimizer.maximize(init_points=5, n_iter=npts - 5)

# Batching

batches = 1
batch_size = 1
dim = 2
next_target = np.empty((batch_size,dim),dtype=dict)
value = np.zeros(batch_size)


# nt = optimizer.suggest()
# point = f(**nt)
# optimizer.register(nt,point)

presample_lh(10,optimizer,f)

nt = optimizer.suggest()

comb = np.dstack(X2)

for i in range(batches):
    optimizer._gp.fit(optimizer.space.params, optimizer.space.target)
    acu = -1 * optimizer.acquisition_function._get_acq(gp = optimizer._gp)(comb)
    total_sum = np.sum(acu)
    weights = [value / total_sum for value in acu]
    # Kan vara intressant att ändra vikterna under körningen för att ha mer exploraion eller explotation
    weights = [x**2 for x in weights]
    for j in range(batch_size):
        chosen_index = random.choices(range(len(acu)), weights=weights, k=1)[0]
        next_target[j,:] = np.unravel_index(chosen_index, (100,100), order='F')
        next_target[j,:] = next_target[j,:]/100
        value[j] = f(*next_target[j,:])
    for k in range(batch_size):
        optimizer.register(params=next_target[k],target=value[k])

mu = plot_gp_2d(optimizer, x, y, Z)
h_reg = entropy(Z.flatten(), np.abs(mu).flatten())
print(f"Entropy of regression: {h_reg}")
print(f"h_unif/h_reg = {h_unif/h_reg}")



plt.show()


