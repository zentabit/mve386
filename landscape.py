import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

ndim = 2
nfunc = 3
d = 1

def plot2d():
    x,y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x,y))

    z = f(pos)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x,y,z, vmin=z.min() * 2)

    plt.show()

def generate_mv_gaussians(nfunc, ndim, d):
    mus = 2 * d * np.random.rand(nfunc, ndim) - d
    covs = np.zeros((nfunc, ndim, ndim))
    for i in range(0, nfunc):
        covs[i] = np.diag(0.1 + np.random.rand(ndim))

    return mus, covs

def f(x):

    mus, covs = generate_mv_gaussians(nfunc, ndim, d)

    z2 = 0
    
    for vals in zip(mus, covs):
        rv = multivariate_normal(vals[0], vals[1])
        z2 += rv.pdf(x)

    return z2

plot2d()