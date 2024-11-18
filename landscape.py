import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

ndim = 2
nfunc = 5
peakedness = 100
d = 1

def plot2d(mus, covs):
    x1,x2 = np.mgrid[0:d:.001, 0:d:.001]
    pos = np.dstack((x1,x2))

    y = f(pos, mus, covs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x1,x2,y, vmin=y.min() * 2)

    plt.show()

def generate_mv_gaussians(nfunc, ndim, d):
    mus = d * np.random.rand(nfunc, ndim)
    covs = np.zeros((nfunc, ndim, ndim))
    for i in range(0, nfunc):
        covs[i] = np.diag(0.05 + np.random.rand(ndim)/peakedness)

    return mus, covs

def f(x, mus, covs):
    y = 0
    
    for vals in zip(mus, covs):
        rv = multivariate_normal(vals[0], vals[1])
        y += rv.pdf(x)

    return y

mus, covs = generate_mv_gaussians(nfunc, ndim, d)
print(f((0.5,0.5), mus, covs))

plot2d(mus, covs)