"""landscape
This file generates test functions as sums of gaussian peaks by randomising
mean and covariance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

nin = 1 # N, dimension of invector x
nout = 1 # M, dimension of outvector y
nfunc = 5 # How many gaussians should be used
peakedness = 100 # To make the gaussians more pointy
d = 1 # Side length of the [0,d]^N cube

def plot2d(mus, covs): # plots N = 2, M = 1 distributions on [0,d]^2 (implicity assuming M = 1)
    x1,x2 = np.mgrid[0:d:.001, 0:d:.001]
    pos = np.dstack((x1,x2))
    y = f_sca(pos, mus, covs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x1,x2,y, vmin=y.min() * 2)

    plt.show()

def plot1d(mus, covs): # plots N = 1, M = 1 distributions on [0,d] (implicity assuming M = 1)
    x1 = np.arange(0,1,0.001)
    y = f_sca(x1, mus, covs)

    plt.plot(x1, y)

    plt.show()

def gen_gauss(nfunc, nin, d): # randomly select means and covariances for nfunc multivariate gaussians
    mus = d * np.random.rand(nfunc, nin)
    covs = np.zeros((nfunc, nin, nin))

    for i in range(0, nfunc):
        covs[i] = np.diag(1e-3 + np.random.rand(nin)/peakedness)

    return mus, covs

def gen_gauss_vec(nfunc, nin, nout, d): # use gen_gauss for nout outputs
    muss = np.zeros((nfunc, nin, nout))
    covss = np.zeros((nfunc, nin, nin, nout))

    for i in range(0, nout):
        muss[..., i], covss[..., i] = gen_gauss(nfunc, nin, d)
    
    return muss, covss

def f_vec(x, muss, covss): # calculate f when M > 1
    y = np.zeros(nout)

    for i in range(0, nout):
        y[i] = f_sca(x, muss[..., i], covss[..., i])

    return y

def f_sca(x, mus, covs): # calculate f when M = 1
    y = 0
    
    for vals in zip(mus, covs):
        rv = multivariate_normal(vals[0], vals[1])
        y += rv.pdf(x)

    return y

# mus, covs = gen_gauss(nfunc, nin, d)
# muss, covss = gen_gauss_vec(nfunc, nin, nout, d)
# mus = muss[..., 0]
# covs = covss[..., 0]
# print(np.shape(muss[..., 0]))
# print(f_sca((0.5, 0.5), mus, covs))
# print(f_vec((0.5, 0.5), muss, covss))

# plot1d(mus, covs)