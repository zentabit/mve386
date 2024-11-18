import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

nin = 2
nout = 2
nfunc = 5
peakedness = 100
d = 1

def plot2d(mus, covs): # we implicitly assume nout = 1
    x1,x2 = np.mgrid[0:d:.001, 0:d:.001]
    pos = np.dstack((x1,x2))

    y = f_scalar(pos, mus, covs)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x1,x2,y, vmin=y.min() * 2)

    plt.show()

def generate_mv_gaussians(nfunc, nin, d):
    mus = d * np.random.rand(nfunc, nin)
    covs = np.zeros((nfunc, nin, nin))
    for i in range(0, nfunc):
        covs[i] = np.diag(0.01 + np.random.rand(nin)/peakedness)

    print(np.shape(mus))
    return mus, covs

def gen_mv_gaussians_multi_out(nfunc, nin, nout, d):
    muss = np.zeros((nfunc, nin, nout))
    covss = np.zeros((nfunc, nin, nin, nout))

    for i in range(0, nout):
        muss[..., i], covss[..., i] = generate_mv_gaussians(nfunc, nin, d)
    
    return muss, covss

def f_vector(x, muss, covss):
    y = np.zeros(nout)
    for i in range(0, nout):
        y[i] = f_scalar(x, muss[..., i], covss[..., i])

    return y

def f_scalar(x, mus, covs):
    y = 0
    
    for vals in zip(mus, covs):
        rv = multivariate_normal(vals[0], vals[1])
        y += rv.pdf(x)

    return y

# mus, covs = generate_mv_gaussians(nfunc, nin, d)
muss, covss = gen_mv_gaussians_multi_out(nfunc, nin, nout, d)
mus = muss[..., 0]
covs = covss[..., 0]
print(np.shape(muss[..., 0]))
print(f_scalar((0.5, 0.5), mus, covs))
print(f_vector((0.5, 0.5), muss, covss))

plot2d(mus, covs)