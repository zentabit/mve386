import landscape
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def gauss1d(x):
    mu = [ 0.5 ]
    cov = [ 0.01 ]

    return landscape.f_sca(x, mu, cov)

def gauss2d(x,y): # two vectors as input
    mus = [ 0.5, 0.5 ]
    covs = np.diag([0.01, 0.01])

    X, Y = np.meshgrid(x,y)
    pos = np.dstack((X,Y))
    rv = multivariate_normal(mus, covs)
    return X,Y,rv.pdf(pos)

def w_gauss2d(p):
    return gauss2d(p[0],p[1])[2]

def trough1d(x):
    mus = [[0], [1]]
    covs = [[0.1], [0.1]]

    return landscape.f_sca(x, mus, covs)

def trough2d(x,y):
    mus = [
        [1,0],
        [1,1]
    ]

    covs = [
        np.diag([0.1, 0.05]),
        np.diag([0.1, 0.05])
    ]

    X, Y = np.meshgrid(x,y)
    pos = np.dstack((X,Y))

    return X, Y, landscape.f_sca(pos, mus, covs)

def w_trough2d(p):
    return trough2d(p[0],p[1])[2]

def sheet1d(x, k = 10):
    return 1/(1 + np.exp(-k * (x-0.5)))

def sheet2d(x,y, k = 10):
    X, Y = np.meshgrid(x,y)
    pos = np.dstack((X,Y))

    return X, Y, 1/(1 + np.exp(-k * (X-0.5)))

def w_sheet2d(p):
    return sheet2d(p[0],p[1])[2]



def main():
    x = np.linspace(0, 1)
    # plt.plot(x, sheet1d(x))
    # plt.show()
    X,Y,Z = gauss2d(x,x)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X,Y,Z, vmin=Z.min() * 2)

    plt.show()
    

if __name__ == '__main__':
    main()