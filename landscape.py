import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def mv_norm(indim, d):
    X, Y = np.mgrid[-d:d:0.01, -d:d:0.01]
    print(np.shape(X))
    Z = 0
    print(np.shape(Y))

    for i in range(0,5):
        nor = generate_mv_norm(indim, d)
        Z += get_value(nor, np.dstack((X,Y)))
    
    return X, Y, Z


def generate_mv_norm(n, d):
    return sp.stats.multivariate_normal(mean = 2*d*np.random.rand(n)-d, cov = np.diag(np.random.rand(n)))
    # return sp.stats.multivariate_normal(mean = [0.2, 2], cov = [[0.1,0],[0,2]])

def get_value(nor, x):
    return nor.pdf(x)

X, Y, Z = mv_norm(2, 4)
print("hej")

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, vmin=Z.min() * 2)

plt.show()