import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal




# # Gör en 2D 100x100 
# n = 2
# x = [ np.linspace(-1,1,num=100) for _ in range(0,n)]
# grid = np.meshgrid(*x)

# # Tror denna behövs av koden precis nedan, osäker exakt hur man formulerar det
# pos = np.dstack(grid)



x,y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x,y))

def f_wrapper(x,y):
    return f((x,y))

def f(pos, nfunc = 3, ndim = 2, d = 2):
    
    # mus = [
    #     [0.5, 0], 
    #     [-0.8, -0.4],
    #     [0, 1]
    #     ]
    # covs = [
    #     np.diag([0.2, 0.4]),
    #     np.diag([0.5, 0.1]),
    #     np.diag([0.1, 0.1])
    #     ]
    
    # mus = [
    #     [2*np.random.rand() - 1, 2*np.random.rand() - 1], 
    #     [2*np.random.rand() - 1, 2*np.random.rand() - 1],
    #     [2*np.random.rand() - 1, 2*np.random.rand() - 1]
    #     ]
    
    mus = 2 * d * np.random.rand(nfunc, ndim) - d

    print(mus)
    
    covs = [
        np.diag([0.1 + np.random.rand(), 0.1 + np.random.rand()]),
        np.diag([0.1 + np.random.rand(), 0.1 + np.random.rand()]),
        np.diag([0.1 + np.random.rand(), 0.1 + np.random.rand()])
        ]
    
    # covs = np.zeros((nfunc, ndim, ndim))
    
    # for i in range(0, nfunc):
    #     covs[i] = np.diag(0.1 + np.random.rand(ndim))
    
    print(covs)

    z2 = 0
    
    for vals in zip(mus, covs):
        rv =  multivariate_normal(vals[0], vals[1])
        z2 += rv.pdf(pos)

    return z2

# from bayes_opt import BayesianOptimization
# bounds = {'x': (-1,1), 'y':(-1,1)}

# opt = BayesianOptimization(
#     f=f_wrapper,
#     pbounds=bounds,
#     random_state=1
# )

# opt.maximize(
#     init_points=2,
#     n_iter=10
# )

# print(opt.max)

z = f(pos)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x,y,z, vmin=z.min() * 2)


# print(z)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.contourf(x,y, z)

plt.show()
