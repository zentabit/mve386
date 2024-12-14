import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
# import benchmarks

benchmark_sizes = [5,10,15,20,30]
entropy = np.zeros((5,11))

for i in range(np.size(benchmark_sizes)):
    data = np.load(f"benchmarks/CB/CB-{benchmark_sizes[i]}.npy")
    print(data.shape)
    for j in range(data.shape[1]):
        for k in range(data.shape[0]):
            entropy[i,j] = entropy[i,j] + data[k,j,0,0,-1]
entropy = entropy/data.shape[0]

plt.imshow(entropy)
plt.colorbar()
plt.show()


# peaks = 10
# a = np.zeros((peaks,2))
# c = np.zeros((peaks,4))
# for i in range(peaks):
#     a[i,0] = np.random.rand()*2-1
#     a[i,1] = np.random.rand()*2-1
#     c[i,0] = np.random.rand()/10
#     c[i,1] = np.random.rand()/10
# 
# def blackbox(pos):
#   
#     mus = a
#     covs = []
#     for i in range(peaks):
#         covs.append(np.diag([c[i,0], c[i,1]]))
#     z2 = 0
#   
#     for vals in zip(mus, covs):
#         rv =  multivariate_normal(vals[0], vals[1])
#         z2 += rv.pdf(pos)
#     return z2
# 
# 
# x,y = np.mgrid[-1:1:.01, -1:1:.01]
# pos = np.dstack((x,y))
# 
# 
# z = blackbox(pos)
# print(z)
# 
# 
# 
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# 
# # Plot the surface.
# surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# 
# # Customize the z axis.
# ax.set_zlim(0, np.max(z))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
# 
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# 
# plt.show()
# 
# 
# run_times = np.array([[4.76507691e-02, 1.85713248e+02],
#             [2.05746591e-01, 1.04767118e+02],
#             [1.00079083e-01, 6.89122161e+01],
#             [2.54073580e-02, 5.23946594e+01],
#             [3.38734734e-02, 4.00771613e+01],
#             [8.81170022e-02, 3.56480618e+01],
#             [1.28597448e-01, 2.94809392e+01],
#             [1.78325471e-02, 3.33486115e+01],
#             [2.35131897e-01, 2.51346792e+01],
#             [5.78387754e-01, 3.17275105e+01]])
# 
# x = np.linspace(10,100,10)
# 
# plt.title("Entropy")
# plt.xlabel("Batch size")
# plt.plot(x,run_times[:,0])
# plt.show()
# 
# plt.title("Run time in seconds")
# plt.xlabel("Batch size")
# plt.plot(x,run_times[:,1])
# plt.show()