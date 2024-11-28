import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal

peaks = 10
a = np.zeros((peaks,2))
c = np.zeros((peaks,4))
for i in range(peaks):
    a[i,0] = np.random.rand()*2-1
    a[i,1] = np.random.rand()*2-1
    c[i,0] = np.random.rand()/10
    c[i,1] = np.random.rand()/10

def blackbox(pos):
  
    mus = a
    covs = []
    for i in range(peaks):
        covs.append(np.diag([c[i,0], c[i,1]]))
    z2 = 0
  
    for vals in zip(mus, covs):
        rv =  multivariate_normal(vals[0], vals[1])
        z2 += rv.pdf(pos)
    return z2


x,y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x,y))


z = blackbox(pos)
print(z)



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, np.max(z))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


