# run with python3 -m plotting.plot_sampling from ..
from lib.sampling_lh import LHSampler
from lib.sampling_randUnif import UniformSampler
from lib.sampling_unifrefine import unifspacing
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharex='all', sharey='all')
ax1.set_aspect('equal')
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.set_xticks(np.arange(0,1,0.5))
ax1.set_xticks(np.arange(0,1,0.0625), minor = True)
ax1.set_yticks(np.arange(0,1,0.5))
ax1.set_yticks(np.arange(0,1,0.0625), minor = True)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.grid(which = 'major', alpha = 0.5)
ax1.grid(which = 'minor', alpha = 0.2)
ax1.set_title('Random')

ax2.set_aspect('equal')
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax1.set_xticks(np.arange(0,1,0.5))
ax1.set_xticks(np.arange(0,1,0.0625), minor = True)
ax1.set_yticks(np.arange(0,1,0.5))
ax1.set_yticks(np.arange(0,1,0.0625), minor = True)
ax2.grid(which = 'major', alpha = 0.5)
ax2.grid(which = 'minor', alpha = 0.2)
ax2.set_title('Fixed interval')

ax3.set_aspect('equal')
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
ax1.set_xticks(np.arange(0,1,0.5))
ax1.set_xticks(np.arange(0,1,0.0625), minor = True)
ax1.set_yticks(np.arange(0,1,0.5))
ax1.set_yticks(np.arange(0,1,0.0625), minor = True)
ax3.grid(which = 'major', alpha = 0.5)
ax3.grid(which = 'minor', alpha = 0.2)
ax3.set_title('Latin hypercube')


lh = LHSampler(2)
points_lhs = lh.sample(16)
pts = np.transpose(points_lhs)
ax3.scatter(pts[0], pts[1], marker="x")

u = UniformSampler(2)
pts = np.transpose(u.sample(16))
ax1.scatter(pts[0], pts[1], marker = "x")

pts = unifspacing(1, 2, 4)
ax2.scatter(pts[0], pts[1], marker = "x")

fig.tight_layout()
plt.show()