# This program takes a CSV file from the benchmarking scripts and plots it in linear-log scale.
import numpy as np
import matplotlib.pyplot as plt


timestamp = "1733906073.614063"
fname = f"benchmark-{timestamp}.npy"

ax = plt.axes()

vals = np.load(fname)
print(vals)
print([ str(label) for label in vals[:, 0] ])
ax.set_yscale('log')
ax.boxplot(vals[ :, 0, 0, 0, : ]) #, tick_labels =[ str(int(label)) for label in vals[:, 0] ] )
ax.set_title("Hej")
ax.set_xlabel('N')
ax.set_ylabel('log(Relative entropy)')
plt.show()