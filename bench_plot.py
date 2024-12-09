# This program takes a CSV file from the benchmarking scripts and plots it in linear-log scale.
import numpy as np
import matplotlib.pyplot as plt

acqf = "ExpectedImprovement"
timestamp = "1733734363.269824"
fname = f"points_benchmark-{acqf}-{timestamp}.csv"

ax = plt.axes()

vals = np.loadtxt(fname, skiprows=1, delimiter=',')
print([ str(label) for label in vals[:, 0] ])
ax.set_yscale('log')
ax.boxplot(np.transpose(vals[ :, 1: ]),  tick_labels =[ str(int(label)) for label in vals[:, 0] ] )
ax.set_title("Hej")
ax.set_xlabel('N')
ax.set_ylabel('log(Relative entropy)')
plt.show()