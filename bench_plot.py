# This program takes a CSV file from the benchmarking scripts and plots it in linear-log scale.
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def plot1d(vals, log, ax, fig):
    ax.set_yscale('log')
    ax.boxplot(vals[ :, 0, 0, 0, : ], tick_labels = log['batch_numbers'] )
    ax.set_title("Hej")
    ax.set_xlabel('N')
    ax.set_ylabel('Relative entropy')
    fig.tight_layout()

def plot2d(vals, log, ax, fig):
    vals_mean_func = np.mean(vals, axis = 0)
    print(np.shape(vals_mean_func))
    ax.imshow(vals_mean_func[: , 0, 0, : ], norm = 'log')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(log['batch_numbers'])
    fig.tight_layout()

def main():
    fname = sys.argv[1]
    vals = np.load(f"{fname}.npy")
    with open(f"{fname}.json") as f:
        log = json.load(f)
    
    print(log)

    fig = plt.figure()
    ax = plt.axes()
    plot2d(vals, log, ax, fig)
    plt.show()

    

main()