# This program takes a CSV file from the benchmarking scripts and plots it in linear-log scale.
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
plt.rcParams.update({'font.size': 14})

def plot1d(vals, log, ax, fig):
    ax.set_yscale('log')
    # ax.grid()
    ax.boxplot(vals[ :, 0, 0, 0, :-1 ], tick_labels = log['batch_numbers'][1:], sym = '' )
    ax.set_title(fr"UCB, batch size {log['batch_size']}, $\{list(log['args'][0].keys())[0]} = {list(log['args'][0].values())[0]}$") # cursed
    ax.set_xlabel('N')
    ax.set_ylabel(r"$m(\mu_D)$")
    ax.set_ylim(1e-4,1.2)
    fig.set_dpi(300)
    fig.tight_layout()

def plot1d_unif(vals, log, ax, fig):
    ax.set_yscale('log')
    # ax.grid()
    ax.boxplot(vals[ :, 0, :, 0, 0 ], tick_labels = [str(i**3) for i in range(1, 8)], sym = '' )
    ax.set_title(fr"Uniform sampling with GPR")
    ax.set_xlabel('N')
    ax.set_ylabel(r"$m(\mu_D)$")
    ax.set_ylim(1e-4,1.2)
    fig.set_dpi(300)
    fig.tight_layout()

def plot2d(vals, log, ax, fig):
    vals_mean_func = np.mean(vals, axis = 0)
    print(np.shape(vals_mean_func))
    fig.tight_layout()
    im = ax.imshow(vals_mean_func[: , 0, 0, : ], norm = 'log')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks(np.arange(0,log['batches'],1))
    ax.set_xticklabels(log['batch_numbers'])
    fig.colorbar(im, ax = ax)
    

def main():
    fname = sys.argv[1]
    vals = np.load(f"{fname}.npy")
    with open(f"{fname}.json") as f:
        log = json.load(f)
    
    print(log)

    fig = plt.figure()
    ax = plt.axes()
    plot1d(vals, log, ax, fig)
    print(vals[:, 0,0,0,-1])
    plt.savefig(f"{fname}.svg")
    plt.show()

    

main()