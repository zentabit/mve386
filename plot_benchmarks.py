import matplotlib.pyplot as plt
import numpy as np
import json
import sys

def plotting(acq):
    if (acq == "CB"):
        entropy = np.zeros((5,1))
    else:
        entropy = np.zeros((5,11))
    batch_sizes = [5,10,15,20,30]
    

    for i in range(np.size(batch_sizes)):
        data = np.load(f"benchmarks/{acq}/{acq}-{batch_sizes[i]}.npy")
        
        for j in range(data.shape[1]):
            for k in range(data.shape[0]):
                entropy[i,j] = entropy[i,j] + data[k,j,0,0,-2]
    entropy = entropy/data.shape[0]

    with open(f"benchmarks/{acq}/{acq}-{batch_sizes[i]}.json") as f:
        log = json.load(f)

    
    # fig = plt.figure(layout='constrained', figsize=(10, 4))
    # sub = fig.subplots(2,3,sharey=True)
    ax = plt.axes()
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(['5','10','15','20','30'])
    xticks = [np.round(list((d.values()))[0],2) for d in log["args"]]
    ax.set_xticks(np.arange(np.size(xticks)))
    ax.set_xticklabels(xticks)
    xlabel = list(log["args"][0])
    # print(type(xlabel[0]), list(xlabel[0]))
    ax.set_xlabel(xlabel[0])
    ax.set_ylabel('Batch size')

    plt.imshow(entropy, norm = 'log', cmap='hot')
    plt.colorbar()
    

    plt.title(acq)
    plt.savefig(f'figures/Benchmarks/{acq}.svg')

            
acqusiotion_names = ["CB", "UCB", "EI", "GP_UCB", "RGP_UCB"]
for i in enumerate(acqusiotion_names):
    plotting(i[1])
    if (i[0] != 4):
        plt.figure()
plt.show()

