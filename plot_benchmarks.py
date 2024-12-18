import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import math

plt.rcParams.update({'font.size': 12})
prettynames = {'UCB':'(a) UCB', 'EI':'(b) EI', 'CB':'(e) COV', 'GP_UCB':'(c) GP-UCB', 'RGP_UCB':'(d) RGP-UCB'}

def plotting(acq, ax):
    if (acq[1] == "CB"):
        entropy = np.zeros((5,1))
    else:
        entropy = np.zeros((5,11))
    batch_sizes = [5,10,15,20,30]
    

    for i in range(np.size(batch_sizes)):
        data = np.load(f"benchmarks/{acq[1]}/{acq[1]}-{batch_sizes[i]}.npy")
        
        for j in range(data.shape[1]):
            for k in range(data.shape[0]):
                entropy[i,j] = entropy[i,j] + data[k,j,0,0,-2]
    entropy = entropy/data.shape[0]

    with open(f"benchmarks/{acq[1]}/{acq[1]}-{batch_sizes[i]}.json") as f:
        log = json.load(f)

    
    
    pic = 0
    if acq[1] == 'EI':
        pic = 1
    ax_current = ax[pic]
    ax_current.set_yticks(np.arange(5))
    ax_current.set_yticklabels(['5','10','15','20','30'])
    xticks = [np.round(list((d.values()))[0],2) for d in log["args"]]
    ax_current.set_xticks(np.arange(np.size(xticks)))
    ax_current.set_xticklabels(xticks)
    xlabel = list(log["args"][0])
    ax_current.set_xlabel(fr"$\{xlabel[0]}$")
    ax_current.set_ylabel('Batch size')
    ax_current.set
    img = ax_current.imshow(entropy, norm = 'log', cmap='hot', vmin=0.012, vmax = 0.12)
    # plt.colorbar()
    

    ax_current.set_title(prettynames[acq[1]])
    
    return img


acqusiotion_names = ["UCB", "EI", "GP_UCB", "RGP_UCB", "CB"]
fig = plt.figure(layout='constrained', figsize=(12, 6))
subfigs = fig.subfigures(2,1,wspace=0.07)

images = []

ax1 = subfigs[0].subplots(1, 2)
for i in enumerate(acqusiotion_names[:2]):
    img = plotting(i, ax1)
    images.append(img)

# Second subfigure with a custom size for the middle plot
sub_ax2 = subfigs[1].add_gridspec(1, 3, width_ratios=[1,1,0.1], wspace = 0)  # Customize the width ratios
ax2_left = fig.add_subplot(sub_ax2[0, 0])
ax2_middle = fig.add_subplot(sub_ax2[0, 1])
ax2_right = fig.add_subplot(sub_ax2[0, 2])

# ax2_middle.set_aspect()

# Plotting for the second subfigure
for i in enumerate(acqusiotion_names[2:]):
    if i[0] == 0:
        img = plotting(i, [ax2_left])
    elif i[0] == 1:
        img = plotting(i, [ax2_middle])
    elif i[0] == 2:
        img = plotting(i, [ax2_right])


plt.colorbar(images[0], ax=ax1, location='right', fraction=0.1, pad=0.04)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)

plt.savefig(f'figures/Benchmarks/total.svg')
plt.show()

