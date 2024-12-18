# This program takes a CSV file from the benchmarking scripts and plots it in linear-log scale.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import sys
plt.rcParams.update({'font.size': 14})

def plot1d(vals, log, ax, fig):
    ax.set_yscale('log')

    # Ensure grid lines are below all elements
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='gray', linestyle='solid', which='major', zorder=0, alpha = 0.5)
    ax.yaxis.grid(True, color='lightgray', linestyle='solid', which='minor', alpha=0.3, zorder=0)
    
    # Create the boxplot
    bp = ax.boxplot(vals[:, 0, 0, 0, :-1], 
                    tick_labels=log['batch_numbers'][1:], 
                    sym='', 
                    zorder=3)  # Boxplot lines on top

    # Add fills to the boxes manually
    for i, box in enumerate(bp['boxes']):
        # Extract the x and y coordinates of the box vertices
        x_left = box.get_xdata()[0]  # Left edge of the box
        x_right = box.get_xdata()[2]  # Right edge of the box
        y_bottom = box.get_ydata()[1]  # Bottom edge
        y_top = box.get_ydata()[2]  # Top edge

        # Compute box width and center
        box_width = x_right - x_left
        x_center = (x_right + x_left) / 2

        # Create a rectangle patch to fill the box
        rect = Rectangle((x_center - box_width / 2, y_bottom),  # Bottom-left corner
                         width=box_width, 
                         height=y_top - y_bottom,
                         facecolor='skyblue',  # Fill color
                         edgecolor='none',  # No border for the patch
                         zorder=2.5)  # Fill behind the lines but above grid
        ax.add_patch(rect)

    # Labels and title
    ax.set_title(fr"GP-UCB, batch size {log['batch_size']}, $\{list(log['args'][0].keys())[0]} = {list(log['args'][0].values())[0]}$")
    ax.set_xlabel('N')
    ax.set_ylabel(r"$m(\mu_D)$")
    ax.set_ylim(1e-5, 1.2)
    print(np.array([ l.get_ydata()[0] for l in bp['medians'] ]))
    print(np.array([bp['whiskers'][i].get_ydata()[1] for i in range(1, len(bp['whiskers']), 2)]))

    fig.set_dpi(300)
    fig.tight_layout()

def plot1d_unif(vals, log, ax, fig):
    ax.set_yscale('log')

    # Ensure grid lines are below all elements
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='gray', linestyle='solid', which='major', zorder=0, alpha = 0.5)
    ax.yaxis.grid(True, color='lightgray', linestyle='solid', which='minor', alpha=0.3, zorder=0)
    
    # Create the boxplot
    bp = ax.boxplot(vals[ :, 0, :, 0, 0 ], tick_labels = [str(i**3) for i in range(1, 8)], sym = '', zorder = 3)
    

    # Add fills to the boxes manually
    for i, box in enumerate(bp['boxes']):
        # Extract the x and y coordinates of the box vertices
        x_left = box.get_xdata()[0]  # Left edge of the box
        x_right = box.get_xdata()[2]  # Right edge of the box
        y_bottom = box.get_ydata()[1]  # Bottom edge
        y_top = box.get_ydata()[2]  # Top edge

        # Compute box width and center
        box_width = x_right - x_left
        x_center = (x_right + x_left) / 2

        # Create a rectangle patch to fill the box
        rect = Rectangle((x_center - box_width / 2, y_bottom),  # Bottom-left corner
                         width=box_width, 
                         height=y_top - y_bottom,
                         facecolor='skyblue',  # Fill color
                         edgecolor='none',  # No border for the patch
                         zorder=2.5)  # Fill behind the lines but above grid
        ax.add_patch(rect)

    # Labels and title
    ax.set_title(fr"Uniform sampling with GPR")
    # ax.set_title(fr"Uniform sampling with linear regression")
    ax.set_xlabel('N')
    ax.set_ylabel(r"$m(\mu_D)$")
    ax.set_ylim(1e-5, 1.2)
    print(np.array([ l.get_ydata()[0] for l in bp['medians'] ]))
    print(np.array([bp['whiskers'][i].get_ydata()[1] for i in range(1, len(bp['whiskers']), 2)]))

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
    plt.savefig(f"{fname}.svg")
    plt.show()

    

main()