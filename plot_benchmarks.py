import matplotlib.pyplot as plt
import numpy as np

def plotting(acq):
    entropy = np.zeros((5,11))
    batch_sizes = [5,10,15,20,30]
    for i in range(np.size(batch_sizes)):
        data = np.load(f"benchmarks/{acq}/{acq}-{batch_sizes[i]}.npy")
        
        
        # print(data[5,5,0,0,:])
        
        for j in range(data.shape[1]):
            for k in range(data.shape[0]):
                entropy[i,j] = entropy[i,j] + data[k,j,0,0,-2]
    entropy = entropy/data.shape[0]
    plt.imshow(entropy)
    plt.colorbar()
    plt.title(acq)

            
acqusiotion_names = ["CB", "UCB", "EI", "GP_UCB", "RGP_UCB"]
for i in enumerate(acqusiotion_names):
    plotting(i[1])
    if (i[0] != 4):
        plt.figure()
plt.show()

