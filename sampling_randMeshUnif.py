import numpy as np
import matplotlib.pyplot as plt
from random import sample

batchSize = 10

resolution = 10
stepSize = 1/resolution

x1,x2 = np.mgrid[0:1:stepSize, 0:1:stepSize]
pos = np.dstack((x1,x2))
allPos = pos.reshape(resolution*resolution,2)



def randMeshUnifSample(pointList, batchSize):
    # All points have an equal probability of being sampled
    
    nPoints = pointList.shape[0]  # Number of available points to sample
    
    if batchSize >= nPoints:
        return range(0, nPoints)
    
    sampledIndex = sample(range(0,nPoints), batchSize)
    
    return sampledIndex


def plotPoints(pointList):
    pts = np.transpose(pointList)

    x = pts[0]
    y = pts[1]
    
    plt.scatter(x,y,marker="x")
    
    
plt.ion() # Disable plot locking code execution
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()


# Create copies of array to work on
positionArray = np.copy(allPos)
selectedArray = np.empty((1,2))

while(positionArray.size != 0):
    ind = randMeshUnifSample(positionArray, batchSize)
    
    selectedArray = np.append(selectedArray, positionArray[ind],0)

    plotPoints(positionArray[ind])
    positionArray = np.delete(positionArray, ind,0)
    input("[Enter] to sample next set of points")
    

input("[Enter] to exit (and close plot)")
