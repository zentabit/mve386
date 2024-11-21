import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sampler import Sampler

class Uniform2DMeshSampler(Sampler):
    def __init__(self, res):
        self.res = res
        stepSize = 1/res
        
        x1,x2 = np.mgrid[0:1:stepSize, 0:1:stepSize]
        pos = np.dstack((x1,x2))
        
        self.gridPoints = pos.reshape(res*res,2)
        self.sampledPoints = np.empty((1,2))

    def sample(self, n):
        
        nPoints = self.gridPoints.shape[0] 
        
        ind = []
        if n >= nPoints:
            ind = range(0, nPoints)
        else:
            ind = sample(range(0,nPoints), n)
        
        self.sampledPoints = np.append(self.sampledPoints, self.gridPoints[ind],0)
        
        temp = np.copy(self.gridPoints[ind])
        
        self.gridPoints = np.delete(self.gridPoints, ind,0)
        
        return temp

    def canSample(self):
        return (len(self.gridPoints) > 0)
    
    def getSampledPoints(self):
        return self.sampledPoints
    
    def getSampledPercentage(self):
        return len(self.sampledPoints) / (len(self.sampledPoints) + len(self.gridPoints))


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
    
def main():
    
    
    batchSize = 10

    resolution = 10
    stepSize = 1/resolution

    x1,x2 = np.mgrid[0:1:stepSize, 0:1:stepSize]
    pos = np.dstack((x1,x2))
    allPos = pos.reshape(resolution*resolution,2)

    
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

if __name__ == '__main__':
    main()