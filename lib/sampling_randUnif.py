import numpy as np
import matplotlib.pyplot as plt

from .sampler import Sampler

class UniformSampler(Sampler):
    
    def __init__(self, dim:int):
        self.dim = dim
    
    def sample(self, n):
        return np.random.rand(n, self.dim)
    

def randUnifSample(dim, batchSize):
    return np.random.rand(batchSize,dim)

def plotPoints(pointList):
    pts = np.transpose(pointList)

    x = pts[0]
    y = pts[1]
    
    plt.scatter(x,y,marker="x")
    
    
def main():
    batchSize = 10
    dim = 2

    
    plt.ion() # Disable plot locking code execution
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

    while(True): # CTRL + C to break loop in terminanl

        plotPoints(randUnifSample(dim, batchSize))



        input("[Enter] to generate new points")