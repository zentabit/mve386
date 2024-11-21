import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube

from sampler import Sampler

class LHSampler(Sampler):
    
    def __init__(self, dim:int):
        self.lh = LatinHypercube(dim)
    
    def sample(self, n):
        return self.lh.random(n)



def plotPoints(pointList):
    pts = np.transpose(pointList)

    x = pts[0]
    y = pts[1]
    
    plt.scatter(x,y,marker="x")


def main():
    batchSize = 10

    lh = LatinHypercube(2)

        
    plt.ion() # Disable plot locking code execution
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

    while(True): # CTRL + C to break loop in terminanl

        plotPoints(lh.random(batchSize))



        input("[Enter] to generate new points")