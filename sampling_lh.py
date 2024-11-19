import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube

batchSize = 10

lh = LatinHypercube(2)


def plotPoints(pointList):
    pts = np.transpose(pointList)

    x = pts[0]
    y = pts[1]
    
    plt.scatter(x,y,marker="x")
    
    
plt.ion() # Disable plot locking code execution
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

while(True): # CTRL + C to break loop in terminanl

    plotPoints(lh.random(batchSize))



    input("[Enter] to generate new points")