import matplotlib.pyplot as plt
import numpy as np

def plotPoints(pointList):
    pts = np.transpose(pointList)

    x = pts[0]
    y = pts[1]
    
    plt.scatter(x,y,marker="x")

def unifrefine(d, nin, refine):
    x = [np.arange(1/2**refine, d, 1/2**refine) for _ in range(nin)]
    grid = np.meshgrid(*x)
    return grid

def unifspacing(d, nin, n): # n is the number of points in each dimension, the spacing will be 1/(n+1)
    delta = 1/(n+1)
    x = [np.arange(delta, d, delta) for _ in range(nin)]
    grid = np.meshgrid(*x)
    return grid

def main():
    X = unifspacing(1, 2, 4)
    print(np.shape(X))
    print(X)

    # plotPoints((X,Y))
    plt.show()

# main()