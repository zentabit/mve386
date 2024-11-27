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


def main():
    X = unifrefine(1, 2, 3)
    print(np.shape(X))
    # print(Y)

    # plotPoints((X,Y))
    plt.show()