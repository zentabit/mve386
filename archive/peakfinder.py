#Namnge saker är svårt

import test_functions
from sampler import Sampler
from sampling_randMeshUnif import Uniform2DMeshSampler
from sampling_randUnif import UniformSampler
from lib.sampling_lh import LHSampler

from test_functions import *


def findPeak(f, s : Sampler, batchSize, iterationLimit, verbose=False):
    
   
    if not s.canSample():
        print("Cannot sample!")
        return
    
    doLoop = True
    
    maxValue = 0
    maxPoint = None
    
    # Iterations without improvement
    iwi = 0
    
    while(doLoop):
        iwi += 1
        
        pts = s.sample(batchSize)
       
        # Om inte f är vektoriserad
        for p in pts:
            val = f(p)
            
            if val > maxValue:
                maxPoint = p
                maxValue = val
                iwi = 0
                
        
        
        doLoop = (s.canSample() and (iwi < iterationLimit))
    
    if verbose:
        print("{}: Maximum value {} found at {} for {}".format(f.__name__,maxValue,maxPoint, type(s).__name__))
    
    return maxValue, maxPoint
    
    
    
# Loop
# Sample points
# Calculate values of sampled points
# Save max if > previous max
# Repeat until X iterations without improvement OR cannot sample more

def testF(p):
    return sum(p)

def main():
    
    r = Uniform2DMeshSampler(1000) # 2D grid of 1000x1000 points
    r2 = UniformSampler(2) # 2D grid
    r3 = LHSampler(2) # 2D grid
    
    batchSize = 10
    ilwi = 10 # Iteration limit without improvement
    
    #findPeak(w_gauss2d, r, batchSize, ilwi, True)
    #findPeak(w_gauss2d, r2, batchSize, ilwi, True)
    #findPeak(w_gauss2d, r3, batchSize, ilwi, True)
    
    
    iterations = 100
    
    avg = 0
    for _ in range(0,iterations):
        val, _ = findPeak(w_gauss2d, r3, batchSize, ilwi)
        
        avg += val    
    
    avg /= iterations
    
    print("Avg:{}".format(avg)) 
    
    
if __name__ == '__main__':
    main()