#Namnge saker är svårt

import test_functions
from sampler import Sampler
from sampling_randMeshUnif import Uniform2DMeshSampler
from sampling_randUnif import UniformSampler
from sampling_lh import LHSampler


def findPeak(f, s : Sampler, batchSize, iterationLimit):
    
   
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
    
    print("Maximum value {} found at {}".format(maxValue,maxPoint))
    
    
    
# Loop
# Sample points
# Calculate values of sampled points
# Save max if > previous max
# Repeat until X iterations without improvement OR cannot sample more

def testF(p):
    return sum(p)

def main():
    
    r = Uniform2DMeshSampler(100)
    r2 = UniformSampler(2)
    r3 = LHSampler(2)
    
    findPeak(testF, r3, 10, 2)
    
    
if __name__ == '__main__':
    main()