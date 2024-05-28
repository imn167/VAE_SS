import numpy as np 
import matplotlib.pyplot as plt 
import openturns as ot 

from IPython.display import clear_output


#Simulation of a truncated normal distribution 
def truncatedDistribution(n_samples, dim, mean, variance, lower, higher):
    sd = np.sqrt(variance)
    L = list()
    i = 0 
    while i < n_samples:
        prop = np.random.normal(loc = mean, scale=sd)
        prop_norm = np.linalg.norm(prop[0:2], ord=-np.inf)

        if prop_norm > higher : 
            L.append(prop)
            i += 1
            if i %10 == 0:
                clear_output(wait=True)
                print("boucle %d terminéE" %(i))

    return np.array(L)

dist = truncatedDistribution(10000, 20, np.zeros(20), np.ones(20), 2, 2)
dist.shape

np.save('2_component_truncated.npy', dist)