#Coding the 4_branch problem

import numpy as np 
import matplotlib.pyplot as plt 

def four_branch(X):
    
    quant1 = np.expand_dims((X[0]-X[1])**2 /10 - (X[0] + X[0]) / np.sqrt(2) + 3, 2)
    quant2 = np.expand_dims((X[0]-X[1])**2 /10 + (X[0] + X[0]) / np.sqrt(2) + 3,2)
    quant3 = np.expand_dims((X[0]-X[1]) + 7/ np.sqrt(2) ,2)
    quant4 = np.expand_dims((X[1]-X[0]) + 7/ np.sqrt(2) ,2)

    tensor = np.concatenate([quant1, quant2, quant3, quant4], axis =2 )

    return np.min(tensor, axis=2)#np.min([quant1, quant2 , quant3, quant4] )


#X according to a standard gaussian 

def construction_grid(xmin, xmax, ymin, ymax, npoints):
    x = np.linspace(xmin, xmax, npoints)
    y = np.linspace(ymin, ymax, npoints)

    
    #grid = np.dstack((X,Y))
    return np.meshgrid(x,y)

pos = construction_grid(-6,6,-6,6, 1000)


#plotting 
fig, ax = plt.subplots()

pc = ax.pcolormesh(pos[0], pos[1], four_branch(pos))
fig.colorbar(pc)
cs = ax.contour(pos[0], pos[1], four_branch(pos), levels = [-1.5])
ax.clabel(cs, cs.levels, inline=True, fontsize=15)
plt.show()

