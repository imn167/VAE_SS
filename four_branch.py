#Coding the 4_branch problem

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as sp
import time 

################## Small dimension :: 2-Dimension #################################

def four_branch(X):
    if type(X) is tuple:
        quant1 = np.expand_dims((X[0]-X[1])**2 /10 - (X[0] + X[1]) / np.sqrt(2) + 3, 2)
        quant2 = np.expand_dims((X[0]-X[1])**2 /10 + (X[0] + X[1]) / np.sqrt(2) + 3,2)
        quant3 = np.expand_dims((X[0]-X[1]) + 7/ np.sqrt(2) ,2)
        quant4 = np.expand_dims((X[1]-X[0]) + 7/ np.sqrt(2) ,2)

        tensor = np.concatenate([quant1, quant2, quant3, quant4], axis =2 )
        minimum = -np.min(tensor, axis=2)
    else :
        quant1 = np.expand_dims((X[0]-X[1])**2 /10 - (X[0] + X[1]) / np.sqrt(2) + 3, 1)
        quant2 = np.expand_dims((X[0]-X[1])**2 /10 + (X[0] + X[1]) / np.sqrt(2) + 3,1)
        quant3 = np.expand_dims((X[0]-X[1]) + 7/ np.sqrt(2) ,1)
        quant4 = np.expand_dims((X[1]-X[0]) + 7/ np.sqrt(2) ,1)

        tensor = np.concatenate([quant1, quant2, quant3, quant4], axis =1 )
        minimum = -np.min(tensor, axis=1)

    return minimum


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
cs = ax.contour(pos[0], pos[1], four_branch(pos), levels = [0])
ax.clabel(cs, cs.levels, inline=True, fontsize=15)
plt.show()

#Prposal class for the 4_branch Numerical exemple 

class gaussian_kernel():
    def __init__(self, sd,  **kwargs):
        super(gaussian_kernel,self).__init__(**kwargs)
        self.rv = sp.norm
        self.sd = sd

    def call(self, X):
        samples = self.rv.rvs(loc = X, scale = self.sd)
        return samples 

    def density(self, x): #the X distribution is a product of standard gaussian 
        return self.rv.pdf(x)

proposal = gaussian_kernel(.1)
#print(proposal.call(np.ones(12)))

##########################################################

import sys 

sys.path.append('.../function')

from function.SS_class import MMA, subset_simulation, CMC

modified_metropolis = MMA(proposal)
N = 10000
samples = np.random.normal(size = (2,N))

############# Credule Monte Carlo ###############

cmc = list()
for i in np.arange(0,N, 10):
    cmc.append(np.mean(four_branch(samples[:, :i]) >= 0))

plt.plot(cmc)
plt.hlines(2.22e-3, 0, 1000, colors='red', linestyles='--')
plt.savefig('.../figures_ss/CMC_traj.png')

start = time.time()
sequence, k, failure, quantile, accep_rate = modified_metropolis.call(samples, 0, four_branch)
time.time() - start 

print(k, failure, quantile)


plt.figure(figsize= (15,5))
for i in range(len(sequence)):
    plt.subplot(1,len(sequence), i+1)
    plt.plot(sequence[i][0], label = r"$X_1^%i$"%(i+1))
    plt.legend()


plt.figure(figsize= (15,5))
for i in range(len(sequence)):
    plt.subplot(1,len(sequence), i+1)
    plt.plot(sequence[i][1], label = r"$X_1^%i$"%(i+1))
    plt.legend()

plt.show()

first_simu = sequence[0]
last_simu = sequence[-1]

fig, ax = plt.subplots()
pc = ax.pcolormesh(pos[0], pos[1], four_branch(pos))
fig.colorbar(pc)
cs = ax.contour(pos[0], pos[1], four_branch(pos), levels = [0])
ax.scatter(last_simu[0], last_simu[ 1], c = "red", s= 6)
ax.scatter(first_simu[0], first_simu[ 1], c = "orange", s= 6)
ax.clabel(cs, cs.levels, inline=True, fontsize=15)
ax.set_title('Estimation with Modified Metropolis')
fig.savefig('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/figures_ss/MMA_10000.png')
plt.show()


sequence, n_event, failure, quantile, rate = subset_simulation(samples, 0, four_branch, 1 )

print(failure, n_event)


plt.figure(figsize= (15,5))
for i in range(len(sequence)):
    plt.subplot(1,(len(sequence)), i+1)
    plt.plot(sequence[i][0], label = r"$X_1^%i$"%(i+1))
    plt.legend()


plt.figure(figsize= (15,5))
for i in range(len(sequence)):
    plt.subplot(1,len(sequence), i+1)
    plt.plot(sequence[i][1], label = r"$X_1^%i$"%(i+1))
    plt.legend()

plt.show()

first_simu = sequence[0]
last_simu = sequence[-1]

fig, ax = plt.subplots()
pc = ax.pcolormesh(pos[0], pos[1], four_branch(pos))
fig.colorbar(pc)
cs = ax.contour(pos[0], pos[1], four_branch(pos), levels = [0])
ax.scatter(last_simu[0], last_simu[ 1], c = "red", s= 6)
ax.scatter(first_simu[0], first_simu[1], c = 'orange', s = 6)
ax.clabel(cs, cs.levels, inline=True, fontsize=15)
ax.set_title("Estimation with Vanilla SS")
fig.savefig('.../figures_ss/Vanilla_SS_10000.png')
plt.show()



def four_branch_HD(X, beta =0):
    d, N = X.shape 
    quant1 = beta + np.expand_dims(np.sum(X, axis=0) / np.sqrt(d), axis=1)
    quant2 = beta - np.expand_dims(np.sum(X, axis=0) / np.sqrt(d), axis= 1 )
    quant3 = beta + np.expand_dims((np.sum(X[: int(d/2), :], axis= 0) - np.sum(X[int(d/2)+1 :, :], axis= 0)) / np.sqrt(d), axis=1)
    quant4 = beta - np.expand_dims((np.sum(X[: int(d/2), :], axis= 0) - np.sum(X[int(d/2)+1 :, :], axis= 0)) / np.sqrt(d), axis=1)

    tensor = np.concatenate([quant1, quant2, quant3, quant4], axis=1)

    return - np.min(tensor, axis = 1)


samples = np.random.normal(size = (10, N))
pt = 9.3e-4
t = 3.5
cmc = CMC(N)
print(cmc(four_branch_HD, samples, t))

cmc.plot_trajectory(pt,cmc(four_branch_HD, samples, t) )