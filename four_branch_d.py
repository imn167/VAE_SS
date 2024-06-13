import numpy as np
from function.SS_class import *
import scipy.stats as sp 
import time 
################ Script pour le cas lineaire du 4-branches ###############

#################### TEST DU MODELE DANS LE CAS 2D ########################

def four_branch(X):
   
    d, N = X.shape 
    quant1 =  np.expand_dims(np.sum(X, axis=0) / np.sqrt(d), axis=1)
    quant2 = - np.expand_dims(np.sum(X, axis=0) / np.sqrt(d), axis= 1 )
    quant3 = + np.expand_dims((np.sum(X[: int(d/2), :], axis= 0) - np.sum(X[int(d/2) :, :], axis= 0)) / np.sqrt(d), axis=1)
    quant4 = - np.expand_dims((np.sum(X[: int(d/2), :], axis= 0) - np.sum(X[int(d/2) :, :], axis= 0)) / np.sqrt(d), axis=1)

    tensor = np.concatenate([quant1, quant2, quant3, quant4], axis=1)
    minimum = np.min(tensor, axis = 1)
        

    return - minimum


###### 4-branch Monte Carlo #######
N= 10000
samples = np.random.normal(size = (2, N))
pt = 9.3e-4
t = 3.5

def construction_grid(xmin, xmax, ymin, ymax, npoints):
    x = np.linspace(xmin, xmax, npoints)
    y = np.linspace(ymin, ymax, npoints)

    
    #grid = np.dstack((X,Y))
    return np.meshgrid(x,y)

pos = construction_grid(-8,8,-8,8, 1000)


def four_branch_2d(X, linear = False):
    if linear:
        quant1 = np.expand_dims((X[0]+X[1]) / np.sqrt(2), 1)
        quant2 = np.expand_dims((-X[0]-X[1])/ np.sqrt(2),1)
        quant3 = np.expand_dims((X[0]-X[1]) / np.sqrt(2) ,1)
        quant4 = np.expand_dims((X[1]-X[0]) / np.sqrt(2) ,1)

        tensor = np.concatenate([quant1, quant2, quant3, quant4], axis =1 )
        minimum = np.min(tensor, axis=1)
    else :
        quant1 = np.expand_dims((X[0]-X[1])**2 /10 - (X[0] + X[1]) / np.sqrt(2) + 3, 1)
        quant2 = np.expand_dims((X[0]-X[1])**2 /10 + (X[0] + X[1]) / np.sqrt(2) + 3,1)
        quant3 = np.expand_dims((X[0]-X[1]) + 7/ np.sqrt(2) ,1)
        quant4 = np.expand_dims((X[1]-X[0]) + 7/ np.sqrt(2) ,1)

        tensor = np.concatenate([quant1, quant2, quant3, quant4], axis =1 )
        minimum = np.min(tensor, axis=1)

    return -minimum


from function.SS_VAE import * 
d = 10
N = 10000
rv = sp.multivariate_normal()
ss_vae = SS_VAE(2,d, rv)


##### Monte Carlo simple #######
cmc = CMC(N)
print(cmc(four_branch, samples, t))

cmc.plot_trajectory(pt,cmc(four_branch, samples, t) )


sequence, k, failure, quantile, accep_rate = subset_simulation(samples, t, four_branch, .5)

print(quantile, failure)

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
pc = ax.pcolormesh(pos[0], pos[1], four_branch_2d(pos, linear=True))
fig.colorbar(pc)
cs = ax.contour(pos[0], pos[1], four_branch_2d(pos, linear=True), levels = [t])
ax.scatter(last_simu[0], last_simu[ 1], c = "red", s= 6)
ax.scatter(first_simu[0], first_simu[ 1], c = "orange", s= 6)
ax.set_title("Linear 4-branch with Vanilla SS")
fig.savefig('.../figures_ss/Linear_VanillaSS.png')
plt.show()




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

proposal = gaussian_kernel(.5)

modified_metropolis = MMA(proposal)


start = time.time()
sequence, k, failure, quantile, accep_rate = modified_metropolis.call(samples, t, four_branch)
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
pc = ax.pcolormesh(pos[0], pos[1], four_branch_2d(pos, True))
fig.colorbar(pc)
cs = ax.contour(pos[0], pos[1], four_branch_2d(pos, True), levels = [t])
ax.scatter(last_simu[0], last_simu[ 1], c = "red", s= 6)
ax.scatter(first_simu[0], first_simu[ 1], c = "orange", s= 6)
ax.clabel(cs, cs.levels, inline=True, fontsize=15)
ax.set_title('Estimation with Modified Metropolis')
fig.savefig('.../figures_ss/Linear_MMA.png')
plt.show()

time.time() - start



