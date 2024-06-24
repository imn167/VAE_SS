#MCMC chain visualisation with Vamprior 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as sp
import openturns as ot

chain = np.load('mcmcChain.npy')
nchain, d = chain.shape
from statsmodels.graphics import tsaplots


##### latent space ####
z_variationnel = np.load('z_variationnel.npy')
ps_mean = np.load('ps_mean.npy')
ps_logvar = np.load('ps_logvar.npy')

def plot_contour(samples, mean, logvar):
  K = np.shape(mean)[0] #number of mixture 
  x = np.linspace(np.min(mean[:,0])-3, np.max(mean[:, 0])+3, 1000)
  y = np.linspace(np.min(mean[:,1])-3, np.max(mean[:, 1])+3, 1000)
  X, Y = np.meshgrid(x,y)
  pos = np.dstack((X,Y))
  gaussians = [sp.multivariate_normal(mu, np.diag(sigma)) for mu, sigma in zip(mean, np.exp(logvar))]
  mix = np.zeros((1000, 1000))
  for i in range(K):
    mix += np.array( gaussians[i].pdf(pos))
    plt.contour(X, Y,gaussians[i].pdf(pos), levels = [.1,.2,.5, .8])
  #plt.contour(X, Y,mix/K, levels = [.1,.2,.5, .8])
  plt.scatter(samples[:, 0], samples[:, 1], s = 6)
  plt.savefig('../Vamprior/latent_space_Vamprior.png')
  plt.show()
  
plot_contour(z_variationnel, ps_mean, ps_logvar)

M = chain.shape[0]

chain_lag = chain[np.arange(0,M, 5), :]

fig,ax = plt.subplots(2,int(d/2))
for i in range(int(d/2)):
    tsaplots.plot_acf(chain[:, i], ax = ax[0,i], title= r"$X_{%i}$"%(i+1))
    tsaplots.plot_acf(chain[:, int(d/2)-1+i], ax = ax[1,i], title= r"$X_{%i}$"%(int(d/2)+i+1))
plt.savefig('../Vamprior/autocorrelation.png')
plt.show()



plt.figure(figsize=(15,10))
for i in range(d):
  plt.subplot(4, int(d/4), i+1)
  plt.hist(chain[:, i], density=True, bins=100)
plt.savefig('../Vamprior/marginale_hist.png')
plt.show()


traj_ratio = np.load('ratio-traj.npy')
print(traj_ratio)
plt.plot(traj_ratio.reshape(-1))
plt.hlines(1, 0, len(traj_ratio), 'red', linestyles= "--")
plt.title('MCMC ratio')
plt.savefig('../Vamprior/ratio_traj.png')
plt.show() #enormement de ratio à 0 !!!
