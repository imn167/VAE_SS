#MCMC chain visualisation with Vamprior 
import numpy as np 
import matplotlib.pyplot as plt 

chain = np.load('mcmcChain.npy')
nchain, d = chain.shape
from statsmodels.graphics import tsaplots

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
