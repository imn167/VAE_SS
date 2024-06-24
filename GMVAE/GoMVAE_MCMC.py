import sys 
import time

sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS')

sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/figures_ss')

from function.EM import *
from function.VAE_GoM import *


start = time.time()
#importing data 
two_mode = np.load('../two_component_truncated.npy') #10000 x 20
#print(two_mode)
N, d = two_mode.shape

####### Establishing a Latent spae for the data ###########

encoder = Encoder(d,2, True)
decoder = Decoder(d,2, True)

ae = AutoEncoder(encoder, decoder)
ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3))
ae.fit(two_mode,epochs=110, batch_size=150, shuffle=True, verbose = 0)

z_mean, _, _ = encoder(two_mode)


prior = MoGPrior(2,15) # 35 gaussienne diagonales dnas R2
w_t, mu_t, sigma2_t, n_iter = EM(z_mean, prior, 80, 1e-2)
print(w_t, n_iter)



mixture_plot(z_mean.numpy(), w_t, mu_t, sigma2_t, min(z_mean.numpy()[:,0])-2, max(z_mean.numpy()[:,0])+2, min(z_mean.numpy()[:,1])-2, max(z_mean.numpy()[:,1])+2 )


prior.means.assign(tf.constant(mu_t, dtype=tf.float32))
prior.logvars.assign(tf.math.log(tf.constant(sigma2_t, dtype=tf.float32)))
prior.w.assign(tf.constant(w_t.reshape(1,-1), dtype=tf.float32))


vae = VAE(encoder, decoder, prior, name_prior = 'GoM')
vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
vae.fit(two_mode, epochs = 150, batch_size = 100, shuffle = True) 

vae.save('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/GMVAE/vae.keras')
z_mean, _, z_variationnel = encoder(two_mode)

mixture_plot(z_variationnel.numpy(), w_t, mu_t, sigma2_t, min(z_variationnel.numpy()[:,0]), max(z_variationnel.numpy()[:,0]), min(z_variationnel.numpy()[:,1]),
              max(z_variationnel.numpy()[:,1]) )

print(time.time() - start)

ColDist = [ot.Normal(np.array(mu), np.exp(0.5*np.array(sigma))) for mu, sigma in zip(prior.means, prior.logvars)]
weight = np.array(tf.nn.softmax(prior.w, axis =1)).reshape(-1)
myMixture = ot.Mixture(ColDist, weight)
z =myMixture.getSample(10000)
z= np.array(z)
mean_x, log_var_x = decoder(z) #we get each mean and log variance of the several distribution then we sample from it

sample = np.random.normal(loc=mean_x, scale= tf.exp(log_var_x/2))


xx= np.linspace(-5,5, 1000)
dist1 = ot.TruncatedDistribution(ot.Normal(1), 2., ot.TruncatedDistribution.LOWER)
dist2 = ot.TruncatedDistribution(ot.Normal(1), -2., ot.TruncatedDistribution.UPPER)
dist = ot.Mixture([dist1,dist2])

plt.figure(figsize = (12,7))
for i in range(2,d):
  plt.subplot(5,5,i+1)
  plt.hist(sample[:,i], bins = 100, density=True);
  plt.plot(xx, sp.norm.pdf(xx))

plt.subplot(5,5,1)
plt.hist(sample[:,0], bins = 100, density=True);
plt.plot(xx, dist.computePDF(xx.reshape(-1,1)))

plt.subplot(5,5,2)
plt.hist(sample[:,1], bins = 100, density=True);
plt.plot(xx, dist.computePDF(xx.reshape(-1,1)))

plt.savefig('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/figures_ss/Truncated_Gaussian_reconstruction.png')
plt.show()

#### 2d plot, truncated area 
plt.hist2d(sample[:, 0], sample[:, 1], bins= (100, 100), cmap = plt.cm.jet)
plt.show()

def truncated_density(x, rv, n_comp, distri):
  return rv.pdf(x[n_comp : ]) * distri.computePDF(x[0]) * distri.computePDF(x[1])

#### MCMC configuration with the vae as proposal 
#approximation de la loi empirique du mélange 
Nc = 10000
GoM = prior.sampling() #getting the Gaussian mixture sampler 
Z = np.array(GoM.getSample(Nc))
idx = np.random.choice(np.arange(Nc), size = Nc)

Z_N = Z[idx, :] #sachant Z, Z_N suit la loi empirique, on l'utilise pour approximer le mélange infini

mean_x, logvar_x  = decoder(Z_N) #lois conditionnelles 
gaussians = [ot.Normal(mu, sigma) for mu, sigma in zip(mean_x.numpy(), np.exp(logvar_x.numpy() * 0.5))]
inf_mixture = ot.Mixture(gaussians, np.ones(Nc)/Nc)
M = 1000 #taille chaine MCMC
chain = np.zeros((M+1 , d))


print(len(gaussians))

chain[0] = two_mode[6]
rv = sp.multivariate_normal(mean = np.zeros(d-2))
acceptance = 0
for i in range(M):
  
  r =  np.random.choice(np.arange(Nc))
  zr = Z_N[r].reshape(1,-1)
  mu, logvar =  decoder(zr) # d 
  rv_gom = sp.multivariate_normal(mean = mu.numpy().reshape(-1), cov = np.exp(logvar.numpy()).reshape(-1))
  candidat = rv_gom.rvs() #d
  ratio_traj = list()
  ratio = truncated_density(candidat, rv, 2, dist) * inf_mixture.computePDF(chain[i]) / (truncated_density(chain[i], rv, 2, dist) * inf_mixture.computePDF(candidat))
  ratio_traj.append(ratio)
  u = np.random.uniform()
  if u < ratio:
    chain[i+1] = candidat
    acceptance += 1 
    gaussians.append(rv_gom)
  else :
    chain[i+1] = chain[i]

print(acceptance/M)
plt.figure(figsize = (15, 7))
for i in range(d):
  plt.subplot(4,5, i+1)
  plt.plot(chain[:, i])
plt.show()

plt.hist(chain[:, 0], density=True, bins=100)
plt.show()
plt.hist(chain[:, 1], density = True, bins=100)
plt.show()
plt.hist(chain[:, 2], density = True, bins= 100)
plt.show()

np.save('GoM_MCMC_chain.npy', chain)
np.save('GoM_ratio_MCMC.npy', np.array(ratio_traj))