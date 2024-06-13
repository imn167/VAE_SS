
import numpy as np
import matplotlib.pyplot as plt 

def four_branch(X):
   
    N, d = X.shape 
    quant1 =  np.expand_dims(np.sum(X, axis=1) / np.sqrt(d), axis=1)
    quant2 = - np.expand_dims(np.sum(X, axis=1) / np.sqrt(d), axis= 1 )
    quant3 = + np.expand_dims((np.sum(X[:, : int(d/2)], axis= 1) - np.sum(X[:, int(d/2) :], axis= 1)) / np.sqrt(d), axis=1)
    quant4 = - np.expand_dims((np.sum(X[:, : int(d/2)], axis= 1) - np.sum(X[:, int(d/2) :], axis= 1)) / np.sqrt(d), axis=1)

    tensor = np.concatenate([quant1, quant2, quant3, quant4], axis=1)
    minimum = np.min(tensor, axis = 1)
        

    return - minimum

d = 10
N = 10000
samples = np.random.normal(size = (N,d))
PHI = four_branch(samples)

quantile = np.quantile(PHI, 0.5)

idx = np.where(PHI > quantile)[0]

random_seed = np.random.choice(idx, size=N)

L = np.zeros((50, N, d))


sample_threshold = samples[idx, :]
print(sample_threshold.shape)
L[0] = samples[random_seed, :]
phi_threshold = PHI[ random_seed]

from function.VAE_GoM import *

encoder = Encoder(d, 2, True)
decoder = Decoder(d,2, True)

ae = AutoEncoder(encoder, decoder)
ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3))
ae.fit(sample_threshold ,epochs=150, batch_size=64, shuffle=True, verbose = 0)

plt.figure()
plt.plot(ae.history.history['loss'])
plt.title('Loss trajectory for AutoEncoder')
plt.savefig('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/figures_ss/Truncated_Gaussian_LossAE.png')
plt.show()

z_mean, _, _ = encoder(sample_threshold)

plt.figure(figsize=(12,6))
for i in range(1):
    plt.subplot(1,2, i+1)
    plt.scatter(z_mean[:, i], z_mean[:,i+1], s = 5)
plt.show()

from function.EM import *

start = time.time()
prior = MoGPrior(2,4)
w_t, mu_t, sigma2_t, j = EM(z_mean, prior, 100, 1e-2)
print(w_t, j)
print(time.time() - start)



mixture_plot(z_mean.numpy(), w_t, mu_t, sigma2_t, min(z_mean.numpy()[:,0]), max(z_mean.numpy()[:,0]), min(z_mean.numpy()[:,1]), max(z_mean.numpy()[:,1]) )


prior.means.assign(tf.constant(mu_t, dtype=tf.float32))
prior.logvars.assign(tf.math.log(tf.constant(sigma2_t, dtype=tf.float32)))
prior.w.assign(tf.constant(w_t.reshape(1,-1), dtype=tf.float32))


vae = VAE(encoder, decoder, prior)
vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))
vae.fit(sample_threshold, epochs = 150, batch_size = 128, shuffle = True) 

plt.plot(vae.history.history['kl_loss'])
plt.show()
plt.plot(vae.history.history['reconstruction_loss'])
plt.show()

z_mean,_, _ = encoder(sample_threshold)

myMixture = prior()
z =myMixture.getSample(10000)
z= np.array(z)
mean_x, log_var_x = decoder(z) #we get each mean and log variance of the several distribution then we sample from it

sample = np.random.normal(loc=mean_x, scale= tf.exp(log_var_x/2))

plt.figure(figsize = (12,7))
for i in range(0,d):
  plt.subplot(5,5,i+1)
  plt.hist(sample[:,i], bins = 'auto', density=True);

plt.show()
rv1 = sp.multivariate_normal(mean=np.zeros(d), cov= np.ones(d))
for j in range(49):
   idx = np.random.choice(np.arange(10000), size = 10000)
   z = np.zeros((N, 10000, 2))
   for i in range(N):
      L[i] = myMixture.getSample(10000)
   zr = list(map(lambda i : L[:, i, :], idx)) #N*2
   mu, logvar2 = decoder(zr) # Nxd , Nxd
   print(j)
   for i in range(N):
      #tirer un nouveau candidat selon le vae 
      z = np.array(myMixture.getSample(10000))
      r = np.random.choice(np.arange(10000))
      mu, logvar2 = decoder(z[r,:].reshape(1,-1))
      rv = sp.multivariate_normal(mean = mu.numpy()[0] , cov = np.exp(logvar2.numpy()[0]/2))

      candidate_i = rv.rvs()
      phi_candidate = four_branch(candidate_i.reshape(1,-1))
      #print(phi_candidate)
      
      ratio = rv1.pdf(candidate_i) * rv.pdf(L[j,i,:]) / (rv1.pdf(L[j,i,:]) * rv.pdf(candidate_i)) * (phi_candidate > quantile)

      if np.random.uniform() < ratio:
         L[j+1, i, : ] = candidate_i
         phi_threshold[i] = phi_candidate[0]
    
      else :
         L[j+1, i, : ] = L[j,i,:]

sample = L[j+1]

plt.figure(figsize = (12,7))
for i in range(0,d):
  plt.subplot(5,5,i+1)
  plt.hist(sample[:,i], bins = 'auto', density=True);
plt.show()