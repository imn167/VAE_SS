
import numpy as np
import matplotlib.pyplot as plt 

def four_branch(X, beta =0):
   
    d, N = X.shape 
    quant1 = beta + np.expand_dims(np.sum(X, axis=0) / np.sqrt(d), axis=1)
    quant2 = beta - np.expand_dims(np.sum(X, axis=0) / np.sqrt(d), axis= 1 )
    quant3 = beta + np.expand_dims((np.sum(X[: int(d/2), :], axis= 0) - np.sum(X[int(d/2)+1 :, :], axis= 0)) / np.sqrt(d), axis=1)
    quant4 = beta - np.expand_dims((np.sum(X[: int(d/2), :], axis= 0) - np.sum(X[int(d/2)+1 :, :], axis= 0)) / np.sqrt(d), axis=1)

    tensor = np.concatenate([quant1, quant2, quant3, quant4], axis=1)
    minimum = np.min(tensor, axis = 1)
        

    return - minimum

d = 10
N = 10000
samples = np.random.normal(size = (N,d))
PHI = four_branch(samples.T)

quantile = np.quantile(PHI, 0.5)

idx = np.where(PHI > quantile)[0]

sample_threshold = samples[idx, :]
phi_threshold = PHI[ idx]

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
w_t, mu_t, sigma2_t, j = EM(z_mean, prior, 1000, 1e-2)
print(w_t, j)
print(time.time() - start)



mixture_plot(z_mean.numpy(), w_t, mu_t, sigma2_t, min(z_mean.numpy()[:,0]), max(z_mean.numpy()[:,0]), min(z_mean.numpy()[:,1]), max(z_mean.numpy()[:,1]) )


prior.means.assign(tf.constant(mu_t, dtype=tf.float32))
prior.logvars.assign(tf.math.log(tf.constant(sigma2_t, dtype=tf.float32)))
prior.w.assign(tf.constant(w_t.reshape(1,-1), dtype=tf.float32))


vae = VAE(encoder, decoder, prior)
vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))
vae.fit(sample_threshold, epochs = 150, batch_size = 32, shuffle = True) 

plt.plot(vae.history.history['kl_loss'])
plt.show()
plt.plot(vae.history.history['reconstruction_loss'])
plt.show()

z_mean,_, _ = encoder(sample_threshold)

ColDist = [ot.Normal(np.array(mu), np.exp(0.5*np.array(sigma))) for mu, sigma in zip(prior.means, prior.logvars)]
weight = np.array(tf.nn.softmax(prior.w, axis =1)).reshape(-1)
myMixture = ot.Mixture(ColDist, weight)
z =myMixture.getSample(10000)
z= np.array(z)
mean_x, log_var_x = decoder(z) #we get each mean and log variance of the several distribution then we sample from it

sample = np.random.normal(loc=mean_x, scale= tf.exp(log_var_x/2))

plt.figure(figsize = (12,7))
for i in range(0,d):
  plt.subplot(5,5,i+1)
  plt.hist(sample[:,i], bins = 'auto', density=True);

plt.show()

z =myMixture.getSample((10000, N))