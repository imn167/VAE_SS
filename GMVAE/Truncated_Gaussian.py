import sys 
import time

sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS')

from function.EM import *
from function.VAE_GoM import *


#importing data 
two_mode = np.load('../two_component_truncated.npy')
#print(two_mode)
d = two_mode.shape[1]

####### Establishing a Latent spae for the data ###########

encoder = Encoder(d,2, True)
decoder = Decoder(d,2, True)

ae = AutoEncoder(encoder, decoder)
ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3))
ae.fit(two_mode,epochs=110, batch_size=150, shuffle=True, verbose = 0)

z_mean, _, _ = encoder(two_mode)

plt.figure()
plt.plot(ae.history.history['loss'])
plt.title('Loss trajectory for AutoEncoder')
plt.show()

plt.figure(figsize=(12,6))
for i in range(1):
    plt.subplot(1,2, i+1)
    plt.scatter(z_mean[:, i], z_mean[:,i+1], s = 5)
plt.show()

beggening = time.time()
prior = MoGPrior(2,4)
w_t, mu_t, sigma2_t, storage = EM(z_mean, prior, 1000, 1e-3)
print(w_t)
print(time.time() - beggening)



mixture_plot(z_mean.numpy(), w_t, mu_t, sigma2_t, min(z_mean.numpy()[:,0]), max(z_mean.numpy()[:,0]), min(z_mean.numpy()[:,1]), max(z_mean.numpy()[:,1]) )

prior.means.assign(tf.constant(mu_t, dtype=tf.float32))
prior.logvars.assign(tf.math.log(tf.constant(sigma2_t, dtype=tf.float32)))
prior.w.assign(tf.constant(w_t.reshape(1,-1), dtype=tf.float32))


vae = VAE(encoder, decoder, prior)
vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
vae.fit(two_mode, epochs = 150, batch_size = 100, shuffle = True) 


z_mean,_, _ = encoder(two_mode)

ColDist = [ot.Normal(np.array(mu), np.exp(0.5*np.array(sigma))) for mu, sigma in zip(prior.means, prior.logvars)]
weight = np.array(tf.nn.softmax(prior.w, axis =1)).reshape(-1)
myMixture = ot.Mixture(ColDist, weight)
z =myMixture.getSample(10000)
z= np.array(z)
mean_x, log_var_x = decoder(z) #we get each mean and log variance of the several distribution then we sample from it

sample = np.random.normal(loc=mean_x, scale= tf.exp(log_var_x/2))

mixture_plot(z_mean.numpy(), w_t, mu_t, sigma2_t, min(z_mean.numpy()[:,0]), max(z_mean.numpy()[:,0]), min(z_mean.numpy()[:,1]), max(z_mean.numpy()[:,1]) )


xx= np.linspace(-5,5, 1000)
dist1 = ot.TruncatedDistribution(ot.Normal(1), 2., ot.TruncatedDistribution.LOWER)
dist2 = ot.TruncatedDistribution(ot.Normal(1), -2., ot.TruncatedDistribution.UPPER)
dist = ot.Mixture([dist1,dist2])

plt.figure(figsize = (12,7))
for i in range(2,d):
  plt.subplot(5,5,i+1)
  plt.hist(sample[:,i], bins = 'auto', density=True);
  plt.plot(xx, sp.norm.pdf(xx))

plt.subplot(5,5,1)
plt.hist(sample[:,0], bins = 'auto', density=True);
plt.plot(xx, dist.computePDF(xx.reshape(-1,1)))

plt.subplot(5,5,2)
plt.hist(sample[:,1], bins = 'auto', density=True);
plt.plot(xx, dist.computePDF(xx.reshape(-1,1)))

plt.show()