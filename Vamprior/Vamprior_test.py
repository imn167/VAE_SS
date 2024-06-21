import sys 
sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS')
import matplotlib.pyplot as plt 
from function.VAE_GoM import *

dist1 = ot.TruncatedDistribution(ot.Normal(1), 2., ot.TruncatedDistribution.LOWER)
dist2 = ot.TruncatedDistribution(ot.Normal(1), -2., ot.TruncatedDistribution.UPPER)
dist = ot.Mixture([dist1,dist2])

plt.hist(np.array(dist.getSample(10000)), density=True, bins = 70)
plt.show()

two_mode = np.load('../two_component_truncated.npy')
N, d = two_mode.shape

# from IPython.display import clear_output #
# #Simulation of a truncated normal distribution 
# def truncatedDistribution(n_samples,  mean, variance,  bound):
#     sd = np.sqrt(variance)
#     L = list()
#     i = 0 
#     while i < n_samples:
#         prop = np.random.normal(loc = mean, scale=sd)
#         #prop_norm = np.linalg.norm(prop[0:2], ord=-np.inf)

#         if abs(prop[0]) > bound : 
#             L.append(prop)
#             i += 1
#             if i %10 == 0:
#                 clear_output(wait=True)
#                 print("boucle %d terminée" %(i))

#     return np.array(L)


# two_mode = truncatedDistribution(10000, np.zeros(d), np.ones(d), 2)

for i in range(d):
    plt.subplot(4, int(d/4), i+1)
    plt.hist(two_mode[:, i], density=True, bins= 100)
plt.show()


(tfp.distributions.MultivariateNormalDiag(loc = 2, scale_diag = np.ones(1)))

prior = VP(d, 16)
encoder = Encoder(d, 2, True)
decoder = Decoder(d, 2, True)

#init prior 

prior.initialized_ps(two_mode, .001)

plt.plot(prior.history.history['loss'])
plt.show()

vae = VAE(encoder, decoder, prior, name_prior = 'vamprior')

ae = AutoEncoder(encoder, decoder)
ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3))
ae.fit(two_mode,epochs=110, batch_size=150, shuffle=True, verbose = 0)

z_mean, _, _ = encoder(two_mode)

plt.scatter(z_mean[:, 0], z_mean[:, 1], s =6)
plt.show()



vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
vae.fit(two_mode, epochs = 150, batch_size = 100, shuffle = True) 


pseudo_inputs = prior(tf.eye(prior.K))
ps_mean, ps_logvar, _ = encoder(pseudo_inputs)
aggregated_posterior = prior.mixture(ps_mean, ps_logvar)
z = aggregated_posterior.sample(10000)
mean_x, log_var_x = decoder(z) #we get each mean and log variance of the several distribution then we sample from it

sample = np.random.normal(loc=mean_x, scale= tf.exp(log_var_x/2))


xx= np.linspace(-5,5, 1000)

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

plt.show()