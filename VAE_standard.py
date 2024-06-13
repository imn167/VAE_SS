import sys 

sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/function')

from function.VAE import *
import matplotlib.pyplot as plt 
from IPython.display import clear_output #
import openturns as ot 




N, dim = 10000, 10
print(f"Données de dimesion {dim} et de taille {N}")

#Simulation of a truncated normal distribution 
def truncatedDistribution(n_samples,  mean, variance,  bound):
    sd = np.sqrt(variance)
    L = list()
    i = 0 
    while i < n_samples:
        prop = np.random.normal(loc = mean, scale=sd)
        #prop_norm = np.linalg.norm(prop[0:2], ord=-np.inf)

        if prop[0] > bound : 
            L.append(prop)
            i += 1
            if i %10 == 0:
                clear_output(wait=True)
                print("boucle %d terminée" %(i))

    return np.array(L)

one_side_mode = truncatedDistribution(N, np.zeros(dim), np.ones(dim), 2 )

plt.figure(figsize=(10,5))
for i in range(dim):
    plt.subplot(4, int(dim/4) + 1, i+1)
    plt.hist(one_side_mode[:, i], density=True, bins='auto')
plt.show()

#ACP 
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

pca_data = pca.fit_transform(one_side_mode)

plt.scatter(pca_data[:, 0], pca_data[:, 2], s = 6)
plt.title('Plan ACP')
plt.show()

from sklearn.manifold import TSNE



encoder = Encoder(input_dim= dim, latent_dim=2)
decoder = Decoder(input_dim= dim, latent_dim=2)
vae = VAE(encoder, decoder)######## Training #########
vae.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001))
history = vae.fit(one_side_mode ,epochs=100, batch_size=100)

plt.plot(vae.history.history['kl_loss'])
plt.show()

plt.plot(vae.history.history['reconstruction_loss'])
plt.show()

z = np.random.normal(size=(N, 2))


z_mean, _, _ = encoder(one_side_mode)

plt.scatter(z_mean[:, 0], z_mean[:, 1], s=6)
plt.title('Espace latent')
plt.show()


mean_x, logvar_x = decoder(z)

sample = np.random.normal(loc= mean_x, scale= np.exp(logvar_x.numpy() / 2))

plt.figure(figsize=(10,5))
col = int(dim/2)
xx = np.linspace(-5,5, 1000)
distr1 = ot.TruncatedDistribution(ot.Normal(1), 2, ot.TruncatedDistribution.LOWER)
for i in range(1,dim):
    plt.subplot(2, col, i+1)
    plt.hist(sample[:,i], density=True, bins = 'auto');
    plt.plot(xx, sp.norm.pdf(xx))
plt.subplot(2, col, 1)
plt.hist(sample[:,0], density=True, bins = 'auto');
plt.plot(xx, distr1.computePDF(xx.reshape((-1,1))))
plt.show()

x_mean, _ = decoder(z_mean)
plt.figure(figsize=(10,5))
for i in range(dim):
    plt.subplot(2, col, i+1)
    plt.hist(x_mean[:,i], density=True, bins = 'auto');
plt.title('reconstruction')
plt.show()