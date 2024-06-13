import sys
import matplotlib.pyplot as plt 
sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/function')

from VAE_GoM import *

from EM import *


class SS_VAE(): #Modifier Metropolis Algorithm 
    #-------------------------------- This Algo works only in the cas of independant variables ------------------------#
    def __init__(self, latent_dim, input_dim,unvariate, **kwargs) :
        super(SS_VAE, self).__init__(**kwargs)

        self.quantile = list()
        self.sequence = list()
        self.accep_rate = list()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.unvariate = unvariate

    def call(self, sample, threshold, phi,  level = .1):
        N, dim = np.shape(sample)
        PHI = phi(sample)
        self.quantile.append(np.quantile(PHI, 1-level)) #look for another way to do quantiles 

        k = 0

        encoder = Encoder(self.input_dim, self.latent_dim, True)
        decoder = Decoder(self.input_dim, self.latent_dim, True)
        while self.quantile[k] < threshold:
            idx = np.where(PHI > self.quantile[k])[0] #index that statisfie.s the condition 
        # RAISING AN ERROR "vae approximation could be bad"
            try: 
                idx[0]
            except IndexError:
                print("Exception raised")
                break

            #SELECTING SAMPLES
            sample_threshold = sample[idx, :] # d x(N*level) samples that lie in Fk 

            #tensor to parallelize the process 
            L = np.zeros((int(1/level), N, dim)) 
            accep_sequance = np.zeros(N)
            random_seed = np.random.choice(idx, size = N) #BOOTSTRAP
            L[0] = sample[random_seed, :]
            phi_threshold = PHI[random_seed] # N
        

            ######### Trainig of the vae ##############

            #autoencoder training 
            ae = AutoEncoder(encoder, decoder)
            ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3))
            ae.fit(sample_threshold ,epochs=100, batch_size=32, shuffle=True, verbose = 0)

            z_mean, _, _ = encoder(sample_threshold)
            #visualisation 
            plt.figure()
            plt.plot(ae.history.history['loss'])
            plt.show()
            
            plt.figure(figsize=(12,6))
            for i in range(1):
                plt.subplot(1,2, i+1)
                plt.scatter(z_mean[:, i], z_mean[:,i+1], s = 5)
            plt.show()

            start = time.time()
            prior = MoGPrior(2,4)
            w_t, mu_t, sigma2_t, j = EM(z_mean, prior, 100, 1e-2)

            mixture_plot(z_mean.numpy(), w_t, mu_t, sigma2_t, min(z_mean.numpy()[:,0]), max(z_mean.numpy()[:,0]), min(z_mean.numpy()[:,1]), max(z_mean.numpy()[:,1]) )

            prior.means.assign(tf.constant(mu_t, dtype=tf.float32))
            prior.logvars.assign(tf.math.log(tf.constant(sigma2_t, dtype=tf.float32)))
            prior.w.assign(tf.constant(w_t.reshape(1,-1), dtype=tf.float32))

            vae = VAE(encoder, decoder, prior)
            vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))
            vae.fit(sample_threshold, epochs = 150, batch_size = 128, shuffle = True) 


            ColDist = [ot.Normal(np.array(mu), np.exp(0.5*np.array(sigma))) for mu, sigma in zip(prior.means, prior.logvars)]
            weight = np.array(tf.nn.softmax(prior.w, axis =1)).reshape(-1)
            myMixture = ot.Mixture(ColDist, weight)

            for j in range(int(1/level-1)):
                for i in range(N): #independance allow to treat dimension by dimension 

                    z = np.array( myMixture.getSample(10000))
                    r = np.random.choice(np.arange(10000))
                    mu, logvar2 = decoder(z[r, : ].reshape(1,-1))
                    rv = sp.multivariate_normal(mean = mu.numpy()[0] , cov = np.exp(logvar2.numpy()[0]/2))
                    candidate_i = rv.rvs(size = 1)

                    phi_candidate = phi(candidate_i.reshape(1,-1))
                    u = np.random.uniform() 
                    ratio = (self.unvariate.pdf(candidate_i) * rv.pdf(L[j,i, :])) / (self.unvariate.pdf(L[j,i,:]) * rv.pdf(candidate_i)) *  (phi_candidate > self.quantile[k])

                    L[j+1, i, :] = L[j, i, :] * (u >= ratio) + candidate_i * (u< ratio)
                    if (u< ratio):
                        phi_threshold[ i ] = phi_candidate
                    accep_sequance += 1 * (u<ratio)
                print(j)
            sample = L[j+1]
            print(sample)
            PHI = phi_threshold

            self.sequence.append(sample)
            self.accep_rate.append(accep_sequance / int(1/level-1))
            self.quantile.append(np.quantile(PHI, 1-level )) #searching the next intermediate 
            k += 1 
            print(self.quantile[k])
        failure = (level)**(k) * np.mean(PHI > threshold)

        return self.sequence, k, failure, self.quantile, self.accep_rate
    
