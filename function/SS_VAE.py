import sys
import matplotlib.pyplot as plt 
sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/function')

from VAE_GoM import *

from EM import *


class SS_VAE(): #Modifier Metropolis Algorithm 
    #-------------------------------- This Algo works only in the cas of independant variables ------------------------#
    def __init__(self, latent_dim, input_dim, **kwargs) :
        super(SS_VAE, self).__init__(**kwargs)

        self.quantile = list()
        self.sequence = list()
        self.accep_rate = list()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def call(self, sample, threshold, phi,  level = .1):
        dim, N = np.shape(sample)
        PHI = phi(sample)
        self.quantile.append(np.quantile(PHI, 1-level)) #look for another way to do quantiles 

        k = 0
        while self.quantile[k] < threshold:
            idx = np.where(PHI > self.quantile[k])[0] #index that statisfie.s the condition 
        # RAISING AN ERROR "vae approximation could be bad"
            try: 
                idx[0]
            except IndexError:
                print("Exception raised")
                break

            #SELECTING SAMPLES
            sample_threshold = sample[:, idx] # d x(N*level) samples that lie in Fk 
            phi_threshold = PHI[idx] # (N*level)

            #tensor to parallelize the process 
            L = np.zeros((int(1/level), dim, N)) 
            accep_sequance = np.zeros(N)
            L[0] = sample_threshold
            phi_accepted = phi_threshold

            ######### Trainig of the vae ##############
            encoder = Encoder(self.input_dim, self.latent_dim, True)
            decoder = Decoder(self.input_dim, self.latent_dim, True)


            #autoencoder training 
            ae = AutoEncoder(encoder, decoder)
            ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3))
            ae.fit(L[0].T ,epochs=100, batch_size=32, shuffle=True, verbose = 0)

            z_mean, _, _ = encoder(L[0])
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
            w_t, mu_t, sigma2_t, j = EM(z_mean, prior, 15, 1e-3)

            mixture_plot(z_mean.numpy(), w_t, mu_t, sigma2_t, min(z_mean.numpy()[:,0]), max(z_mean.numpy()[:,0]), min(z_mean.numpy()[:,1]), max(z_mean.numpy()[:,1]) )

            prior.means.assign(tf.constant(mu_t, dtype=tf.float32))
            prior.logvars.assign(tf.math.log(tf.constant(sigma2_t, dtype=tf.float32)))
            prior.w.assign(tf.constant(w_t.reshape(1,-1), dtype=tf.float32))

            vae = VAE(encoder, decoder, prior)
            vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))
            vae.fit(sample_threshold, epochs = 150, batch_size = 32, shuffle = True) 


            ColDist = [ot.Normal(np.array(mu), np.exp(0.5*np.array(sigma))) for mu, sigma in zip(prior.means, prior.logvars)]
            weight = np.array(tf.nn.softmax(prior.w, axis =1)).reshape(-1)
            myMixture = ot.Mixture(ColDist, weight)

            for j in range(int(1/level-1)):
                candidate = np.zeros((dim, N)) # dim x N
                for i in range(N): #independance allow to treat dimension by dimension 
                    z = myMixture.getSample(10000)
                    r = np.random.choice(np.arange(10000))
                    mu, logvar2 = decoder()
                    candidate_i = self.proposal.call(L[j, i, :]) #N
                    u = np.random.uniform(size = N) #N
                    r = self.proposal.density(candidate_i) / self.proposal.density(L[j,i,:]) #N
                    candidate_i = candidate_i * (u < r) + L[j+1, i, : ] * (u>= r) #N
                    candidate[i] = candidate_i

                phi_candidate = phi(candidate) #N
                phi_accepted[(phi_candidate > self.quantile[k])] = phi_candidate[(phi_candidate > self.quantile[k])]
                L[j+1] = candidate * (phi_candidate > self.quantile[k]) + L[j] * (phi_candidate <= self.quantile[k])
                accep_sequance += 1 * phi_candidate > self.quantile[k]
            
            sample = L[j+1]
            phi_threshold = phi_accepted
            PHI = phi_threshold

            self.sequence.append(sample)
            self.accep_rate.append(accep_sequance / int(1/level-1))
            self.quantile.append(np.quantile(PHI, 1-level )) #searching the next intermediate 
            k += 1 
        failure = (level)**(k) * np.mean(PHI > threshold)

        return self.sequence, k, failure, self.quantile, self.accep_rate
    
