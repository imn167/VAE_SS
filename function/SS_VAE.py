import sys

sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/function')

from VAE_GoM import *


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
            ae.fit(L[0] ,epochs=110, batch_size=150, shuffle=True, verbose = 0)

            z_mean, _, _ = encoder(L[0])
            

            for j in range(int(1/level-1)):
                candidate = np.zeros((dim, N)) # dim x N
                for i in range(dim): #independance allow to treat dimension by dimension 
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
    
