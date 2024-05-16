import numpy as np 
from scipy import stats
from function.VAE import *


############# SUBSET SIMULATION ALGORITHME ########################

def subset_simulation(sample, threshold,phi, latent_dim, sd,kernel_param,  level = .1, vae_ss = True):
    dim, N = np.shape(sample)
    #list of the intermediate threshold 
    quantile = list()
    sequence = list()
    accep_rate = list()
    #first threshold 
    PHI = phi(sample)
    quantile.append(np.quantile(PHI, 1-level))
    k = 0
    rv = stats.multivariate_normal(mean=np.zeros(dim)) #computy the probability density of a normal vector of dimension dim
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    vae = VAE(encoder, decoder)
    while quantile[k] < threshold:
        idx = np.where(PHI > quantile[k])[0] #index that statisfie.s the condition 
        # RAISING AN ERROR "vae approximation could be bad"
        try: 
            idx[0]
        except IndexError:
            print("Exception raised")
            break

        #BOOTSTRAP ON THE N*LEVEL SELECTED SAMPLES 
        random_seed = np.random.choice(a=idx, size =N)
        sample_threshold = sample[:, random_seed] #N*level samples that lie in Fk 
        phi_threshold = PHI[random_seed]
        ## simulation according to mcmc 
        chain = np.zeros((dim, N)) # mcmc chain 
    
        #tensor to parallelize the process 
        L = np.zeros((int(1/level), dim, N)) 
        accep_sequance = np.zeros(N)
        L[0] = sample_threshold
        phi_accepted = phi_threshold
        for j in range(int(1/level-1)):
            if vae_ss :  #generating samples according to the vae reconstruction density 
                vae_sample = (sample[:, idx]).T
                vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3))
                vae.fit(vae_sample, epochs = 10, batch_size=32)
                _,_, z = encoder(sample_threshold.T)
                mean_x, log_var_x = decoder(z)
                candidate =( np.random.normal(loc= mean_x, scale= tf.exp(log_var_x /2))).T
                #print(candidate.shape)
            else :
                candidate = (L[j] + 
                             kernel_param * np.random.normal(scale= sd, size=(dim,N))) / (np.sqrt(1+kernel_param**2)) #simulation of N gaussien points candidates for each value of the Bootstraped chain
            #stock for phi(candidate) in order to calculat it only once 
            phi_candidate = phi(candidate)
            ratio =  rv.pdf(candidate.T) / rv.pdf(L[j].T) *(phi_candidate> quantile[k]) # Transposé pour candidate si no vae 
            #MAJ
            u = np.random.uniform(size= N)
            L[j+1] =  L[j] * (u >= ratio) + candidate * (u< ratio)
            phi_accepted[(u< ratio)] = phi_candidate[(u< ratio)]
            accep_sequance += 1 * (u<ratio)
        #phi_value management 
        phi_threshold = phi_accepted
        PHI = phi_threshold #new phi_values 
        
        #we only keep the last mcmc simulation
        chain = L[j+1]   
        ##### chain for the step k completed 
        sequence.append(chain) 
        quantile.append(np.quantile(PHI, 1-level )) #searching the next intermediate 
        accep_rate.append(accep_sequance / int(1/level-1))
        sample = chain
        k += 1 
        
    failure = (level)**(k) * np.mean(PHI > threshold)
    return sequence, k, failure, quantile, accep_rate


