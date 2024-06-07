import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 
import sys 

sys.path.append("/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/function")
#from VAE import *


############# SUBSET SIMULATION ALGORITHME ########################

def subset_simulation(sample, threshold,phi, sd,  level = .1):
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
            candidate = L[j] + np.random.normal(scale= sd, size=(2,N)) 
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



class MMA():
    def __init__(self, proposal, **kwargs) :
        super(MMA, self).__init__(**kwargs)

        self.quantile = list()
        self.sequence = list()
        self.accep_rate = list()
        self.proposal = proposal

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
    


class CMC():
    def __init__(self,N, **kwargs):
        super(CMC, self).__init__(**kwargs)
        self.N = N

    def __call__(self, phi, samples, threshold):
        L = list()
        for i in np.arange(0, self.N):
            L.append(np.mean(phi(samples[:,:i ]) > threshold))
        return np.array(L)
    
    def plot_trajectory(self, target, x):
        plt.plot(x, label = r'$\hat{P_f}$')
        plt.hlines(target, 0, self.N, colors= 'red', linestyles= '--', label= r'$P_f')
        plt.legend()
        plt.show()