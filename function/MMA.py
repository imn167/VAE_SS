import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as sp 


def MMA(sample, phi, threshold, rv, sd, level):
    N, d = sample.shape 

    quantile = list()
    sequence = list()
    acceptance_rate = list()

    # first fixed quantile 
    PHI = phi(sample)
    quantile.append(np.quantile(PHI, (1-level) )) 

    k = 0
    while quantile[k] <= threshold :
        idx = np.where(PHI > quantile[k])[0] #N x level 
        #Bootstrap so that we have a sample of N length following the empirical process
        random_seed = np.random.choice(idx, size=N)
        sample_threshold = sample[random_seed, :] #N x d in Fk
        phi_threshold = PHI[random_seed] #N

        #MCMC with the modified Metropolis 
        L = np.zeros((int(1/level), N, d))
        L[0] = sample_threshold
        for i in range(int(1/level)-1):
            candidate = np.zeros((N, d))
            chain_acceptance = np.zeros(N)
            for j in range(d):
                candidate_j = np.random.normal(loc = L[i, :, j],  scale = sd)
                ratio = rv.pdf(candidate_j) / rv.pdf(L[i, :, j]) #N 
                u = np.random.uniform(size = N)
                candidate[:, j] = candidate_j * (u < ratio) + L[i, :, j] * (u >= ratio)
            
            #now we test if the candidates belongs to Fk
            phi_candidate = phi(candidate)
            accepted_candidate = phi_candidate > quantile[k] #N
            chain_acceptance += 1 * (accepted_candidate)
            phi_threshold[accepted_candidate] = phi_candidate[accepted_candidate] #N : MAJ of phi values 
            L[i+1] =  candidate * accepted_candidate.reshape(-1,1)+  L[i] * np.invert(accepted_candidate).reshape(-1,1)
        
        sample = L[i+1]
        #samples for each conditionnal distribution
        sequence.append(sample)
        acceptance_rate.append(chain_acceptance / (int(1/level)))
        PHI = phi_threshold

        #next quantile 
        quantile.append(np.quantile(PHI, (1-level) ))
        k += 1 
    
    failure = (level)**(k) * np.mean(PHI > threshold)
    

    return quantile, sequence, acceptance_rate, failure, np.mean(PHI) 


def four_branch(X):
   
    N, d = X.shape 
    quant1 =  np.expand_dims(np.sum(X, axis=1) / np.sqrt(d), axis=1)
    quant2 = - np.expand_dims(np.sum(X, axis=1) / np.sqrt(d), axis= 1 )
    quant3 =  np.expand_dims((np.sum(X[:, : int(d/2)], axis= 1) - np.sum(X[:, int(d/2) :], axis= 1)) / np.sqrt(d), axis=1)
    quant4 =  np.expand_dims((-np.sum(X[:, : int(d/2)], axis= 1) + np.sum(X[:, int(d/2) :], axis= 1)) / np.sqrt(d), axis=1)

    tensor = np.concatenate([quant1, quant2, quant3, quant4], axis=1)
    minimum = np.min(tensor, axis = 1)
        

    return - minimum

rv = sp.multivariate_normal()
samples = np.random.normal(size = (10000, 10))
N, d = samples.shape


print('------------------------------------------')
print(f"Taille de l'echantillon {samples.shape}")
print(f"MC estimation {np.mean(four_branch(samples) > 3.5)}")
quantile, chain, acceptance, failure, mean = MMA(samples, four_branch, 3.5, rv, 1, .1)

print(f"Estimation SS de défaillance {failure} et en moyenne {mean} et quantile {quantile}")
acceptance = np.array(acceptance)
print(f"Taux d'acceptation en moyenne {acceptance.mean(axis=1)}")


if d ==2 : 
    plt.figure()
    plt.scatter(chain[-1][:, 0], chain[-1][:, 1], s= 6)
    plt.show()