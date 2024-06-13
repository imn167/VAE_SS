from function.SS_VAE import *

#Problem test 

def four_branch(X):
   
    N, d = X.shape 
    quant1 =  np.expand_dims(np.sum(X, axis=1) / np.sqrt(d), axis=1)
    quant2 = - np.expand_dims(np.sum(X, axis=1) / np.sqrt(d), axis= 1 )
    quant3 = + np.expand_dims((np.sum(X[:, : int(d/2)], axis= 1) - np.sum(X[:, int(d/2) :], axis= 1)) / np.sqrt(d), axis=1)
    quant4 = - np.expand_dims((np.sum(X[:, : int(d/2)], axis= 1) - np.sum(X[:, int(d/2) :], axis= 1)) / np.sqrt(d), axis=1)

    tensor = np.concatenate([quant1, quant2, quant3, quant4], axis=1)
    minimum = np.min(tensor, axis = 1)
        

    return - minimum


N, d = 10000, 10
samples = np.random.normal(size=(N, d))

pt = 9.3e-4
t = 3.5

rv = sp.multivariate_normal(mean= np.zeros(d), cov= np.ones(d)) #distribution of the data 

ss_vae = SS_VAE(2,d, rv)

ss_vae.call(samples, t, four_branch, .5)