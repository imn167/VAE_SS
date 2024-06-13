####################### Creation of the EM algorithme ######################
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import scipy.stats as sp
import openturns as ot 

from collections import deque #function for storage 

def expectation(z, mu_t, sigma2_t, N, K, w_t):
    gaussian_dist = [sp.multivariate_normal(mu, np.sqrt(sigma)) for mu, sigma in zip(mu_t, sigma2_t)]
    tau_membership = np.zeros((N, K))
    for i in range(K):
        tau_membership[:,i] = np.array(gaussian_dist[i].pdf(z)).reshape(-1,) * w_t[i]
    tau_membership = tau_membership / tau_membership.sum(axis=1).reshape(-1,1) 

    return tau_membership

def maximization(tau_membership, z, K, N, d):
    w_t = tau_membership.mean(axis =0 )
    mu_t =  z.reshape(-1,d,1) * tau_membership.reshape(N,1,K)  #N x d x K
    mu_t = (mu_t.sum(axis=0)).T / tau_membership.sum(axis=0).reshape(-1,1)#Kxd

    sigma2_t = ((z.T).reshape(1,d,N) - mu_t.reshape(K,d,1))**2 * (tau_membership.T).reshape(K,1,-1) #K x d x N
    sigma2_t = sigma2_t.sum(axis = 2) /( tau_membership.sum(axis = 0).reshape(-1,1))

    return w_t, mu_t, sigma2_t




def EM(z, prior, maxiter, epsilon):
    if type(z) is not np.ndarray:
        z = z.numpy()
    N, d = z.shape
    
    traj_means = list()
    traj_sigma2 = list()
    traj_w = list()
    means = (prior.means).numpy() 
    sigma2 = np.exp(prior.logvars.numpy())
    w = tf.nn.softmax(prior.w).numpy().reshape(-1)
    K = w.shape[0]    

    traj_means.append(means)
    traj_sigma2.append(sigma2)
    traj_w.append(w)

    storage = deque()
    ####### initialization #######
    #Expectation 
    tau_membership = expectation(z, means, sigma2, N, K, w)
    
    #Maximization 
    w_t, mu_t, sigma2_t = maximization(tau_membership, z, K, N, d)
    #storage.append([w_t, mu_t, sigma2_t])
    converged = False 
    j=1
    while j< maxiter  and ( not converged):
        #expectation 
        tau_membership = expectation(z, mu_t, sigma2_t, N, K, w_t)
        #maximization
        w_t, mu_t, sigma2_t = maximization(tau_membership, z, K, N, d)
        #storage.append([w_t, mu_t, sigma2_t])
        traj_means.append(means)
        traj_sigma2.append(sigma2)
        traj_w.append(w)
        j+=1
        diff_means = np.min(np.linalg.norm(traj_means[-1]-traj_means[-2], axis = 1))
        diff_sigma2 = np.min(np.linalg.norm(traj_sigma2[-1]- traj_sigma2[-2], axis = 1))
        diff_w = np.min(traj_w[-1]-traj_w[-2])
        converged = (diff_means > epsilon) or (diff_sigma2 > epsilon) or (diff_w > epsilon)
    
    return w_t, mu_t, sigma2, j


##plot 

def mixture_plot(z,w, mu, sigma2, x1,x2, y1, y2):
    K = w.shape[0]
    X, Y = np.meshgrid(np.linspace(x1,x2, 1000), np.linspace(y1,y2, 1000))
    pos = np.dstack((X,Y))
    rv =  [sp.multivariate_normal(mu, np.diag(np.sqrt(sigma))) for mu, sigma in zip(mu, sigma2)]
    plt.scatter(z[:,0], z[:,1], s=8)
    for i in range(K):
        plt.contour(X, Y,np.array( rv[i].pdf(pos)))
    
    plt.savefig('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/figures_ss/Trunacted_Gaussian_EM_mixture.png')
    plt.show()






