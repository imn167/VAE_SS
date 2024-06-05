z = z_mean
z = z.numpy()
N, d = z.shape
means = (prior.means).numpy() 
sigma2 = np.exp(prior.logvars.numpy())
w = tf.nn.softmax(prior.w).numpy().reshape(-1)
K = w.shape[0]  
means

tau = expectation(z, np.ones((K,2)), np.ones((K,2)), N, K, w)
w_t, mu_t, sigma2_t = maximization(tau, z, K, N,d)
(tau.T).reshape(K,1,-1)
(((z.T).reshape(1,d,N)-means.reshape(K,d,1))**2 *(tau.T).reshape(K,1,-1) ).sum(axis=2)/(tau.sum(axis=0)).reshape(-1,1)
sigma2_t


w_t, mu_t, sigma2_t = EM(z_mean, prior, 10, 1e-3)
print(w_t, '\n', mu_t, '\n', sigma2_t)

X, Y = np.meshgrid(np.linspace(-15,5, 1000), np.linspace(-2,16, 1000))
pos = np.dstack((X,Y))
from scipy.stats import multivariate_normal
rv1  = multivariate_normal(mu_t[0], np.diag(tf.sqrt(sigma2_t[0]))) 
plt.scatter(z_mean[:,0], z_mean[:,1], s=8)
plt.contour(X,Y, rv1.pdf(pos), colors = 'red', alpha = .3)


w_t = tau.mean(axis =0 )
mu_t = tau.reshape(-1,1,K) * z.reshape(-1,d,1) #N x d x K
mu_t = (mu_t.sum(axis=0)).T #Kxd

sigma2_t = (z.reshape(1,d,-1) - mu_t.reshape(K,d,1))**2 * tau.reshape(K,1,-1) #K x d x N
sigma2_t = sigma2_t.sum(axis = 2) /( tau.sum(axis = 0)).reshape(-1,1)
#tau = expectation(z, mu_t, sigma2_t,N, K, w_t)
sigma2_t