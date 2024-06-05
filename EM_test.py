##### testing 
import sys
sys.path.append('function')
from function.VAE_GoM import *
from function.EM import *

prior = MoGPrior(2,4)
from sklearn.datasets import make_blobs
Z, y, centers = make_blobs(1000, centers = 4, return_centers=True)
print(y.shape)
plt.scatter(Z[:,0], Z[:,1], s=8, c= y.reshape(-1,1))
plt.show()
w_t, mu_t, sigma2_t, storage = EM(Z, prior, 500, 1e-3)

print(w_t, '\n', mu_t, '\n', sigma2_t)
print(centers, '\n', )

print(storage)

mixture_plot(Z, w_t, mu_t, sigma2_t, min(Z[:,0]), max(Z[:,0]), min(Z[:,1]), max(Z[:,1]))


