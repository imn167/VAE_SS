from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import scipy.stats as sp 
import openturns as ot 

import tensorflow_probability as tfp
tfd = tfp.distributions

import time

##### Creation of the MoG prior class ########

class MoGPrior(tf.keras.Model):
  def __init__(self, latent_dim, num_components, **kwargs):
    super(MoGPrior, self).__init__(**kwargs)
    self.latent_dim = latent_dim
    self.num_components = num_components

    # means and log variances (trainable parameters)
    self.means = tf.Variable(tf.random.normal(shape=(num_components, latent_dim)),
                            trainable=True, name='means')
    self.logvars = tf.Variable(tf.random.normal(shape=(num_components, latent_dim)),
                               trainable=True, name='logvars')

    # mixing weights (trainable parameter)
    self.w = tf.Variable(tf.random.normal(shape=(1,num_components)), trainable=True)

    self.loss_tracker = tf.keras.metrics.Mean(name = 'loss')

  def get_params(self):
    return self.means, self.logvars, self.w

  def sampling(self):
    # get means and log variances of the mixture
    means, logvars, w = self.get_params()

    # normalize mixing weights using softmax (see gumball)
    w = tf.nn.softmax(self.w, axis=1)
    # sample component indices
    #indexes = (tf.random.categorical(tf.math.log(w), batch_size))[0]

    # sample from chosen components
    #z = tf.map_fn(fn= lambda indx : means[indx] + tf.random.normal(shape= (1,2)) * tf.exp(0.5*logvars[indx]),
     #     elems= indexes,
      #    dtype=tf.float32)
    #z = tf.squeeze(z)
    ColDist = [ot.Normal(mu, sigma) for mu, sigma in zip(means.numpy(), np.exp(0.5*logvars.numpy()))]
    weight = np.array(tf.nn.softmax(w)).reshape(-1)
    myMixture = ot.Mixture(ColDist, weight)
  
    #z =myMixture.getSample(batch_size.numpy())

    return myMixture

  #compute the log_density of each gaussian at the point z
  def log_normal_diag(self, z, mean, logvar):
     nn_exp = -0.5*( tf.math.log(2.0*np.pi) + logvar)
     exp = -0.5* (z-mean)**2 * (tf.exp(-logvar))
     return tf.reduce_sum(nn_exp + exp, axis = -1)

  def log_prob(self, z):
     #getting means and vars of the gausiian mixture
     means, logvars, w = self.get_params()

     #normalising the weight with the softmax transformation
     w = tf.transpose(tf.nn.softmax(w, axis=1))
     #reshape for broadcast
     z =  tf.expand_dims(z, axis=0) #1 x batch x latent_dim
     means = tf.expand_dims(means, axis=1) #num_compo x 1 x latent_dim
     logvars = tf.expand_dims(logvars, axis=1) #num_compo x 1 x latent_dim

     #we compute the log of each gaussian for each z
     log_p = self.log_normal_diag(z, means, logvars) + tf.math.log(w) #num_compo x batch
     prob = tf.reduce_logsumexp(tf.squeeze(log_p), axis=0) #log(sum(exp())) #batch
     return prob

  def train_step(self, data):

    with tf.GradientTape() as tape :
      log_likelyhood = self.log_prob(data) #data is latent space sampling
      loss = tf.reduce_mean(-log_likelyhood)

    grads = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    self.loss_tracker.update_state(loss)
    return {m.name: m.result() for m in self.metrics}


############################################################

#### Creation of the Vamprior class #####

class VP(tf.keras.Model):
  def __init__(self, input_dim, K, **kwargs):
    super(VP, self).__init__(name = 'VP', **kwargs)

    self.K = K #number of pseudo-inputs
    self.input_dim = input_dim
    self.canonical = layers.Dense(K, activation = 'linear', name = 'canonical')
    self.hidden = layers.Dense(128, activation = 'linear')
    self.pseudo_inputs = layers.Dense(input_dim, activation = 'linear', name = 'pseudo_inputs' )

  def call(self, inputs):
    u = self.canonical(inputs) #K
    u = self.hidden(u)
    pseudo = self.pseudo_inputs(u)
    return pseudo 

  def initialized_ps(self, X, lr):
    N, d = X.shape 
    #
    id_matrix = tf.eye(self.K)
    idx = np.random.choice(np.arange(N), size= self.K, replace= False)
    target_x = tf.convert_to_tensor(X[idx, :]) 
    #fitting the model 
    self.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= lr), loss = 'mse')
    self.fit(id_matrix, target_x, epochs = 1000, batch_size = self.K, verbose = 0)

    return self 
  @tf.function
  def mixture(self, ps_mean, ps_logvar):
    components = []
    cat = tfd.Categorical(probs = self.K*[1/self.K])
    for k in range(self.K):
      mu = ps_mean[k]
      sigma = tf.exp(0.5* ps_logvar[k])
      d_k = tfp.distributions.MultivariateNormalDiag(loc = mu, scale_diag = sigma)
      components.append(d_k)
    #Coldist = np.array([tfp.distributions.MultivariateMultivariateNormalDiag(loc = mu, scale_diag = sigma) \
                       # for mu, sigma in zip(ps_mean, tf.exp(0.5* ps_logvar))]) #different variationnal distribution according to pseudo inputs 
    aggre_posterior = tfd.Mixture(cat = cat, components = components)
    return aggre_posterior





###################################################
  
class Sampling(layers.Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0] 
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape= (batch, dim))
    return z_mean + tf.exp(.5 * z_log_var) * epsilon
    
##### Creation of the Encoder Class #####
class Encoder(layers.Layer):
  def __init__(self, input_dim, latent_dim, training, **kwargs):
    super(Encoder, self).__init__(name='encoder', **kwargs)
    self.act = layers.LeakyReLU(alpha = .3)
    self.enc1 = layers.Dense(input_dim, activation='relu')
    self.enc2 = layers.Dense(64, activation = 'relu')
    self.enc3 = layers.Dense(32, activation = 'relu')
    self.mean_z = layers.Dense(latent_dim, name = 'z_mean')
    self.logvar_z = layers.Dense(latent_dim, name = 'z_log_var', trainable = training)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.enc1(inputs)
    x = self.enc2(x)
    x = self.enc3(x)
    z_mean = self.mean_z(x)
    z_log_var = self.logvar_z(x)
    z = self.sampling([z_mean, z_log_var])
    return z_mean, z_log_var, z
  
     

############## Creation of the Decoder Class #################
class Decoder(layers.Layer):

 def __init__(self,input_dim, latent_dim,training, **kwargs):
    super(Decoder, self).__init__(name='decoder', **kwargs)
    self.latent_dim = latent_dim
    self.dec1 = layers.Dense(latent_dim, activation='relu' )
    self.dec2 = layers.Dense(32, activation='relu')
    self.dec3 = layers.Dense(64, activation='relu')
    self.out = layers.Dense(input_dim, name = 'x_out')
    self.x_log_var = layers.Dense(input_dim, name = 'x_log_var', trainable = training)

 def call(self, z):
   z = self.dec1(z)
   z = self.dec2(z)
   z = self.dec3(z)
   return self.out(z), self.x_log_var(z)
 
 ############## Class Auto-Encoder ################
class AutoEncoder(tf.keras.Model):
  def __init__(self, encoder,decoder, **kwargs):
    super(AutoEncoder, self).__init__(name='auto-encoder', **kwargs)
    self.encoder = encoder
    self.decoder = decoder

    self.loss_tracker = tf.keras.metrics.Mean(name = 'loss')

  def call(self, inputs):
    z_mean, _, _ = self.encoder(inputs)
    x_mean, _ = self.decoder(z_mean)
    return z_mean, x_mean 

  def train_step(self, data):


    with tf.GradientTape() as tape :

          _, x_mean= self(data)
            #we compute the first loss : the log-likelyhood
          total_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(data-x_mean, 2), axis =1))
            #scale_x = tf.exp(x_log_var) #variance
            #log_pdf = 0.5 * tf.reduce_sum(tf.pow(data-reconstruction, 2) / scale_x + x_log_var, axis = 1) #-log_pdf because we want to maximise it (SGD aim to minimize in keras)
            #total_loss =  tf.reduce_mean(log_pdf) #tf.multiply(log_pdf, y)

    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    self.loss_tracker.update_state(total_loss)
    return {m.name: m.result() for m in self.metrics}
    
##################### Creation of subclass od Model : VAE ########################
## We can see the subclass VAE as a Keras Model therefore it has the several method as fit and compile 
## We overwrite the train function of the model : train_step (customizing the training)
class VAE(tf.keras.Model):
  def __init__(self, encoder, decoder, prior, **kwargs):
      super(VAE, self).__init__(name = 'vae')
      self.encoder = encoder
      self.decoder = decoder
      self.prior = prior
      self.kwargs = kwargs
        
      self.loss_tracker = tf.keras.metrics.Mean(name = 'loss')
      self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')
      self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
    
  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstruction, x_log_var = self.decoder(z)
    return z_mean, z_log_var, reconstruction, x_log_var, z
    
  def get_encoder_decoder(self):
    return self.encoder, self.decoder

  def density_x(self, n_samples):
    z = self.prior.sampling().getSample(n_samples)
    x_mean, x_log_var= self.decoder(np.array(z)) #n_samples distribution 
    Dist = [ot.Normal(np.array(mu), np.exp(0.5*np.array(sigma))) for mu, sigma in zip(x_mean, x_log_var)]
    Distr_x = ot.Mixture(Dist)
    return Distr_x
    
  def train_step(self, data):
       # data, y = data
        #y = tf.reshape(y,[-1])
    with tf.GradientTape() as tape :

      z_mean, z_log_var, reconstruction, x_log_var, z = self(data) 
      #we compute the first loss : the log-likelyhood 
      scale_x = tf.exp(x_log_var) #variance 
      log_pdf = 0.5 * tf.reduce_sum(tf.pow(data-reconstruction, 2) / scale_x + x_log_var, axis = 1) #-log_pdf because we want to maximise it (SGD aim to minimize in keras)
      reconstruction_loss =  tf.reduce_mean(log_pdf) #tf.multiply(log_pdf, y)
      # z_var_inv = 1/tf.exp(z_log_var)            
      # n_01 = tf.sqrt(z_var_inv)*(z-z_mean)
      # N_01_latent = tfd.MultivariateNormalDiag(loc=self.decoder.latent_dim*[0],scale_diag=self.decoder.latent_dim*[1])
      entropy =  tf.reduce_sum(-0.5 * (tf.math.log(2.0*np.pi) + 1 +  z_log_var), axis=1 )
      if self.kwargs['name_prior'] == 'vamprior':
        #pseudo_inputs 
        id_matrix = tf.eye(self.prior.K) 
        pseudo_inputs = self.prior(id_matrix)
        ps_mean, ps_logvar, _ = self.encoder(pseudo_inputs)
        aggregated_posterior = self.prior.mixture(ps_mean, ps_logvar)
        cross_entropy = aggregated_posterior.log_prob(z)
      else :
        cross_entropy= self.prior.log_prob(z)

      kl_loss = tf.reduce_mean(entropy -cross_entropy) 
      total_loss = reconstruction_loss + kl_loss

    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
    self.loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    

    return {m.name: m.result() for m in self.metrics}

  



