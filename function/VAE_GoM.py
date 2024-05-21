from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import scipy.stats as sp 


import time

##### Creation of the Sampling class ########

class MoGPrior(layers.Layer):
  def __init__(self, L, num_components, multiplier=1.0, **kwargs):
    super(MoGPrior, self).__init__(**kwargs)
    self.L = L
    self.num_components = num_components
    self.multiplier = multiplier

    # means and log variances (trainable parameters)
    self.means = tf.Variable(tf.random.normal([num_components, L]) * multiplier, dtype=tf.float32)
    self.logvars = tf.Variable(tf.random.normal([num_components, L]), dtype=tf.float32)

    # mixing weights (trainable parameter)
    self.w = tf.Variable(tf.ones([1,num_components]), dtype=tf.float32)

  def get_params(self):
    return self.means, self.logvars

  def call(self):
    # get means and log variances
    means, logvars = self.get_params()

    # normalize mixing weights using softmax
    w = tf.nn.softmax(self.w, axis=1)

    # sample component indices
    indexes = tf.random.categorical(tf.math.log(w), 1)

    # sample from chosen components
    #z = []
    eps = tf.random.normal((1, self.L))
    indx = indexes
    z = means[ indx[0,0]] + eps * tf.exp(0.5*logvars[indx[0,0]])
    #z.append(z_i)
    #z = tf.stack(z, axis=0)

    return z


class Sampling(layers.Layer):
   def __init__(self, latent_dim, num_components, **kwargs):
      super(Sampling, self).__init__(name='sampler')
      self.latent_dim = latent_dim
      self.n_components = num_components
   def call(self, inputs):
      z_mean, log_var_z = inputs
      batch = tf.shape(z_mean)[0]
      gaussian_mixture = MoGPrior(self.latent_dim, self.n_components).call()
      return z_mean + tf.exp(0.5* log_var_z) * gaussian_mixture
    
##### Creation of the Encoder Class #####
class Encoder(layers.Layer):
  def __init__(self, input_dim, latent_dim, n_components, **kwargs):
    super(Encoder, self).__init__(name='encoder', **kwargs)
    self.enc1 = layers.Dense(input_dim, activation='relu')
    self.enc2 = layers.Dense(64, activation = 'relu')
    self.enc3 = layers.Dense(32, activation = 'relu')
    self.mean_z = layers.Dense(latent_dim, name = 'z_mean')
    self.logvar_z = layers.Dense(latent_dim, name = 'z_log_var')
    self.sampling = Sampling(latent_dim, n_components)

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

 def __init__(self,input_dim, latent_dim, **kwargs):
    super(Decoder, self).__init__(name='decoder', **kwargs)
    self.dec1 = layers.Dense(latent_dim, activation='relu' )
    self.dec2 = layers.Dense(32, activation='relu')
    self.dec3 = layers.Dense(64, activation='relu')
    self.out = layers.Dense(input_dim, name = 'x_out')
    self.x_log_var = layers.Dense(input_dim, name = 'x_log_var')

 def call(self, z):
   z = self.dec1(z)
   z = self.dec2(z)
   z = self.dec3(z)
   return self.out(z), self.x_log_var(z)
 
##################### Creation of subclass od Model : VAE ########################
## We can see the subclass VAE as a Keras Model therefore it has the several method as fit and compile 
## We overwrite the train function of the model : train_step (customizing the training)
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(name = 'vae', **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        self.loss_tracker = tf.keras.metrics.Mean(name = 'loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction, x_log_var = self.decoder(z)
        return z_mean, z_log_var, reconstruction, x_log_var
    
    def get_encoder_decoder(self):
        return self.encoder, self.decoder
    
    def train_step(self, data):
       # data, y = data
        #y = tf.reshape(y,[-1])
        with tf.GradientTape() as tape :
            z_mean, z_log_var, reconstruction, x_log_var = self(data) 

            #we compute the first loss : the log-likelyhood 
            scale_x = tf.exp(x_log_var) #variance 
            log_pdf = 0.5 * tf.reduce_sum(tf.pow(data-reconstruction, 2) / scale_x + x_log_var, axis = 1) #-log_pdf because we want to maximise it (SGD aim to minimize in keras)
            reconstruction_loss =  tf.reduce_mean(log_pdf) #tf.multiply(log_pdf, y)
            kl_loss =  -0.5 * (1 +  z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(  kl_loss, axis =1 )) 
            total_loss =( reconstruction_loss + kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}
        


