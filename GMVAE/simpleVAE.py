from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import scipy.stats as sp
import openturns as ot
import matplotlib.pyplot as plt 


import time

##### Creation of the Sampling class ########

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

  def call(self, batch_size):
    # get means and log variances of the mixture
    means, logvars, w = self.get_params()

    # normalize mixing weights using softmax (see gumball)
    w = tf.nn.softmax(self.w, axis=1)
    # sample component indices
    indexes = (tf.random.categorical(tf.math.log(w), batch_size))[0]

    # sample from chosen components
    z = tf.map_fn(fn= lambda indx : means[indx] + tf.random.normal(shape= (1,2)) * tf.exp(0.5*logvars[indx]),
          elems= indexes,
          dtype=tf.float32)
    z = tf.squeeze(z)

    return z
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


#################################
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape= (batch, dim))
        return z_mean + tf.exp(.5 * z_log_var) * epsilon

##### Creation of the Encoder Class #####
class Encoder(layers.Layer):
  def __init__(self, input_dim, latent_dim, **kwargs):
    super(Encoder, self).__init__(name='encoder', **kwargs)
    self.enc1 = layers.Dense(input_dim, activation='relu')
    self.enc2 = layers.Dense(64, activation = 'relu')
    self.enc3 = layers.Dense(32, activation = 'relu')
    self.mean_z = layers.Dense(latent_dim, name = 'z_mean')
    self.logvar_z = layers.Dense(latent_dim, name = 'z_log_var')
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


#Creation of a model for an auto_encoder




##################### Creation of subclass od Model : VAE ########################
## We can see the subclass VAE as a Keras Model therefore it has the several method as fit and compile
## We overwrite the train function of the model : train_step (customizing the training)
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, prior, **kwargs):
        super(VAE, self).__init__(name = 'vae', **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        self.loss_tracker = tf.keras.metrics.Mean(name = 'loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction, x_log_var = self.decoder(z)
        #batch = tf.shape(z_mean)[0]
        #prior_sample = self.prior(batch)
        return z_mean, z_log_var, reconstruction, x_log_var, z

    def get_encoder_decoder(self):
        return self.encoder, self.decoder

    def train_step(self, data):
       # data, y = data
        #y = tf.reshape(y,[-1])
        with tf.GradientTape() as tape :

            z_mean, z_log_var, reconstruction, x_log_var, z = self(data)
            #we compute the first loss : the log-likelyhood
            scale_x = tf.exp(x_log_var) #variance
            log_pdf = 0.5 * tf.reduce_sum(tf.pow(data-reconstruction, 2) / scale_x + x_log_var, axis = 1) #-log_pdf because we want to maximise it (SGD aim to minimize in keras)
            reconstruction_loss =  tf.reduce_mean(log_pdf) #tf.multiply(log_pdf, y)
            entropy =  tf.reduce_sum(-0.5 * (tf.math.log(2.0*np.pi) + 1 +  z_log_var), axis=1 )
            cross_entropy= self.prior.log_prob(z)
            kl_loss = tf.reduce_mean(entropy -cross_entropy)
            total_loss =( reconstruction_loss + kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}



class AutoEncoder(tf.keras.Model):
  def __init__(self, encoder,decoder, **kwargs):
    super(AutoEncoder, self).__init__(name='auto-encoder', **kwargs)
    self.encoder = encoder
    self.decoder = decoder

    self.loss_tracker = tf.keras.metrics.Mean(name = 'loss')

  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    x_mean, x_log_var = self.decoder(z)
    return z_mean, z_log_var, z, x_mean, x_log_var 

  def train_step(self, data):


    with tf.GradientTape() as tape :

            z_mean, z_log_var, z, reconstruction, x_log_var = self(data)
            #we compute the first loss : the log-likelyhood
            scale_x = tf.exp(x_log_var) #variance
            log_pdf = 0.5 * tf.reduce_sum(tf.pow(data-reconstruction, 2) / scale_x + x_log_var, axis = 1) #-log_pdf because we want to maximise it (SGD aim to minimize in keras)
            total_loss =  tf.reduce_mean(log_pdf) #tf.multiply(log_pdf, y)

    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    self.loss_tracker.update_state(total_loss)
    return {m.name: m.result() for m in self.metrics}