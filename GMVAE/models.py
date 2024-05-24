from tensorflow.keras import layers
import tensorflow as tf 

import numpy as np 
import scipy.stats as sp 
import openturns as ot 

#variationnal inference for the class probability class 
class q_y(layers.Layer):
    def __init__(self, num_components, input_dim, **kwargs):
        super(q_y, self).__init__(**kwargs)
        self.num_components = num_components
        self.input_dim = input_dim

        #layers 
        self.hidden1 = layers.Dense(252, activation = 'relu')
        self.hidden2 = layers.Dense(128, activation = 'relu')
        self.hidden3 = layers.Dense(64, activation = 'relu')
        self.logit = layers.Dense(num_components, activation = 'relu')
        self.out = layers.Dense(num_components, activation = 'softmax')
    
    def call(self, inputs):
        y = self.hidden1(inputs)
        y = self.hidden2(y)
        y = self.hidden3(y)
        y = self.logit(y)
        prob = self.out(y)
        sampled_y = self.sampling(prob)
        return prob, sampled_y
    
    def sampling(self, prob):
        return tf.random.categorical(prob, self.input_dim)


class Encoder(layers.Layer):
    def __init__(self, input_dim, latent_dim, n_components, **kwargs):
        super(Encoder, self).__init__(name='encoder', **kwargs)
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.transform_y = layers.Dense(input_dim, name = 'y_transformation')
        self.enc1 = layers.Dense(input_dim, activation='relu')
        self.enc2 = layers.Dense(64, activation = 'relu')
        self.enc3 = layers.Dense(32, activation = 'relu')
        self.mean_z = layers.Dense(latent_dim, name = 'z_mean')
        self.logvar_z = layers.Dense(latent_dim, name = 'z_log_var')
        self.qy = q_y(n_components, input_dim)

    
    def call(self, inputs): #inputs x 
        prob, y = self.qy(inputs)
        xy = tf.concat((inputs,self.transform_y(y)), axis=1) #transformation of y from [1, .., k] to real numbers
        x = self.enc1(xy)
        x = self.enc2(x)
        x = self.enc3(x)
        z_mean = self.mean_z(x)
        z_log_var = self.logvar_z(x)
        z = self.sampling([z_mean, z_log_var, prob])
        return z_mean, z_log_var, z
    
    def sampling(self, inputs):
        z_mean, z_log_var, y_proba = inputs
        z = tf.random.normal(shape= (self.n_components, self.latent_dim), mean= z_mean, stddev=tf.exp(0.5*z_log_var)) #n_components z 
        return tf.reduce_sum(y_proba * z) 

#########

class P_z(layers.Layer):
    def __init__(self, n_components, latent_dim, **kwargs):
        super(P_z, self).__init__(name = 'Pz', **kwargs)
        self.n_components = n_components
        self.latent_dim = latent_dim

        self.hidden1 = layers.Dense(252, activation = 'relu')
        self.hidden2 = layers.Dense(128, activation = 'relu')
        self.hidden3 = layers.Dense(64, activation = 'relu')
        self.mean_z = layers.Dense(latent_dim, name = 'PZ_mean')
        self.log_var_z = layers.Dense(latent_dim, name = 'PZ_logvar')

    def call(self, inputs):
        z = self.hidden1(inputs)
        z= self.hidden2(z)
        z = self.hidden3(z)
        z_mean = self.mean_z(z)
        z_log_var = self.log_var_z(z)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z
    
    def sampling(self, inputs):
        z_mean, z_log_var = inputs
        return tf.random.normal(shape= (self.n_components, self.latent_dim), mean= z_mean, stddev= tf.exp(z_log_var *0.5))




