import numpy as np
from PIL import Image
import tensorflow as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class Encoder():

    def __init__(self, in_channels:3, latent_dimem:64):
        super(Encoder, self).__init__()

        self.conv1 = layers.Conv2d(in_channels,32,kernal_size=3,stride=2,padding=1
                            kernal_initializer=layers.XavierUniform())
        self.bn1 = layers.Batchnorm2d()

        self.conv2 = layers.Conv2d(64,kernal_size=3,stride=2,padding=1
                            kernal_initializer=layers.XavierUniform())
        self.bn2 = layers.Batchnorm2d()

        self.conv3 = layers.Conv2d(128,kernal_size=3,stride=2,padding=1
                            kernal_initializer=layers.XavierUniform())
        self.bn3 = layers.Batchnorm2d()
        self.conv4 = layers.Conv2d(256,kernal_size=3,stride=2,padding=1
                            kernal_initializer=layers.XavierUniform())                  
        self.bn4 = layers.Batchnorm2d()

        self.droput = layers.Dropout(0.4)

        self.mean = layers.Dense(latent_dimem*4)
        self.variance = layers.Dense(latent_dimem*4)

        self.latent_dimem = latent_dimem

    def forward(self,x):

        x= self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.gelu(x)

        x= self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.gelu(x)

        x= self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.gelu(x)

        x= self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.gelu(x)

        x = layers.flatten(x)
        x = self.droput(x)

        mean = self.mean(x)
        variance = self.variance(x)

        return mean, variance
    
class Decoder():

    def __init__(self, out_channels:3, latent_dimem:64):
        super(Decoder, self).__init__()

        self.transfer = layers.Dense(256*16*16)

        self.dropout = layers.Dropout(0.4)

        self.deconv1 = layers.Conv2dTranspose(128,kernal_size=3,stride=2,padding=1
                            kernal_initializer=layers.XavierUniform())
        self.bn1 = layers.Batchnorm2d()

        self.deconv2 = layers.Conv2dTranspose(64,kernal_size=3,stride=2,padding=1
                            kernal_initializer=layers.XavierUniform())
        self.bn2 = layers.Batchnorm2d()

        self.deconv3 = layers.Conv2dTranspose(32,kernal_size=3,stride=2,padding=1
                            kernal_initializer=layers.XavierUniform())
        self.bn3 = layers.Batchnorm2d()

        self.deconv4 = layers.Conv2dTranspose(out_channels,kernal_size=3,stride=2,padding=1
                            kernal_initializer=layers.XavierUniform())                  

        self.latent_dimem = latent_dimem

    def forward(self,z):

        x = self.transfer(z)
        x = tf.reshape(x,(-1,256,16,16))
        x = self.dropout(x)

        x= self.deconv1(x)
        x = self.bn1(x)
        x = tf.nn.gelu(x)

        x= self.deconv2(x)
        x = self.bn2(x)
        x = tf.nn.gelu(x)

        x= self.deconv3(x)
        x = self.bn3(x)
        x = tf.nn.gelu(x)

        x= self.deconv4(x)
        x = tf.nn.tanh(x)

        return x

class VAE():

    def __init__ (self, in_channels, out_channels, latent_dimen):
        super(VAE, self).__init__()

        self.Encoder = Encoder(in_channels,out_channels, latent_dimen)
        self.Decoder = Decoder(in_channels,out_channels, latent_dimen)

    def reparametrize(self, mean, variance):
        
        variance = tf.clip_by_value(variance, -20.0, 2.0)
        std = tf.exp(0.5*variance)
        std = std+1e-8
        noise =  tf.random.normal(shape=std.shape)
        z = mean + noise*std

        z = tf.reshape(z,(z,np.shape()[0], self.latent_dimen,4,4))
        return z
    
    def forward(self,x):

        mean,variance = self.Encoder.forward(x)
        z = self.reparametrize(mean, variance)
        reconstruction = self.Decoder.forward(z)
        return reconstruction, mean, variance
    
    def encode(self,x, mean, variance):

        mean,variance - self.Encoder.forward(x)
        return self.reparametrize(mean, variance)
    
    def decode(self,z):
        
        return self.Decoder.forward(z)
    
    def VAE_loss(self,x, reconstruction, mean, variance):

        reconstruction_loss = tf.reduce_mean(tf.square(x-reconstruction))
        kl_loss = -0.5 * tf.reduce_mean(1 + variance - tf.square(mean) - tf.exp(variance))
        total_loss = reconstruction_loss + kl_loss
        return total_loss
    
class Attention():

    def __init__(self,channels: int):
        super().__init__()
        self.channels = channels

        self.query = layers.Dense(channels)
        self.key = layers.Dense(channels)
        self.value = layers.Dense(channels)     
        self.out = layers.Dense(channels)
    
    def forward(self,x):

        batch, size, channels,width = x.shape

        x_flatten = tf.reshape(x,[batch,size*width,channels])

        query = self.query(x_flatten)
        key = self.key(x_flatten)
        value = self.value(x_flatten)
                        
        scale = tf.sqrt(tf.cast(self.channels, tf.float32))
        score = tf.matmul(query, key, transpose_b=True) / scale
        weights = tf.nn.softmax(score, axis=1)

        attention_output = tf.matmul(weights, value)
        self.out(attention_output)

        attention_output = tf.reshape(attention_output,[batch,size,width,channels])
                                        
        return attention_output + x
class ResiduelBlock(keras.Model):

    def __init__(self, channels:int, time_embed: int):
        super().__init__()

        self.res1 = layers.Conv2d(channels, 3, padding=1, kernal_initializer=layers.XavierUniform() )
        self.bn_res1 = layers.BatchNorm2d()

        self.res2 = layers.Conv2d(channels, 3, padding=1, kernal_initializer= layers.XavierUniform())
        self.bn_res2 = layers.BatchNorm2d()
        self.time_mlp = layers.Dense(channels)  

    def forward(self,x,time_embed):
        
        h = self.res1(x)
        h = self.bn_res1(h)
        h = tf.nn.silu(h)

        t = self.time_mlp(time_embed)
        t = tf.nn.silu(t)

        batch_size = x.shape()[0]
        channels = x.shape()[1]
        t = tf.reshape(t,[batch_size,channels,1,1])
        h = h+t

        h = self.res2(h)
        h = self.bn_res2(h)

        return tf.nn.silu(h+x)
        
class UNet(keras.Model):

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()

        up_res1 = 
    
