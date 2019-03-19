import tensorflow as tf
import numpy as np

print(tf.__version__)



def lrelu(x, leak=0.2):
    with tf.variable_scope("lrelu"):
        return tf.maximum(x, x*leak)

def batch_norm(inputs, name="batch_norm", reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        return tf.nn.batch_normalization(x=inputs,
                                        name=name)
    
    
def instance_norm(inputs, name="instance_norm", reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse) as scope:
        return tf.contrib.layers.instance_norm(inputs, reuse=reuse, scope=scope)

def conv3d(inputs, filters, kernel_size, strides=2, name=None, padding='same',
            reuse=tf.AUTO_REUSE):
    with tf.variable_scope("conv3d", reuse=reuse) as scope:
        return tf.layers.conv3d(inputs, filters, kernel_size, 
                strides, name=name, padding=padding, reuse=reuse)

        
def deconv3d(inputs, out_channels, kernel_size, strides, name, padding='SAME'):
    with tf.variable_scope("transpose_conv", reuse=False) as scope:
        #return tf.nn.conv3d_transpose(inputs, filters, output_shape, strides,
        #padding)
        return tf.layers.conv3d_transpose(inputs,out_channels,kernel_size,
                strides,padding=padding, name=name, reuse=False)

def tanh(inputs, scale=6.0, name=None, reuse=False):
    with tf.variable_scope('tanh', reuse=reuse):
        '''
        init = tf.constant_initializer(scale)
        a = tf.get_variable("a", shape=[1], dtype=tf.float32, initializer=init,
                            trainable=True)
        return a*tf.nn.tanh(inputs, name)
        '''
        # tanh is just a scaled sigmoid function: tanh(x) = 2*sigmoid(2x) -1
        init = tf.constant_initializer(scale)
        a = tf.get_variable("a", shape=[1], dtype=tf.float32, initializer=init, 
                            trainable=True)
        return a*tf.nn.sigmoid(inputs) - a/2.0

def relu(features, name=None):
    return tf.nn.relu(features, name)

def elu(features, name=None):
    return tf.nn.elu(features, name)

