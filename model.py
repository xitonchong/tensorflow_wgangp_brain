from __future__ import division
import os
import time
import math
import tensorflow as tf
import numpy as np
from opts import *
from utils import *


class GAN(object):
    def __init__(self, sess, input_r=193, input_a=229, input_s=193,
            crop=True, batch_size=8, y_dim=None, z_dim=200, 
            dataset_name='hcp', checkpoint_dir=None, sample_dir=None, 
            data_dir='./data'):

        '''
        args:
        sess: tensorfow session
        batch_size: the size of batch, shoudl be specified before training. 
        z_dim:  dimension of dim of z
        '''
        self.sess = sess
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.input_r = input_r
        self.input_a = input_a
        self.input_s = input_s
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.model_dir = 'model'
        self.build_model()

    def wasserstein_loss(self, predictions, labels):
        return predictions*labels
    
    
    def build_model(self):
        self.z_placeholder = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        
        self.fake_sample = self.build_generator(self.z_placeholder)
        self.real_placeholder = tf.placeholder(tf.float32,
                shape=[None,193,229,193,3])
        # histogram summary to be added to writer
        self.g_z_hist_summ = tf.summary.histogram("G_z_histogram", self.fake_sample)
        self.x_hist_summ = tf.summary.histogram("x_histogram", self.real_placeholder)
        
        with tf.name_scope('d_output'):
            self.d_real = self.build_discriminator(self.real_placeholder)
            self.d_fake = self.build_discriminator(self.fake_sample)
        self.d_real_summ = tf.summary.scalar('d_x', self.d_real)
        self.d_fake_summ = tf.summary.scalar('d_g_z', self.d_fake)

        with tf.name_scope('loss'):
            with tf.name_scope('g_loss'):
                self.g_loss = -tf.reduce_mean(self.d_fake) 
            with tf.name_scope('gradient_penalty_loss'):
                 # gradient penaly loss
                '''def interpolate(a, b):
                    shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
                    alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                    inter = a + alpha * (b - a)
                    inter.set_shape(a.get_shape().as_list())
                    return inter'''
                self.epsilon = tf.random_uniform(
                                shape=[self.batch_size, 1, 1, 1, 1],
                                minval=0.,
                                maxval=1.)
                x_hat = self.real_placeholder + self.epsilon*(self.fake_sample-self.real_placeholder)
                d_x_hat = self.build_discriminator(x_hat, reuse=tf.AUTO_REUSE)
                grad_d_x_hat = tf.gradients(d_x_hat, [x_hat])[0]
                
                slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_d_x_hat), axis=[1,2,3,4]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            with tf.name_scope('d_loss'):
                self.d_loss = - tf.reduce_mean(self.d_real) + tf.reduce_mean(self.d_fake) +\
                                10.0 * gradient_penalty
                
        self.d_loss_summ = tf.summary.scalar('d_loss', self.d_loss)
        self.g_loss_summ = tf.summary.scalar('g_loss', self.g_loss)
        self.gp_loss_summ = tf.summary.scalar('gp_loss', gradient_penalty)
       
        self.d_summ = tf.summary.merge([self.d_loss_summ, self.x_hist_summ, self.gp_loss_summ])
        self.g_summ = tf.summary.merge([self.g_loss_summ, self.g_z_hist_summ])
        #== collect variables =====
        self.g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope="generator")
        self.d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope="discriminator")
        
        self.saver = tf.train.Saver()

        print('trainable g')
        for var in self.g_vars:
            print(var)
        print('trainable d')
        for var in self.d_vars:
            print(var)
        



    def build_discriminator(self, inputs, out_channel=32, reuse=tf.AUTO_REUSE, kernel_size=4):
        with tf.device('/device:GPU:1'):
            with tf.variable_scope("discriminator", reuse=reuse) as scope:
                conv1 = lrelu(instance_norm(conv3d(inputs,out_channel, kernel_size,
                    name='d_conv1'), name='d_inl1'))
                conv2 = lrelu(instance_norm(conv3d(conv1, out_channel*2, kernel_size,
                    name='d_conv2'), name='d_inl2'))
                conv3 = lrelu(instance_norm(conv3d(conv2, out_channel*4, kernel_size,
                    name='d_conv3'), name='d_inl3'))
                conv4 = lrelu(instance_norm(conv3d(conv3, out_channel*8, kernel_size,
                    name='d_conv4'), name='d_inl4'))
                conv5 = lrelu(instance_norm(conv3d(conv4, out_channel*16, kernel_size,
                    name='d_conv5'), name='d_inl5'))
                conv6 = lrelu(instance_norm(conv3d(conv5, out_channel*32, kernel_size, 
                    name='d_conv6'), name='d_inl6'))
                # implement 1x1 convolution
                #conv6 = lrelu(instance_norm(conv3d(conv6, out_channel*8, 1, 1, 
                #    name='d_conv6'), name='d_inl6'))
                flatten_o = tf.layers.flatten(conv6)
                x = tf.layers.dense(flatten_o, units=32, use_bias=True,
                                    name='d_fc_o1')
                fc_o = tf.layers.dense(x, units=1, use_bias=False,
                         name='d_fc_o')

                return fc_o



    def build_generator(self, z, input_channel=512, reuse=tf.AUTO_REUSE, kernel_size=4):
        with tf.device('/device:GPU:0'):
            with tf.variable_scope("generator", reuse=reuse) as scope:
                l1 = tf.layers.dense(z, units=3*3*3*input_channel, use_bias=True, name='g_project')
                l1 = tf.reshape(l1, [-1,3,3,3,input_channel])#list bracket is also counted as a dimension
              
                l2 = elu(instance_norm(deconv3d(inputs=l1, 
                    out_channels=input_channel//2, kernel_size=kernel_size, strides=2,
                    name='l2'), name='inl2'))
                l3 = elu(instance_norm(deconv3d(inputs=l2,
                    out_channels=input_channel//4, kernel_size=kernel_size, strides=2,
                    name='l3'), name='inl3'))
                l3 = tf.pad(l3, [[0,0],[0,0],[1,1],[0,0],[0,0]],"CONSTANT")
                l4 = elu(instance_norm(deconv3d(inputs=l3,
                    out_channels=input_channel//8, kernel_size=kernel_size, strides=2,
                    name='l4'), name='inl4'))
                l5 = elu(instance_norm(deconv3d(inputs=l4, 
                    out_channels=input_channel//16, kernel_size=kernel_size, strides=2,
                    name='l5'), name='inl5'))
                l6 = elu(instance_norm(deconv3d(inputs=l5,
                    out_channels=input_channel//32, kernel_size=kernel_size, strides=2,
                    name='l6'), name='inl6'))
                l6 = tf.pad(l6, [[0,0],[0,0],[1,1],[0,0],[0,0]], "CONSTANT")
                l7 = tanh(deconv3d(inputs=l6,
                    out_channels=3, kernel_size=kernel_size, strides=2,
                    name='l7'))
                l7 = tf.pad(l7, [[0,0],[1,0],[1,0],[1,0],[0,0]], "CONSTANT")
                return l7
    
    def test(self, config):
        print('in test here')

        #===  load model =======
        could_load, checkpoint_counter = self.load(config.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS ", counter)
        else:
            print(" [!] Load failed...")
        #----------------------------

        for epoch in range(config.generate_test_images):
            filename = './C6493_FSPGRBrainExtractionBrain_diffeo1InverseWarp.nii.gz'
            _, affine, hdr = load(filename)
            sample_z = np.random.normal(0, 1, (1,self.z_dim))
            #sample_z = np.random.uniform(-1, 1, (config.batch_size, self.z_dim))
            _fake = self.sess.run(self.fake_sample, feed_dict={
                        self.z_placeholder: sample_z})
            _fake = np.squeeze(_fake)
            _fake = np.expand_dims(_fake, axis=3)
            assert _fake.shape == (193,229,193,1,3)
            path = os.path.join(config.sample_dir, "gen_brain_{}_diffeo1InverseWarp.nii.gz")
            # save is referring to sampling, gonna change later
            save(_fake, affine, path.format(epoch), hdr)


    def train(self, config):
        with tf.name_scope('train'):
            if config.optimizer is 'rmsprop':
                d_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(
                        self.d_loss, var_list=self.d_vars)
                g_optim = tf.train.RMSPropOptimizer(config.g_learning_rate).minimize(
                        self.g_loss, var_list=self.g_vars)
            else:
                d_optim = tf.train.AdamOptimizer(config.learning_rate,
                        beta1=config.beta1).minimize(self.d_loss, 
                                                    var_list=self.d_vars)
                g_optim = tf.train.AdamOptimizer(config.g_learning_rate,
                        beta1=config.beta1).minimize(self.g_loss,
                                                    var_list=self.g_vars)

        print('using ', config.optimizer)
        
        #merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(config.log_dir + '/train', self.sess.graph)
        tf.global_variables_initializer().run()

        #sample_x, no_samples = get_input(config.data_dir, config.batch_size)
        
        could_load, checkpoint_counter = self.load(config.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS ", counter)
        else:
            print(" [!] Load failed...")
        
        batch_no = no_samples//(self.batch_size*config.n_critic)
        for epoch in range(config.epochs+1):
            sample_x, no_samples = get_input(config.data_dir, config.batch_size)
            for idx in range(batch_no): 
                for n in range(config.n_critic):
                    sample_z = np.random.normal(0, 1, (config.batch_size, self.z_dim))
                    _sample_x = self.sess.run(sample_x)
                    _, _d_loss, d_summary, _d_real, _d_fake= self.sess.run([d_optim, self.d_loss,
                                                                            self.d_summ,
                                                                            self.d_real, self.d_fake], 
                                                                     feed_dict={
                                                                        self.z_placeholder: sample_z, 
                                                                        self.real_placeholder: _sample_x})
                    print("   {},{}  D(x): {},    D(G(z)): {}".format(idx,n,_d_real, _d_fake))

                # update generator k times per discriminator update
                _, _g_loss, g_summary,  _d_fake = self.sess.run([g_optim, self.g_loss, self.g_summ, 
                                                                self.d_fake], 
                                                         feed_dict={
                                                            self.z_placeholder: sample_z})

                print("{},{} d loss: {},  g_loss: {}".format(epoch,idx,  _d_loss, _g_loss))
                #print("       D(x): {},    D(G(z)): {}".format(_d_real, _d_fake))

                train_writer.add_summary(d_summary, epoch*batch_no + idx)
                train_writer.add_summary(g_summary, epoch*batch_no + idx)

                    
            # save every few epoch
            self.save(config.checkpoint_dir, epoch)
            if epoch % 2 == 0:
                filename = './C6493_FSPGRBrainExtractionBrain_diffeo1InverseWarp.nii.gz'
                _, affine, hdr = load(filename)
                sample_z = np.random.normal(0, 1, (1,self.z_dim))
                #sample_z = np.random.uniform(-1, 1, (config.batch_size, self.z_dim))
                _fake = self.sess.run(self.fake_sample, feed_dict={
                        self.z_placeholder: sample_z})
                _fake = np.squeeze(_fake)
                _fake = np.expand_dims(_fake, axis=3)
                assert _fake.shape == (193,229,193,1,3)
                path = os.path.join(config.sample_dir, "gen_brain_{}")
                print('saving ', path.format(epoch))
                # save is referring to sampling, gonna change later
                save(_fake, affine, path.format(epoch), hdr)
                print('saved at epoch {}'.format(epoch))


                
    def save(self, checkpoint_dir, step):
        model_name = "WGANGP.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0  
