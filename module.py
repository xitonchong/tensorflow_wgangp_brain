from memory_saving_gradients import gradients
import tensorflow as tf
import numpy as np
from opts import *

''' tf.depth_to_space cannot be used on 3D data '''

def build_resnet(z, input_channel=512, kernel_size=4, reuse=tf.AUTO_REUSE,
        strides=2, residule_number=5):
    with tf.device('/device:GPU:0'):
        with tf.variable_scope("resnet", reuse=reuse):
            
            ''' residule block '''
            def residule_block(x, filters=32, kernel_size=3, strides=1,
                    name='res', padding='SAME'):
                y = elu(instance_norm(conv3d(x, filters=filters, 
                    kernel_size=kernel_size, strides=strides,
                    padding=padding,
                    name=name+'_c1'), name+'_l1'))
                y = instance_norm(conv3d(y, filters=filters,
                    kernel_size=kernel_size, strides=strides,
                    padding=padding,
                    name=name+'_c2'), name+'_l2')
                return y + x

            l1 = tf.layers.dense(z, units=3*3*3*input_channel, use_bias=True,
                    name='g_project')
            l1 = tf.reshape(l1, [-1,3, 3, 3, input_channel])
            l2 = elu(instance_norm(deconv3d(l1, input_channel//2, 
                    kernel_size=kernel_size,
                    strides=2, 
                    name='l2'), name='inl2'))
            l3 = elu(instance_norm(deconv3d(inputs=l2,
                    out_channels=input_channel//4, kernel_size=kernel_size,
                    strides=2,
                    name='l3'), name='inl3'))
            l3 = tf.pad(l3, [[0,0],[0,0],[1,1],[0,0],[0,0]],"CONSTANT")
            l4 = elu(instance_norm(deconv3d(inputs=l3,
                    out_channels=input_channel//8, kernel_size=kernel_size,
                    strides=2, name='l4'), name='inl4'))
            l5 = elu(instance_norm(deconv3d(inputs=l4,
                    out_channels=input_channel//16, kernel_size=kernel_size,
                    strides=2, name='l5'), name='inl5'))
            # l5 shape (?, 48, 56, 48, 32)
            for i in range(residule_number):
                l5 = residule_block(l5, name='res{}'.format(i))

            l6 = elu(instance_norm(deconv3d(inputs=l5,
                    out_channels=input_channel//32, kernel_size=kernel_size,
                    strides=2, name='l6'), name='inl6'))
            l6 = tf.pad(l6, [[0,0],[0,0],[1,1],[0,0],[0,0]], "CONSTANT")
            l7 = tanh(deconv3d(inputs=l6,
                    out_channels=3, kernel_size=kernel_size, strides=2,
                    name='l7'))
            l7 = tf.pad(l7, [[0,0],[1,0],[1,0],[1,0],[0,0]], "CONSTANT")

            print("l1 shape ", l1.get_shape())
            print("l2 shape ", l2.get_shape())
            print("l3 shape ", l3.get_shape())
            print("l4 shape ", l4.get_shape())
            print("l5 shape ", l5.get_shape())
            print("l6 shape ", l6.get_shape())
            print("l7 shape ", l7.get_shape())

            return l7

if __name__ == '__main__':
    ''' build graph '''
    latent_variable = tf.placeholder(tf.float32, shape=[None,200])
    image = build_resnet(latent_variable)
    batch_size = 2
    z = np.random.normal(size=[batch_size, 200])

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        tf.global_variables_initializer().run()
        image_ = sess.run(image, feed_dict={ latent_variable: z})
        print("output shape ", image_.shape)
