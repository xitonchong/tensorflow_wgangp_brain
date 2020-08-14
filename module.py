from memory_saving_gradients import gradients
import tensorflow as tf
import numpy as np
from opts import *

''' tf.depth_to_space cannot be used on 3D data '''

def build_resnet(z, input_channel=512, kernel_size=4, reuse=tf.AUTO_REUSE,
		strides=2, residule_number=5):
	#with tf.device('/device:GPU:0'):
	with tf.device('/device:CPU:0'):
		with tf.variable_scope("generator", reuse=reuse):
			
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

def build_discriminator( inputs, out_channel=32, reuse=tf.AUTO_REUSE, 
					kernel_size=4):
	#with tf.device('/device:GPU:1'):
	with tf.device('/device:CPU:0'):
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
			conv6 = lrelu(instance_norm(conv3d(conv6, out_channel*8, 1, 1,
				name='d_1x1conv7'), name='d_inl7'))
			flatten_o = tf.layers.flatten(conv6)
			x = tf.layers.dense(flatten_o, units=32, use_bias=True,
					name='d_fc_o1')
			fc_o = tf.layers.dense(x, units=1, use_bias=False,
					name='d_fc_o')

			return fc_o

def build_resnet_discriminator(x, out_channel=32, reuse=tf.AUTO_REUSE,
					kernel_size=4, residule_number=5):

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

	with tf.device('/device:GPU:1'):
		with tf.variable_scope("discriminator", reuse=reuse) as scope:
			for i in range(1,7):
				# return 1 .. 6
				x = lrelu(instance_norm(conv3d(x, 
					np.clip(out_channel * np.power(2,i), 
						a_min=out_channel,
						a_max=out_channel*8), 
					kernel_size,
					name='d_conv{}'.format(i)), 
					name='d_inl{}'.format(i)))
			# 1x1 convolution
			x = relu(instance_norm(conv3d(x, 
					out_channel, kernel_size=1, strides=1, 
					name='d_1x1conv7'), 
					name='d_inl7'))
			for i in range(8, residule_number+8):
				x = residule_block(x, name='res_{}'.format(i))
			# flatten layer
			flatten_o = tf.layers.flatten(x)
			x = tf.layers.dense(flatten_o, units=32, use_bias=True,
					name='d_fc_o1')
			fc_o = tf.layers.dense(x, units=1, use_bias=False,
					name='d_fc_o')

			return fc_o


def build_generator( z, input_channel=512, reuse=tf.AUTO_REUSE, 
						kernel_size=4, residule_number=None):

	# residule_number is not used in this network, supposely need to 
	#	pass function decorator to accomodate different kwargs for
	#	build_generator and build_resnet
	with tf.device('/device:GPU:0'):
		with tf.variable_scope("generator", reuse=reuse) as scope:
			l1 = tf.layers.dense(z, units=3*3*3*input_channel, use_bias=True, name='g_project')
			l1 = tf.reshape(l1, [-1,3,3,3,input_channel])
			#list bracket is also counted as a dimension

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
