import os
import numpy as np

import tensorflow as tf
import pprint
from  utils import *
from model import *
import memory_saving_gradients
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

pp = pprint.PrettyPrinter()

flags = tf.app.flags
#define integer
flags.DEFINE_integer("epochs", 100, "epoch to train [100]")
flags.DEFINE_integer("batch_size", 4, "default batch = 4")
flags.DEFINE_integer("generate_test_images", 1, "number of images to geneerate during test. [10]")
flags.DEFINE_integer("n_critic", 5, "generator per discriminator update")
#define float
flags.DEFINE_float("learning_rate", 0.00005, "learning rate of adam, 0.0002")
flags.DEFINE_float("g_learning_rate", 0.00005, "learning rate of generator network [0.00005]")
flags.DEFINE_float("beta1", 0.5, "first momentum of adam [0.5]")
#define string
flags.DEFINE_string("optimizer", "rmsprop", 'adam optimizer')
flags.DEFINE_string("dataset", "hcp", "the name of dataset [hcp]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "direcotry name to save checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "/home/xitonchong/data/", "check directory")
flags.DEFINE_string("sample_dir", "./samples", "directory to save the output brain")
flags.DEFINE_string("log_dir", './logs', "directory to store tensorboard history")
#define boolean
flags.DEFINE_boolean("train", False, "true for training, false of testing,  [true]")
flags.DEFINE_boolean("test", True, "true for sampling, [True]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    path = os.path.join(FLAGS.sample_dir, "gen_brain_{}")
    print('===== saving path: ', path.format('1'))
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        print('main running')
        gan = GAN(sess=sess, batch_size=FLAGS.batch_size,
                checkpoint_dir=FLAGS.checkpoint_dir, 
                sample_dir=FLAGS.sample_dir,
                data_dir=FLAGS.data_dir)

        if FLAGS.train:
            print('===== disc/gen update ', FLAGS.n_critic)
            gan.train(FLAGS)
        if FLAGS.test:
            print('===== producing fake sample ==============')
            gan.test(FLAGS)
    #show_all_variables()



if __name__ == '__main__':
    tf.app.run()


