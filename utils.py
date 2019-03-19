from __future__ import division
import math
import json
import random
import pprint
import tensorflow as tf
import os
import glob
import numpy as np
from time import gmtime, strftime, time
import nibabel as nib
import tensorflow as tf
import tensorflow.contrib.slim as slim

def load(filename):
    fobj = nib.load(filename)
    img = fobj.get_data()
    affine = fobj.affine
    header = fobj.header
    return img, affine, header

# save NIFTI image
def save(data, affine, filepath, header):
    img_nifti = nib.Nifti1Image(data, affine, header=header)
    nib.save(img_nifti, filepath)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def _parser_fn(value, parse_type=tf.float32):
    return tf.parse_tensor(value, parse_type)


def get_input(data_dir, batch_size=2):
    with tf.device('/cpu:0'):
        brain_dir = data_dir
        files = glob.glob(os.path.join(brain_dir, '*.tfrecords'))
        no_files = len(files)
        assert no_files > 0
        print("tfrecords files ", no_files)
        serialized_dataset = tf.data.TFRecordDataset(files)
        raw_dataset = serialized_dataset.map(_parser_fn)
        raw_dataset = raw_dataset.repeat()
        raw_dataset = raw_dataset.batch(batch_size)
        iterator = raw_dataset.make_one_shot_iterator()
        real_data = iterator.get_next()
    return real_data,  no_files


if __name__ == '__main__':
    pp = pprint.PrettyPrinter()
    # pp.pprint(
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 2, "default batch = 2")
    flags.DEFINE_string("data_dir", '/home/tensorflow-tutorial/data/tfrecords_folder/', 'default dir')
    FLAGS = flags.FLAGS

    real_data = get_input(FLAGS.data_dir, 3)
    with tf.Session(config=config) as sess:
        for epoch in range(10):
            _real = sess.run(real_data)
            print("shape of data ",  _real.shape)










