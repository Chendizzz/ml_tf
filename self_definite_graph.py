import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
graph = tf.Graph()
with graph.as_default():
    in_1 = tf.placeholder(tf.float32, shape=[], name='input_a')
    in_2 = tf.placeholder(tf.float32, shape=[], name='input_b')

