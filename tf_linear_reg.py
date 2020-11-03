import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
input_data = np.array([-1, 1, 2, 3, 7, 9])
out_data = np.array([-2, 3, 5, 6, 17, 25])

# by self-made model
x = tf.placeholder(name='input', dtype=tf.float32)
# y = tf.placeholder(name='y', dtype=tf.float32)
y = tf.constant(out_data, tf.float32)
w = tf.Variable(tf.random.uniform([1]), name='slope', dtype=tf.float32)
b = tf.Variable(tf.random.uniform([1]), name='constant', dtype=tf.float32)
pred = tf.add(tf.multiply(w, x), b)
loss = tf.reduce_sum(tf.pow(pred-y, 2))/20

optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss, var_list=[w, b])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(200):
        sess.run(optimizer, feed_dict={x: input_data, y: out_data})
        print('w: ', sess.run(w), 'b: ', sess.run(b), 'loss: ', sess.run(tf.reduce_sum(tf.pow((tf.add(tf.multiply(w, input_data), b))-out_data, 2))/20))


# by using existed dense layer model
prediction =tf.Variable(tf.random.uniform([6]), tf.float32)
input_layer = tf.layers.dense(x, 10)
output_layer = tf.layers.dense(input_layer, prediction)
loss_function = tf.losses.mean_squared_error()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(200):
        sess.run()


