import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

def generateData():
    x = np.random.normal(size=[500, 2])
    label = []
    label_true = []
    for i in range(x.shape[0]):
        if x[i][1] >= x[i][0] and x[i][1] >= -x[i][0]:
            label.append([1., 0., 0., 0.])
            label_true.append(0)
        elif -x[i][0] <= x[i][1] <= x[i][0]:
            label.append([0., 1., 0., 0.])
            label_true.append(1)
        elif x[i][1] <= x[i][0] and x[i][1] <= -x[i][0]:
            label.append([0., 0., 1., 0.])
            label_true.append(2)
        else:
            label.append([0., 0., 0., 1.])
            label_true.append(3)
    return x, np.array(label), np.array(label_true)


def buildNet(x_in, input_features, n1, n2, n_out):
    w1 = tf.Variable(tf.random_normal([input_features, n1]))
    b1 = tf.Variable(tf.random_normal([n1]))
    h1 = tf.nn.relu(tf.matmul(x_in, w1)+b1)
    w2 = tf.Variable(tf.random_normal([n1, n2]))
    b2 = tf.Variable(tf.random_normal([n2]))
    h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)
    w3 = tf.Variable(tf.random_normal([n2, n_out]))
    b3 = tf.Variable(tf.random_normal([n_out]))
    h3 = tf.nn.softmax(tf.matmul(h2, w3)+b3)
    var_list = [w1, w2, w3, b1, b2, b3]
    return h3, var_list


if __name__ == '__main__':
    input_features = 2
    n1 = 5
    n2 = 6
    n_out = 4
    train_x, train_y, true_y_label = generateData()
    X = tf.placeholder(shape=[None, input_features], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, n_out], dtype=tf.float32)

    out, var_list = buildNet(X, input_features, n1, n2, n_out)

    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y-out), reduction_indices=[1]))
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=out))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(2):
        for i in range(200):
            sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
            pred = sess.run(tf.argmax(tf.nn.softmax(out), axis=1), feed_dict={X: train_x, Y: train_y})
            print('epoch: ', epoch, 'pred:', pred, 'loss:', sess.run(loss, feed_dict={X: train_x, Y: train_y}))
    ground_true = np.argmax(train_y, axis=1)
    hits = 0
    print('ground_truth: ', ground_true)
    for t in range(train_x.shape[0]):
        if pred[t] == ground_true[t]:
            hits += 1
print('acc:', hits/train_x.shape[0])
# sess = tf.Session()
# my_tf_multiclass_graph = tf.summary.FileWriter('./my_tf-multiclass_graph', sess.graph)
