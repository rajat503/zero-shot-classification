import tensorflow as tf
import random


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
    y_ = tf.placeholder(tf.float32, shape=[None, 50])
    lr = tf.placeholder(tf.float32)

    W_conv1 = weight_variable([3, 3, 3, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    W_conv3 = weight_variable([3, 3, 32, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

    W_conv4 = weight_variable([3, 3, 128, 256])
    b_conv4 = bias_variable([256])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    W_fc1 = weight_variable([8 * 8 * 256, 1024])
    b_fc1 = bias_variable([1024])

    h_pool4_flat = tf.reshape(h_pool4, [-1, 8*8*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 1024])
    b_fc2 = bias_variable([1024])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    W_fc3 = weight_variable([1024, 50])
    b_fc3 = bias_variable([50])

    y_conv= tf.matmul(h_fc2, W_fc3) + b_fc3

    loss = tf.nn.l2_loss(y_conv-y_)

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess =  tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

def train(X_train, Y_train):
    train_tuple = zip(X_train, Y_train)

    for i in range(10000):
        batch = random.sample(train_tuple, 32)
        batch_X = [j[0] for j in batch]
        batch_Y = [j[1] for j in batch]

        if i%10 == 0 and i!=0:
            print "step", i, "loss", loss_val

        if i<10000:
            rate = 2e-4
        else:
            rate = 2e-5

        _, loss_val = sess.run([train_step, loss], feed_dict={x:batch_X, y_: batch_Y, keep_prob: 0.8, lr: rate})

def predict(X):
    prediction = sess.run([y_conv], feed_dict={x: X, keep_prob: 1.0})
    return prediction[0]
