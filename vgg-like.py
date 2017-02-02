import tensorflow as tf
import random

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
lr = tf.placeholder(tf.float32)


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

W_fc1 = weight_variable([8 * 8 * 256, 4096])
b_fc1 = bias_variable([4096])

h_pool4_flat = tf.reshape(h_pool4, [-1, 8*8*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([4096, 4096])
b_fc2 = bias_variable([4096])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([4096, 10])
b_fc3 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

def train(X_train, Y_train, X_validation, Y_validation):
    train_tuple = zip(X_train, Y_train)

    for i in range(16000):

        batch = random.sample(train_tuple, 32)
        batch_X = [j[0] for j in batch]
        batch_Y = [j[1] for j in batch]
        if i%1000==0:
                va = 0
                for j in xrange(0, len(X_train), 32):
                    mx = min(j+32, len(X_train))
                    va = va + (accuracy.eval(feed_dict={x: X_train[j:mx], y_: Y_train[j:mx], keep_prob: 1.0}))*(mx-j)
                va /= len(X_train)
                print "train", va

                va = 0
                for j in xrange(0, len(X_validation), 32):
                    mx = min(j+32, len(X_validation))
                    va = va + (accuracy.eval(feed_dict={x: X_validation[j:mx], y_: Y_validation[j:mx], keep_prob: 1.0}))*(mx-j)
                va /= len(X_validation)
                print "validation", va

        if i%10 == 0 and i!=0:
            print "step", i, "loss", loss_val

        if i<4000:
            _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x:batch_X, y_: batch_Y, keep_prob: 0.5, lr: 2e-4})
        else:
            _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x:batch_X, y_: batch_Y, keep_prob: 0.5, lr: 2e-5})
