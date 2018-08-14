import tensorflow as tf
from random import randint
import numpy as np

# https://blog.csdn.net/wl1710582732/article/details/78597511
# http://www.cs.toronto.edu/~kriz/cifar.html
file_dir = '/Users/ygu/Code/2_cnn_cifar/data/cifar-10-batches-py/'

batch_size = 64
n_epoches = 100
img_channels = 3
img_size = 32

def unpickle(file_dir):
    import cPickle
    with open(file_dir, 'rb') as fo:
        r_dict = cPickle.load(fo)
    # data -- a 10000x3072 numpy array of uint8s (3072=32*32*3)
    # labels -- a list of 10000 numbers in the range 0-9
    return r_dict


#
# load data as <#data> * <height> * <width> * <#channels>
#
def load_data(file_dir, mode = 'test'):
    import cPickle

    if mode == 'train':
        data = np.empty((0, img_size*img_size*img_channels))
        label = np.array([])
        for idx in range(1,6):
            with open(file_dir + '/data_batch_' + str(idx), 'rb') as fo:
                r_dict = cPickle.load(fo)
                data = np.append(data, r_dict['data'], axis = 0)
                label = np.append(label, r_dict['labels'])
    else:
        with open(file_dir + '/test_batch', 'rb') as fo:
            r_dict = cPickle.load(fo)
            data = np.array(r_dict['data'])
            label = np.array(r_dict['labels'])

    data = data.reshape([-1, img_channels, img_size, img_size])
    data = data.transpose([0, 2, 3, 1])
    print data.shape
    return data, label


def cnn(train_data, train_label, test_data, test_label):
    n_output_layer = 10
    X = tf.placeholder(tf.float32)
    X = tf.reshape(X, [-1, img_size, img_size, img_channels])
    Y = tf.placeholder(tf.int64, shape = None)

    layer_1 = tf.layers.conv2d(inputs = X, filters = 50, kernel_size = 5, padding="same", activation = tf.nn.relu)
    layer_1 = tf.layers.max_pooling2d(inputs = layer_1, pool_size = 3, strides = 2, padding="same")
    # layer_1: -1 * 16 * 16 * 50

    #print '====='
    #print layer_1.get_shape()

    fc1 = tf.contrib.layers.flatten(layer_1)
    new_dim = fc1.get_shape()[1].value
    fc2_w_b = {
            'w': tf.Variable(tf.random_normal(
                shape = [new_dim, n_output_layer], mean = 0, stddev = 0.1)),
            'b': tf.Variable(tf.random_normal(
                shape = [n_output_layer], mean = 0, stddev = 0.1)) }
    fc2 = tf.add(tf.matmul(fc1, fc2_w_b['w']), fc2_w_b['b'])
    #print fc2.get_shape()       # [?, 10]
    output_layer = fc2

    #layer_3 = tf.nn.max_pool(layer_3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    #layer_3 = tf.nn.avg_pool(layer_3, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID')
    #output_layer = tf.reshape(layer_3, [-1,n_output_layer])

    Y_onehot = tf.one_hot(Y, depth = n_output_layer)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = Y_onehot, logits = output_layer))
    optm = tf.train.AdamOptimizer(1e-3).minimize(cost_func)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int( len(train_data) / batch_size )
        for epoch in range(n_epoches):
            total_cost = 0
            for _ in range(total_batch):
                t = randint(0, len(train_data)-batch_size)
                x_batch = train_data[t:t+batch_size]
                y_batch = train_label[t:t+batch_size]

                cost, _ = sess.run([cost_func, optm], feed_dict = {
                    X: x_batch, 
                    Y: y_batch
                    })
                total_cost += cost
            print '{}\t{}'.format(epoch, cost)

            res = tf.equal(tf.argmax(output_layer, axis = 1), Y)
            acc = tf.reduce_mean(tf.cast(res, tf.float32))
            print 'Training accuracy: {}'.format(acc.eval({X: train_data, Y:train_label}))
            print 'Test accuracy: {}'.format(acc.eval({X: test_data, Y:test_label}))



def dnn(train_data, test_data):
    n_features = 3072
    X = tf.placeholder(tf.float32, shape = [None, n_features])
    Y = tf.placeholder(tf.int64, shape = None)

    n_layer_1 = 512
    n_layer_2 = 256
    n_output_layer = 10

    layer_1_w_b = {'w': tf.Variable(tf.random_normal([n_features, n_layer_1])),
            'b': tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
            'b': tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b = {'w': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
            'b': tf.Variable(tf.random_normal([n_output_layer]))}
    
    layer_1 = tf.add(tf.matmul(X, layer_1_w_b['w']), layer_1_w_b['b'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w']), layer_2_w_b['b'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w']), layer_output_w_b['b'])

    Y_onehot = tf.one_hot(Y, depth = n_output_layer)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = Y_onehot, logits = layer_output))
    optm = tf.train.AdamOptimizer(1e-3).minimize(cost_func)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int( len(train_data['data']) / batch_size )
        for epoch in range(n_epoches):
            total_cost = 0
            for _ in range(total_batch):
                t = randint(0, len(train_data['data'])-batch_size)
                x_batch = train_data['data'][t:t+batch_size]
                y_batch = train_data['labels'][t:t+batch_size]

                cost, _ = sess.run([cost_func, optm], feed_dict = {
                    X: x_batch, 
                    Y: y_batch
                    })
                total_cost += cost
            print '{}\t{}'.format(epoch, cost)

            res = tf.equal(tf.argmax(layer_output, axis = 1), Y)
            acc = tf.reduce_mean(tf.cast(res, tf.float32))
            print 'Training accuracy: {}'.format(acc.eval({X: train_data['data'], Y:train_data['labels']}))
            print 'Test accuracy: {}'.format(acc.eval({X: test_data['data'], Y:test_data['labels']}))

            # training:test = 10000:10000
            # Training acc: 0.806 after 100 epoches :o
            # Test acc: 0.35~0.40 after 100 epoches



def train():
    #train_d = unpickle(file_dir + 'data_batch_1')
    #test_d = unpickle(file_dir + 'test_batch')
    #dnn(train_d, test_d)

    train_d, train_l = load_data(file_dir, mode = 'train')
    test_d, test_l = load_data(file_dir, mode = 'test')
    cnn(train_d, train_l, test_d, test_l)


train()

