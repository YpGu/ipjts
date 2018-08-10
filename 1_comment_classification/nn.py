import tensorflow as tf
import numpy as np

pos_file = './data/pos.txt'
neg_file = './data/neg.txt'
n_voc = -1
n_epoch = 20

def read_data():
    voc = {}        # word -> id

    lid = 0; word_id = 0
    with open(pos_file) as fp:
        for line in fp:
            if lid%5 != 0:
                ls = line.split(' ')
                for w in ls:
                    w = w.strip()
                    if len(w) < 3: continue
                    if w not in voc:
                        voc[w] = word_id
                        word_id += 1
            lid += 1
    with open(neg_file) as fp:
        for line in fp:
            if lid%5 != 0:
                ls = line.split(' ')
                for w in ls:
                    w = w.strip()
                    if len(w) < 3: continue
                    if w not in voc:
                        voc[w] = word_id
                        word_id += 1
            lid += 1
    global n_voc
    n_voc = word_id
    print 'Size of vocabulary = {}'.format(n_voc)

    # collect training data
    train_data = []
    lid = 0
    with open(pos_file) as fp:
        for line in fp:
            if lid%5 != 0:
                sentence = np.zeros((n_voc))
                ls = line.split(' ')
                for w in ls:
                    w = w.strip()
                    if len(w) < 3: continue
                    word_id = voc[w]
                    sentence[word_id] += 1
                train_data.append([sentence, [1,0]])
            lid += 1
    with open(neg_file) as fp:
        for line in fp:
            if lid%5 != 0:
                sentence = np.zeros((n_voc))
                ls = line.split(' ')
                for w in ls:
                    w = w.strip()
                    if len(w) < 3: continue
                    word_id = voc[w]
                    sentence[word_id] += 1
                train_data.append([sentence, [0,1]])
            lid += 1
    train_data = np.array(train_data)
    print 'Num of training data = {}'.format(len(train_data))

    # test data
    test_data = []
    lid = 0
    with open(pos_file) as fp:
        for line in fp:
            if lid%5 == 0:
                sentence = np.zeros((n_voc))
                ls = line.split(' ')
                try:
                    for w in ls:
                        w = w.strip()
                        if len(w) < 3: continue
                        word_id = voc[w]
                        sentence[word_id] += 1
                except KeyError:
                    pass
                else:
                    test_data.append([sentence, [1,0]])
            lid += 1
    with open(neg_file) as fp:
        for line in fp:
            if lid%5 == 0:
                sentence = np.zeros((n_voc))
                ls = line.split(' ')
                try:
                    for w in ls:
                        w = w.strip()
                        if len(w) < 3: continue
                        word_id = voc[w]
                        sentence[word_id] += 1
                except KeyError:
                    pass
                else:
                    test_data.append([sentence, [0,1]])
            lid += 1
    test_data = np.array(test_data)
    print 'Num of test data = {}'.format(len(test_data))


    return train_data, test_data


# deep neural network
def forward_propagate(train_X):
    print train_X.get_shape()

    n_input_layer = n_voc
    n_layer_1 = 200
    n_layer_2 = 50
    n_output_layer = 2

    layer_1_w_b = {'w': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
            'b': tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
            'b': tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b = {'w': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
            'b': tf.Variable(tf.random_normal([n_output_layer]))}

    layer_1 = tf.add(tf.matmul(train_X, layer_1_w_b['w']), layer_1_w_b['b'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w']), layer_2_w_b['b'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w']), layer_output_w_b['b'])

    return layer_output


def run(data, test):
    #X = tf.placeholder(tf.float32)
    #Y = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, shape = [None, n_voc])
    Y = tf.placeholder(tf.float32, shape = [None, 2])
    predict = forward_propagate(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = predict))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost_func)


    with tf.Session() as sess:
        writer = tf.summary.FileWriter('/Users/ygu/Code/1_comment_classification/tfb')
        writer.add_graph(sess.graph)

        print '------ running ------'
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            print '  echo {}'.format(epoch)
            opt, err = sess.run([optimizer, cost_func], feed_dict = {
                X: list(data[:,0]), 
                Y: list(data[:,1])
                })
            print err
        

        # training accuracy
        train_predict = tf.equal(tf.argmax(predict, axis = 1), 
                tf.argmax(tf.reshape(Y, [len(data), 2]), axis = 1))
        train_accuracy = tf.reduce_mean(tf.cast(train_predict, 'float'))
        print 'Training accuracy = {}'.format(train_accuracy.eval({
            X: list(data[:,0]),
            Y: list(data[:,1])
            }))

        # test accuracy
        test_predict = tf.equal(tf.argmax(predict, axis = 1), 
                tf.argmax(tf.reshape(Y, [len(test), 2]), axis = 1))
        test_accuracy = tf.reduce_mean(tf.cast(test_predict, 'float'))
        print 'Test accuracy = {}'.format(test_accuracy.eval({
            X: list(test[:,0]),
            Y: list(test[:,1])
            }))

        # len(dict) -> 200 -> 50 -> 2
        # training cost_func: 0.43 (after 20 epoches)
        # training accuracy: 0.989
        # test accuracy: 0.681


train_data, test_data = read_data()
#print data[:,1]; f = raw_input()
run(train_data, test_data)

