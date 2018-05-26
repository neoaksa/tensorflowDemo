"""
this snippet is used for build ANN net work, train the model and test it
before you run this snippet, you should run word2vec.py first which transfer
training and testing data to one-hot code vector.
"""

import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from functools import singledispatch

lemmatizer = WordNetLemmatizer()


# setting ANN variables
with open("lexicon.pickle", 'rb') as f:
    lexicon = pickle.load(f)
input_size = len(lexicon)
output_size = 2
structure = [input_size,500,500,output_size]
x = tf.placeholder(dtype='float',shape=[None,structure[0]])
y = tf.placeholder(dtype='float')
batch_size = 100
epoch_max = 10
total_batches = int(1600000 / batch_size)

# feedforward Model
def netural_network_model(x):
    # l is output value from active function
    l = 0
    l_prev = 0
    for i in range(1,len(structure)):
        # create each layer structure
        hidden_layer = {'weight':tf.Variable(tf.random_normal([structure[i-1],structure[i]])),
                        'biases':tf.Variable(tf.random_normal([structure[i]]))}
        # input layer--> first hidden layer
        if i == 1:
            l = tf.add(tf.matmul(x, hidden_layer['weight']),hidden_layer['biases'])
            l = tf.nn.relu(l)
            l_prev = l
        # --->output layer without active function
        elif i == len(structure)-1:
            l = tf.add(tf.matmul(l_prev, hidden_layer['weight']), hidden_layer['biases'])
        # hidden layer ---> hidden layer with Relu active function
        else:
            l = tf.add(tf.matmul(l_prev, hidden_layer['weight']), hidden_layer['biases'])
            l = tf.nn.relu(l)
            l_prev = l
    return l


tf_log = 'tf.log'


def train_neural_network(x):
    # prediction, cost and gradient decrease
    prediction = netural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2]) + 1
            print('STARTING:', epoch)
        except:
            epoch = 1

        while epoch <= epoch_max:
            # continue the session data
            if epoch != 1:
                saver.restore(sess, "model.ckpt")
            epoch_loss = 1
            # load the prepared lexicon
            with open('lexicon.pickle', 'rb') as f:
                lexicon = pickle.load(f)
            # load shuffled traning dataset
            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                      y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch, '| Batch Loss:', c, )
            # save session for each epoch
            saver.save(sess, "model.ckpt")
            print('Epoch', epoch, 'completed out of', epoch_max, 'loss:', epoch_loss)
            with open(tf_log, 'a') as f:
                f.write(str(epoch) + '\n')
            epoch += 1


train_neural_network(x)

@singledispatch
def test_neural_network():
    prediction = netural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        # load the traning session for model
        try:
            saver.restore(sess, "model.ckpt")
        except Exception as e:
            print(str(e))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # be ware there is no optimizer, since we dont need backprepergate
        feature_sets = []
        labels = []
        counter = 0
        # load test dataset
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Tested', counter, 'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

test_neural_network()

@test_neural_network.register(str)
def test_neural_network(input_data):
    prediction = netural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        with open('lexicon.pickle', 'rb') as f:
            lexicon = pickle.load(f)
        # load the session
        try:
            saver.restore(sess, "model.ckpt")
        except Exception as e:
            print(str(e))
        # lemmatize the input data
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        # one hot coding
        features = np.zeros(len(lexicon))
        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1
        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
        if result[0] == 0:
            print('Positive:', input_data)
        elif result[0] == 1:
            print('Negative:', input_data)

test_neural_network("I hate you")