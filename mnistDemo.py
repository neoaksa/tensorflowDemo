import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# get data from library which includes train, validation and test
# use one-hot coding for output
mnist = input_data.read_data_sets("/home/jie/tensorflow/temp/",one_hot=True)

# setting ANN variables
structure = [784,500,500,10]
x = tf.placeholder(dtype='float',shape=[None,structure[0]])
y = tf.placeholder(dtype='float')
batch_size = 100
epoch_max = 10

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

def train_model(x):
    # prediction, cost and gradient decrease
    prediction = netural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    with tf.Session() as sess:
        # create log for flow chart
        writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
        # initial all variables
        sess.run(tf.global_variables_initializer())
        # start epoch loop
        for epoch in range(epoch_max):
            epoch_loss = 0
            # mini batch
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # run session
                _, c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epoch_max, 'loss:', epoch_loss)

        # evaluation with test data
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        writer.close()

train_model(x)