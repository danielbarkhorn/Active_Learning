import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from data.dataset import Dataset

def one_hot_encode(y_original):
    y_encoded = np.array(np.zeros((y_original.shape[0], 10)))

    i = 0
    for num in y_original:
        y_encoded[i][int(num)] = 1
        i = i + 1

    return y_encoded

def nn_example():
    #### My own data
    f = open("data/pickled/mnist_data.p", "rb")
    my_data = pickle.load(f)

    # from numpy import genfromtxt
    # my_data = genfromtxt('../Active_Learning/src/data/raw/mnist.csv', delimiter=',', dtype = np.float32)
    # print("Finished reading file")

    mnist_x = my_data[:100,1:]
    mnist_y = my_data[:100,:1]

    test_mnist_x = my_data[100:800,1:]
    test_mnist_y = my_data[100:800,:1]

    # Their data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Python optimisation variables
    learning_rate = 0.5
    epochs = 10
    batch_size = 100

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 784])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        # for epoch in range(epochs):

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})

        print("\nTraining complete!")
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

if __name__ == "__main__":
    # run_simple_graph()
    # run_simple_graph_multiple()
    # simple_with_tensor_board()
    nn_example()
