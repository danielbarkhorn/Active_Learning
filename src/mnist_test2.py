import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data

def one_hot_encode(y_original):
    y_encoded = np.array(np.zeros((y_original.shape[0], 10)))

    i = 0
    for num in y_original:
        y_encoded[i][int(num)] = 1
        i = i + 1

    return y_encoded

def nn_example():
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    my_data = pickle.load(open( "data/pickled/mnist_data.p", "rb" ))
    mnist_x = my_data[1:200,1:]
    mnist_y = my_data[1:200,:1]

    test_mnist_x = my_data[200:800,1:]
    test_mnist_y = my_data[200:800,:1]

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

    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)

    # start the session
    # with tf.Session() as sess:
    #     # initialise the variables
    #     sess.run(init_op)
    #     total_batch = int(len(mnist.train.labels) / batch_size)
    #     for epoch in range(epochs):
    #         avg_cost = 0
    #         for i in range(total_batch):
    #             batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
    #             _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
    #             avg_cost += c / total_batch
    #         print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    #         summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    #         writer.add_summary(summary, epoch)
    #
    #     print("\nTraining complete!")
    #     writer.add_graph(sess.graph)
    #     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

    with tf.Session() as sess:
        sess.run(init_op)

        accuracies = []
        for i in range(0,len(mnist_x), batch_size):
            x_batch = mnist_x[i:i+batch_size]
            y_batch = mnist_y[i:i+batch_size]

            #Maybe change y_batch so that it is (batch_size, 10) size instead of (batch_size, 1)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: x_batch, y: one_hot_encode(y_batch)})

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            value = accuracy.eval(feed_dict={x: test_mnist_x, y: one_hot_encode(test_mnist_y)})
            accuracies.append(value)

        # Calculate accuracy
        print(sess.run(accuracy, feed_dict={x: mnist_x, y: one_hot_encode(mnist_y)}))

if __name__ == "__main__":
    # run_simple_graph()
    # run_simple_graph_multiple()
    # simple_with_tensor_board()
    nn_example()
