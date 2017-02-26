from Task1Helper import binarize_images, RNN_LSTM, RNN_GRU, RNN_Stacked_LSTM, RNN_Stacked_GRU, load_model, save_model, plot_errors
from tensorflow.examples.tutorials.mnist import input_data
import os.path
import tensorflow as tf
import pickle

def task_1(checkpoint_folder, checkpoint_name, figure_title, figure_path, initial_learning_rate, n_hidden_units, cell_type, epochs):
    with tf.variable_scope(checkpoint_folder):    
        mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)    
        
        n_pixels = 784
        n_steps = 784
        step_size = 1
        n_second_layer = 100
        n_classes = 10
        
        learning_rate = tf.placeholder(tf.float32, shape=[])
        x = tf.placeholder(tf.float32, [None, n_pixels])
        y = tf.placeholder(tf.float32, [None, n_classes])
        
        weights = {
            'first_layer': tf.Variable(tf.random_normal([n_hidden_units, n_second_layer], stddev=0.1)),
            'second_layer': tf.Variable(tf.random_normal([n_second_layer, n_classes], stddev=0.1))
        }
        biases = {
            'first_layer': tf.Variable(tf.constant(0.1, shape=[n_second_layer])),
            'second_layer': tf.Variable(tf.constant(0.1, shape=[n_classes]))
        }
            
        if cell_type == 'LSTM':
            y_hat = RNN_LSTM(x, weights, biases, n_steps, step_size, n_hidden_units)
        elif cell_type == 'GRU':
            y_hat = RNN_GRU(x, weights, biases, n_steps, step_size, n_hidden_units)
        elif cell_type == 'Stacked_LSTM':
            y_hat = RNN_Stacked_LSTM(x, weights, biases, n_steps, step_size, n_hidden_units)
        elif cell_type == 'Stacked_GRU':
            y_hat = RNN_Stacked_GRU(x, weights, biases, n_steps, step_size, n_hidden_units)
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        error = tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)), tf.float32))
        
        training_errors = []
        test_errors = []
        
        initialization = tf.global_variables_initializer()
        tf.logging.set_verbosity(tf.logging.ERROR)
        
        with tf.Session() as sess:
            sess.run(initialization)
            
            if os.path.isfile(checkpoint_folder + "/" + checkpoint_name):
                load_model(sess, checkpoint_folder, checkpoint_name)
            
            else:   
                print("Running...")
                learning_rate_0 = initial_learning_rate
                test_error_0 = 1
                n_reverts = 0
        
                for epoch in range(epochs):
                    for i in range(100):
                        batch_x, batch_y = mnist_data.train.next_batch(550)
                        batch_x = binarize_images(batch_x)
                        sess.run(train, feed_dict={x: batch_x, y: batch_y, learning_rate: learning_rate_0})
                        
                    training_error = sess.run(error, feed_dict={x: binarize_images(mnist_data.train.images[:9999]), y: mnist_data.train.labels[:9999]})
                    test_error = sess.run(error, feed_dict={x: binarize_images(mnist_data.test.images), y: mnist_data.test.labels})                
                    training_errors.append(training_error)
                    test_errors.append(test_error)
                    
                    print('------Epoch ' + str(epoch) + '------')            
                    print('Training error:', training_error)
                    print('Test error:', test_error)                
                    
                    if (test_error - test_error_0) > 0.15:
                        print('Test error jumped; reverting to older model')
                        load_model(sess, checkpoint_folder, checkpoint_name)
                        n_reverts = n_reverts + 1
                        
                    if n_reverts == 2:
                        learning_rate_0 = learning_rate_0 * 0.8
                        print('Reverted 2 times already; lowering learning rate to', learning_rate_0)
                        n_reverts = 0
                            
                    if test_error < test_error_0:
                        save_model(sess, checkpoint_folder, checkpoint_name)
                        test_error_0 = test_error
                
                final_test_error = sess.run(error, feed_dict={x: binarize_images(mnist_data.test.images), y: mnist_data.test.labels})
                print('Final test error is', final_test_error)
                
                plot_errors(training_errors, test_errors, figure_title, figure_path)
            
        with open(figure_path, "wb") as handle:
            pickle.dump(test_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
