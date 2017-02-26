import tensorflow as tf
import matplotlib.pyplot as plt
import os.path

def binarize_images(images, threshold=0.1):
    return (threshold < images).astype('float32')    

def RNN(inputs, weights, biases, n_steps, step_size, n_hidden_units, cell):    
    inputs = tf.reshape(inputs, [-1, n_steps, step_size])    
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    first_layer = tf.nn.relu(tf.matmul(outputs[:, -1], weights['first_layer']) + biases['first_layer'])
    second_layer = tf.matmul(first_layer, weights['second_layer']) + biases['second_layer']
    return second_layer

def RNN_LSTM(inputs, weights, biases, n_steps, step_size, n_hidden_units):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, state_is_tuple=True)
    return RNN(inputs, weights, biases, n_steps, step_size, n_hidden_units, lstm_cell)

def RNN_GRU(inputs, weights, biases, n_steps, step_size, n_hidden_units):
    gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden_units)
    return RNN(inputs, weights, biases, n_steps, step_size, n_hidden_units, gru_cell)
    
def RNN_Stacked_LSTM(inputs, weights, biases, n_steps, step_size, n_hidden_units):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, state_is_tuple=True)
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 3, state_is_tuple=True)
    return RNN(inputs, weights, biases, n_steps, step_size, n_hidden_units, stacked_lstm)

def RNN_Stacked_GRU(inputs, weights, biases, n_steps, step_size, n_hidden_units):
    gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden_units)
    stacked_gru = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * 3, state_is_tuple=True)
    return RNN(inputs, weights, biases, n_steps, step_size, n_hidden_units, stacked_gru)

def load_model(session, model_dir, model_filename):
    model_path = model_dir + "/" + model_filename
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(session, model_path)
    print('Model succesfully loaded from: ', model_path)

def save_model(sess, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir + "/" + model_filename
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.save(sess, model_path)
    print('Model succesfully saved at:', model_path)

def plot_errors(training_error, test_error, title, figure_name):
    
    plt.figure()
    plt.gca().set_color_cycle(['red', 'blue'])
    plt.plot(range(len(training_error)), training_error)
    plt.plot(range(len(test_error)), test_error)
    plt.xlabel("Epoch")
    plt.ylabel("Training Error: Red; Test Error: Blue")
    plt.title(title)
    plt.grid(True)
    
    plt.savefig(figure_name)
    #plt.show()
