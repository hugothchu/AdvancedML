import matplotlib
matplotlib.use('Agg')

from task1 import task_1

# LSTM(32)
checkpoint_folder = "Task1_LSTM_32_results"
checkpoint_name = "Task1_LSTM_32_results.ckpt"
figure_title = "Task 1 LSTM(32)"
figure_path = 'Task1_LSTM_32_errors/Task_1_LSTM_32'
initial_learning_rate = 0.0005
n_hidden_units = 32
cell_type = 'LSTM'
epochs = 100
task_1(checkpoint_folder, checkpoint_name, figure_title, figure_path, initial_learning_rate, n_hidden_units, cell_type, epochs)

# LSTM(64)
checkpoint_folder = "Task1_LSTM_64_results"
checkpoint_name = "Task1_LSTM_64_results.ckpt"
figure_title = "Task 1 LSTM(64)"
figure_path = 'Task1_LSTM_64_errors/Task_1_LSTM_64'
initial_learning_rate = 0.0005
n_hidden_units = 64
cell_type = 'LSTM'
epochs = 100
task_1(checkpoint_folder, checkpoint_name, figure_title, figure_path, initial_learning_rate, n_hidden_units, cell_type, epochs)


# LSTM(128)
checkpoint_folder = "Task1_LSTM_128_results"
checkpoint_name = "Task1_LSTM_128_results.ckpt"
figure_title = "Task 1 LSTM(128)"
figure_path = 'Task1_LSTM_128_errors/Task_1_LSTM_128'
initial_learning_rate = 0.0005
n_hidden_units = 128
cell_type = 'LSTM'
epochs = 100
task_1(checkpoint_folder, checkpoint_name, figure_title, figure_path, initial_learning_rate, n_hidden_units, cell_type, epochs)


# GRU(32)
checkpoint_folder = "Task1_GRU_32_results"
checkpoint_name = "Task1_GRU_32_results.ckpt"
figure_title = "Task 1 GRU(32)"
figure_path = 'Task1_GRU_32_errors/Task_1_GRU_32'
initial_learning_rate = 0.0005
n_hidden_units = 32
cell_type = 'GRU'
epochs = 50
task_1(checkpoint_folder, checkpoint_name, figure_title, figure_path, initial_learning_rate, n_hidden_units, cell_type, epochs)


# GRU(64)
checkpoint_folder = "Task1_GRU_64_results"
checkpoint_name = "Task1_GRU_64_results.ckpt"
figure_title = "Task 1 GRU(64)"
figure_path = 'Task1_GRU_64_errors/Task_1_GRU_64'
initial_learning_rate = 0.0005
n_hidden_units = 64
cell_type = 'GRU'
epochs = 50
task_1(checkpoint_folder, checkpoint_name, figure_title, figure_path, initial_learning_rate, n_hidden_units, cell_type, epochs)


# GRU(128)
checkpoint_folder = "Task1_GRU_128_results"
checkpoint_name = "Task1_GRU_128_results.ckpt"
figure_title = "Task 1 GRU(128)"
figure_path = 'Task1_GRU_128_errors/Task_1_GRU_128'
initial_learning_rate = 0.0002
n_hidden_units = 128
cell_type = 'GRU'
epochs = 50
task_1(checkpoint_folder, checkpoint_name, figure_title, figure_path, initial_learning_rate, n_hidden_units, cell_type, epochs)


# Stacked LSTM(32)
checkpoint_folder = "Task1_Stacked_LSTM_32_results"
checkpoint_name = "Task1_Stacked_LSTM_32_results.ckpt"
figure_title = "Task 1 Stacked LSTM(32)"
figure_path = 'Task1_Stacked_LSTM_32_errors/Task_1_Stacked_LSTM_32'
initial_learning_rate = 0.0005
n_hidden_units = 32
cell_type = 'Stacked_LSTM'
epochs = 50
task_1(checkpoint_folder, checkpoint_name, figure_title, figure_path, initial_learning_rate, n_hidden_units, cell_type, epochs)


# Stacked GRU(32)
checkpoint_folder = "Task1_Stacked_GRU_32_results"
checkpoint_name = "Task1_Stacked_GRU_32_results.ckpt"
figure_title = "Task 1 Stacked GRU(32)"
figure_path = 'Task1_Stacked_GRU_32_results/Task_1_Stacked_GRU_32'
initial_learning_rate = 0.0005
n_hidden_units = 32
cell_type = 'Stacked_GRU'
epochs = 50
task_1(checkpoint_folder, checkpoint_name, figure_title, figure_path, initial_learning_rate, n_hidden_units, cell_type, epochs)
