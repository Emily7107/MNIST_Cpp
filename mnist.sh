echo "This script should trigger the training and testing of your neural network implementation..."

#!/bin/bash

# Check if the configuration file path is provided as a command-line argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <config_file_path>"
  exit 1
fi

config_file=$1

# Function to read config file
read_config() {
  grep -E "^$1" "$config_file" | awk -F'=' '{gsub(/[ \t]+/, "", $2); print $2}'
}

# Read parameters from config file
num_epochs=$(read_config "num_epochs")
batch_size=$(read_config "batch_size")
hidden_size=$(read_config "hidden_size")
learning_rate=$(read_config "learning_rate")

train_images=$(read_config "rel_path_train_images")
train_labels=$(read_config "rel_path_train_labels")
test_images=$(read_config "rel_path_test_images")
test_labels=$(read_config "rel_path_test_labels")
log_file=$(read_config "rel_path_log_file")

# Compile the program
g++ -std=c++11 -O3 -o main main_mnist.cpp mnist.cpp read_dataset_images.cpp read_dataset_labels.cpp

# Check if compilation was successful
if [ $? -eq 0 ]; then
  echo "Compilation successful."
  ./main $num_epochs $batch_size $hidden_size $learning_rate $train_images $train_labels $test_images $test_labels $log_file
else
  echo "Compilation failed."
fi

# Clean up the compiled executable
rm main
