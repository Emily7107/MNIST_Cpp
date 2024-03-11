echo "This script should read a dataset label into a tensor and pretty-print it into a text file..."

#!/bin/bash

# Define the filename with the full path
inputFile=$1
#inputFile="mnist-datasets/t10k-labels.idx1-ubyte"

# Define the output path
outputFile=$2

# Set label index
labelIndex=$3

# Compile the C++ code with the correct Eigen header path
g++ -std=c++11 -o main main_labels.cpp read_dataset_labels.cpp 

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful."
    echo "Running the program with filename: $inputFile"
    echo "Output the data to: $outputFile"
    ./main "$inputFile" "$outputFile" "$labelIndex"
else
    echo "Compilation failed."
fi

# Clean up the compiled executable
rm main

