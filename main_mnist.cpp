#include "mnist.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include "eigen-3.4.0/Eigen/Dense"
#include "read_dataset_images.hpp"
#include "read_dataset_labels.hpp"


int main(int argc, char* argv[]) {
    if (argc != 10) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_epochs> <batch_size> <hidden_size> <learning_rate>"
                  << " <train_images> <train_labels> <test_images> <test_labels> <log_file>"
                  << std::endl;
        return 1;
    }

    // Get the attributes from the command line arguments
    int num_epochs = std::stoi(argv[1]);
    int batch_size = std::stoi(argv[2]);
    int hidden_size = std::stoi(argv[3]);
    double learning_rate = std::stod(argv[4]);
    std::string train_images = argv[5];
    std::string train_labels = argv[6];
    std::string test_images = argv[7];
    std::string test_labels = argv[8];
    std::string log_file = argv[9];

    Eigen::MatrixXd hidden1;
    Eigen::MatrixXd hidden2;
    Eigen::MatrixXd bias1;
    Eigen::MatrixXd bias2;

    // Create an MnistImageReader object
    MnistImageReader imageReader;
    MnistLabelReader labelReader;


    Eigen::MatrixXd images(1,1);
    imageReader.readMnistImages(train_images,images);
    // imageReader.readMnistImages(test_images,images);

    Eigen::MatrixXd labels(1,1);
    labelReader.readMnistLabel(train_labels, labels);
    // labelReader.readMnistLabel(test_labels, labels);

    Mnist mnist;
    std::tie(hidden1, hidden2, bias1,bias2)=mnist.training(images,labels,num_epochs,batch_size,hidden_size,learning_rate);

    Eigen::MatrixXd testImages(1,1);
    imageReader.readMnistImages(test_images,testImages);

    Eigen::MatrixXd testLabels(1,1);
    labelReader.readMnistLabel(test_labels, testLabels);
    
    mnist.testing(testImages,testLabels,hidden1,hidden2,bias1,bias2,batch_size,log_file);

    return 0;
}
