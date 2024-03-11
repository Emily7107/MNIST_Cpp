#ifndef MINST_HPP
#define MINST_HPP

#include <iostream>
#include <fstream>
#include <string>
#include "eigen-3.4.0/Eigen/Dense"

class Mnist {
public:
    Mnist();
    ~Mnist();

	std::tuple<Eigen::MatrixXd, Eigen::MatrixXd,Eigen::MatrixXd, Eigen::MatrixXd> training(Eigen::MatrixXd& data, Eigen::MatrixXd& label, int ,int ,int, double);
    void testing(Eigen::MatrixXd& testImages,Eigen::MatrixXd& testLabels, Eigen::MatrixXd& hidden1, Eigen::MatrixXd& hidden2,Eigen::MatrixXd& bias1,Eigen::MatrixXd& bias2, int batch_size, std::string );
    void writePredictionToFile(std::string filename,Eigen::MatrixXd& labels,Eigen::MatrixXd& predict, int imageIndex, int batch_size);

private:
    //functions for forward propagation
    Eigen::MatrixXd hiddenLayer(int inputSize, int outputSize);
	Eigen::MatrixXd setBias(int rows, int cols);
    Eigen::MatrixXd ReLU(Eigen::MatrixXd input);
	Eigen::MatrixXd softmax(Eigen::MatrixXd input);
	double crossEntropy(const Eigen::MatrixXd& input, const Eigen::MatrixXd& label);

    //functions for backward propagation
    Eigen::MatrixXd d_crossEntropy(Eigen::MatrixXd input, const Eigen::MatrixXd& label);
    Eigen::MatrixXd d_softmax(Eigen::MatrixXd input, Eigen::MatrixXd error);
    Eigen::MatrixXd d_ReLU(const Eigen::MatrixXd input, Eigen::MatrixXd error);
    Eigen::MatrixXd oneHotEncoding(Eigen::MatrixXd& softmaxOut);
    Eigen::VectorXd oneHotEncodingInverse(Eigen::MatrixXd& labels);

};

#endif // MNIST_HPP
