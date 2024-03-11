#ifndef READ_DATASET_LABEL_HPP
#define READ_DATASET_LABEL_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "eigen-3.4.0/Eigen/Dense"

class MnistLabelReader {
public:
    MnistLabelReader();
    ~MnistLabelReader();

    void readMnistLabel(std::string inputFile, Eigen::MatrixXd& labels);
    void writeLabelsToFile(std::string filename, Eigen::MatrixXd& labels, int labelIndex);

private:
    int ReverseInt(int i);
    Eigen::VectorXd oneHotEncoding(double label);

};

#endif // READ_DATASET_IMAGE_HPP