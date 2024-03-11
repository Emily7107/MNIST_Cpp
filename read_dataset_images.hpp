#ifndef READ_DATASET_IMAGE_HPP
#define READ_DATASET_IMAGE_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "eigen-3.4.0/Eigen/Dense"


class MnistImageReader {
public:
    MnistImageReader();
    ~MnistImageReader();

    void readMnistImages(std::string inputtFile, Eigen::MatrixXd& images);
    void writeImagesToFile(std::string filename, Eigen::MatrixXd& images, int imageIndex);
private:
    int ReverseInt(int i);
};

#endif // READ_DATASET_IMAGE_HPP
