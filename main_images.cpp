#include "read_dataset_images.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include "eigen-3.4.0/Eigen/Dense"

int main(int argc, char* argv[]) {
    // Check if the correct number of command line arguments is provided
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename>" << std::endl;
        return 1;
    }

    // Get the input filename from the command line arguments
    std::string inputFile = argv[1];
    std::string outputFile =argv[2];
    int imageIndex =atoi(argv[3]);

    // Create an MnistImageReader object
    MnistImageReader imageReader;

    Eigen::MatrixXd images(1,1);
    imageReader.readMnistImages(inputFile, images);
    imageReader.writeImagesToFile(outputFile,images,imageIndex);
    return 0;
}
