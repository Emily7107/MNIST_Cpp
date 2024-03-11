#include "read_dataset_images.hpp"
#include "eigen-3.4.0/Eigen/Dense"

MnistImageReader::MnistImageReader() {}

MnistImageReader::~MnistImageReader() {}

void MnistImageReader::readMnistImages(std::string inputFile, Eigen::MatrixXd& images) {
    std::ifstream file(inputFile, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        unsigned char label;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);

        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of images = " << number_of_images << std::endl;
        std::cout << "rows = " << n_rows << std::endl;
        std::cout << "cols = " << n_cols << std::endl;

        //make sure the true size of images matrix.
        images.resize(number_of_images,n_rows*n_cols);

        //write data into images matrix.
        for (int i = 0; i < number_of_images; i++) {
            for (int r = 0; r < n_rows; r++) {
                for (int c = 0; c < n_cols; c++) {
                    unsigned char pixel = 0;
                    file.read((char*)&pixel, sizeof(pixel));
                    images(i,r*28+c)=static_cast<double>(pixel)/255;
                }
            }
        }

        
    }
}

void MnistImageReader::writeImagesToFile(std::string filename, Eigen::MatrixXd& images, int imageIndex) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        
        outFile << 2 << "\n";
        outFile << 28 <<"\n";
        outFile << 28 <<"\n";


        for (int j = 0; j < images.cols(); j++) {
            outFile << images(imageIndex, j) << "\n";
        }
 
        outFile.close();
        std::cout << "Values written to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open " << filename << " for writing." << std::endl;
    }
}

int MnistImageReader::ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
