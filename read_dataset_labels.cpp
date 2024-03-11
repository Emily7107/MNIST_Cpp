#include "read_dataset_labels.hpp"
#include "eigen-3.4.0/Eigen/Dense"

MnistLabelReader::MnistLabelReader() {}

MnistLabelReader::~MnistLabelReader() {}

void MnistLabelReader::readMnistLabel(std::string inputFile, Eigen::MatrixXd& labels) {
    std::ifstream file(inputFile, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        magic_number = ReverseInt(magic_number);
        number_of_labels = ReverseInt(number_of_labels);
        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of labels = " << number_of_labels << std::endl;

        labels.resize(number_of_labels,10);

        for (int i = 0; i < number_of_labels; i++) {
            unsigned char label = 0;
            Eigen::VectorXd temp(10);
            file.read((char*)&label, sizeof(label));

            temp << oneHotEncoding(static_cast<double>(label));
            for(int j=0; j<temp.size();j++){
                labels(i,j)=temp(j);
            }
        }
        
    }
}

Eigen::VectorXd MnistLabelReader::oneHotEncoding(double label){

    Eigen::VectorXd code(10);
    code << 0,0,0,0,0,0,0,0,0,0;
    code(static_cast<int>(label))=1;

    return code;
}

void MnistLabelReader::writeLabelsToFile(std::string filename, Eigen::MatrixXd& labels, int labelIndex) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        
        outFile << 1 << "\n";
        outFile << labels.cols() <<"\n";


        for(int j=0; j<labels.cols();j++){
            outFile << labels(labelIndex,j)<<"\n";
        }

        outFile.close();
        std::cout << "Values written to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open " << filename << " for writing." << std::endl;
    }
}

int MnistLabelReader::ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}