#include "mnist.hpp"
#include "eigen-3.4.0/Eigen/Dense"

Mnist::Mnist(){}
Mnist::~Mnist(){}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd,Eigen::MatrixXd, Eigen::MatrixXd> Mnist::training(Eigen::MatrixXd& images, Eigen::MatrixXd& labels,int num_epochs, int batch_size, int hidden_size,double lr){

    double loss=0;
    Eigen::MatrixXd temp;

    Eigen::MatrixXd batchImages(batch_size,images.cols());
    Eigen::MatrixXd batchLabels(batch_size,labels.cols());

    //set up hidden layers
    Eigen::MatrixXd hidden1 = hiddenLayer(784,hidden_size);
    Eigen::MatrixXd hidden2 = hiddenLayer(hidden_size,10);

    //matrix that contain resluts of every layer
    Eigen::MatrixXd h1Out(batch_size,hidden_size);
    Eigen::MatrixXd reluOut(h1Out.rows(),h1Out.cols());
    Eigen::MatrixXd h2Out(batch_size,10);
    Eigen::MatrixXd softmaxOut(h2Out.rows(),h2Out.cols());

    //set up bias
    Eigen::MatrixXd bias1=setBias(batch_size,hidden_size);
    Eigen::MatrixXd bias2=setBias(batch_size,10);
    
    for(int tern=0;tern<num_epochs;tern++){

        for(int i=0; i<images.rows()/batch_size;i++)
        {

            for(int j=0;j<batch_size;j++){
                for(int k=0;k<images.cols();k++){
                    batchImages(j,k)=images(i*batch_size+j,k);
                }

                for(int k=0;k<labels.cols();k++){
                    batchLabels(j,k)=labels(i*batch_size+j,k);
                }
            }

            //forward propagation
            // std::cout<<"forward propagation start!\n";
            h1Out=batchImages*hidden1;
            reluOut=ReLU(h1Out);
            h2Out=reluOut*hidden2;
            softmaxOut=softmax(h2Out);

            //calculate loss value
            loss=crossEntropy(softmaxOut, batchLabels);
            std::cout<<"Loss Value: "<<loss<<std::endl;
            
            //backward propagations
            Eigen::MatrixXd error1(softmaxOut.rows(),softmaxOut.cols());
            Eigen::MatrixXd error2(h1Out.rows(),h1Out.cols());
            Eigen::MatrixXd updateHidden2(hidden2.rows(),hidden2.cols());
            Eigen::MatrixXd updateHidden1(hidden1.rows(),hidden1.cols());

            error1=d_crossEntropy(softmaxOut,labels);
            error1=d_softmax(softmaxOut,error1);

            updateHidden2=reluOut.transpose()*error1;

            temp=hidden2*(error1.transpose());

            // std::cout<<"temp("<<temp.rows()<<","<<temp.cols()<<")\n";

            error2=d_ReLU(h1Out,temp.transpose());

            updateHidden1=batchImages.transpose()*error2;

            // Gradient Clipping
            double grad_clip_threshold = 1.0; // Adjust this threshold as needed
            double updateHidden2Norm = updateHidden2.norm();
            double updateHidden1Norm = updateHidden1.norm();

            if (updateHidden2Norm > grad_clip_threshold) {
                updateHidden2 *= (grad_clip_threshold / updateHidden2Norm);
            }

            if (updateHidden1Norm > grad_clip_threshold) {
                updateHidden1 *= (grad_clip_threshold / updateHidden1Norm);
            }

            hidden1 = hidden1 - lr * updateHidden1;
            hidden2 = hidden2 - lr * updateHidden2;

            std::cout<<"batch "<< i << " finish!\n";
        }
        std::cout<<"epoch "<< tern <<" finish!\n";
    }

    
    return  std::make_tuple(hidden1, hidden2, bias1, bias2);
}


void Mnist::testing(Eigen::MatrixXd& testImages,Eigen::MatrixXd& testlabels,Eigen::MatrixXd& hidden1, Eigen::MatrixXd& hidden2,Eigen::MatrixXd& bias1,Eigen::MatrixXd& bias2, int batch_size,std::string outFile )
{
    Eigen::MatrixXd batchImages(batch_size,testImages.cols());
    Eigen::MatrixXd batchLabels(batch_size,testlabels.cols());

    Eigen::MatrixXd h1Out(batch_size,hidden1.cols());
    Eigen::MatrixXd reluOut(h1Out.rows(),h1Out.cols());
    Eigen::MatrixXd h2Out(batch_size,10);
    Eigen::MatrixXd softmaxOut(h2Out.rows(),h2Out.cols());

    double loss=0.0;

    for(int i=0; i<testImages.rows()/batch_size;i++)
        {

            for(int j=0;j<batch_size;j++){
                for(int k=0;k<testImages.cols();k++){
                    batchImages(j,k)=testImages(i*batch_size+j,k);
                }

                for(int k=0;k<testlabels.cols();k++){
                    batchLabels(j,k)=testlabels(i*batch_size+j,k);
                }
            }

            h1Out=batchImages*hidden1;
            reluOut=ReLU(h1Out);
            h2Out=reluOut*hidden2;
            softmaxOut=softmax(h2Out);  

            //calculate loss value
            loss=crossEntropy(softmaxOut, batchLabels);

            softmaxOut=oneHotEncoding(softmaxOut);
            writePredictionToFile(outFile, batchLabels,softmaxOut, i, batch_size);
        }
}

void Mnist::writePredictionToFile(std::string filename,Eigen::MatrixXd& labels,Eigen::MatrixXd& predict, int i, int batch_size){
    Eigen::VectorXd inverseLabels = oneHotEncodingInverse(labels);
    Eigen::VectorXd inversePredice = oneHotEncodingInverse(predict);

    std::ofstream outFile(filename, std::ios::app);

    if (outFile.is_open()) {

        outFile << "Current batch: "<<i<<"\n";

        for (int j = 0; j < batch_size; j++) {
            outFile << " - image "<<i*batch_size+j<<": Prediction="<<inversePredice(j)<<". Label="<<inverseLabels(j)<<"\n";
        }
 
        outFile.close();
        std::cout << "Values written to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open " << filename << " for writing." << std::endl;
    }
}

Eigen::MatrixXd Mnist::oneHotEncoding(Eigen::MatrixXd& softmaxOut){
    Eigen::MatrixXd oneHotMatrix(softmaxOut.rows(), softmaxOut.cols());
    
    for (int i = 0; i < softmaxOut.rows(); ++i) {
        // Find the index of the maximum value in each row
        int maxIndex = 0;
        double maxValue = softmaxOut(i, 0);

        for (int j = 1; j < softmaxOut.cols(); ++j) {
            if (softmaxOut(i, j) > maxValue) {
                maxValue = softmaxOut(i, j);
                maxIndex = j;
            }
        }

        // Set the one-hot encoding for the maximum value to 1, others to 0
        oneHotMatrix(i, maxIndex) = 1.0;
        for (int j = 0; j < softmaxOut.cols(); ++j) {
            if (j != maxIndex) {
                oneHotMatrix(i, j) = 0.0;
            }
        }
    }

    return oneHotMatrix;
}

Eigen::VectorXd Mnist::oneHotEncodingInverse(Eigen::MatrixXd& labels){
    Eigen::VectorXd result(labels.rows());
    for(int i=0;i<labels.rows();i++){
        for(int j=0;j<labels.cols();j++){
            if(labels(i,j)==1){
                result(i)=j;
            }
        }
    }
    return result;
}


Eigen::MatrixXd Mnist::hiddenLayer(int inputSize, int outputSize){

    srand((unsigned int) time(0));
    return std::move(Eigen::MatrixXd::Random(inputSize,outputSize));
}

Eigen::MatrixXd Mnist::setBias(int rows, int cols){
   
   Eigen::MatrixXd bias(rows,cols);

   return std::move(bias.setOnes());
}

Eigen::MatrixXd Mnist::ReLU(Eigen::MatrixXd input){

    for(int i=0;i<input.rows();++i){
        for(int j=0;j<input.cols();++j){
            input(i,j)=std::max(0.0,input(i,j));
        }
    }

    return input;
}

Eigen::MatrixXd Mnist::softmax(Eigen::MatrixXd input){
    for (int i = 0; i < input.rows(); ++i) {
        double maxCoeff = input.row(i).maxCoeff();
        double exponent = 0.0;

        // calculate the sum of every colume
        for (int j = 0; j < input.cols(); ++j) {
            exponent += std::exp(input(i, j) - maxCoeff);
        }

        // calculate softmax value of every elements
        for (int j = 0; j < input.cols(); ++j) {
            input(i, j) = std::exp(input(i, j) - maxCoeff) / exponent;
        }
    }
    return input;
}

double Mnist::crossEntropy( const Eigen::MatrixXd& input, const Eigen::MatrixXd& label){
    double result=0.0;

    for(int i=0;i<input.rows();++i){
        for(int j=0;j<input.cols();++j){
            result+=(-1)*label(i,j)*std::log(input(i,j));
        }
    }

    return result/input.rows();
}

Eigen::MatrixXd Mnist::d_crossEntropy(Eigen::MatrixXd input, const Eigen::MatrixXd& label){
    for(int i=0;i<input.rows();++i){
        for(int j=0;j<input.cols();++j){
            input(i,j)=(-1)*label(i,j)/input(i,j);
        }
    }
    return input;
}

Eigen::MatrixXd Mnist::d_softmax(Eigen::MatrixXd input, Eigen::MatrixXd error){
    for (int i = 0; i < input.rows(); ++i) {
        double sumInput = 0.0;

        for (int j = 0; j < input.cols(); ++j) {
            sumInput += input(i, j)*error(i,j);
        }

        for (int j = 0; j < input.cols(); ++j) {
            error(i,j) = input(i,j)*(error(i,j)-sumInput);
        }
    }

    return error;
}

Eigen::MatrixXd Mnist::d_ReLU( const Eigen::MatrixXd input, Eigen::MatrixXd error){
    for(int i=0;i<input.rows();++i){
        for(int j=0;j<input.cols();++j){
            if(input(i,j)<=0)
                error(i,j)=0;
        }
    }

    return input;
}
