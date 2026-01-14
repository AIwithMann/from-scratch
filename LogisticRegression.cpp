#include<Eigen/Dense>
#include<iostream>
class LogisticRegression{
    private:
        Eigen::VectorXd W;
        double b;
    public:
        LogisticRegression(int numFeatures){
            W = Eigen::VectorXd::Zero(numFeatures);
            b = 0.0f;
        }

        Eigen::VectorXd predict(const Eigen::MatrixXd& X){
            assert(X.cols() == W.size());
            Eigen::VectorXd z = X * W;
            z.array() += b;
            Eigen::VectorXd exponential = (-z.array()).exp();
            Eigen::VectorXd y_hat = (1.0/(1.0+exponential.array()));
            return y_hat;
        }

        double loss(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat){
            assert(y.size() == y_hat.size());
            double eps = 1e-15;

            double loss = -(
                 y.array() * (y_hat.array() + eps).log()
                + (1.0 - y.array()) * (1.0 - y_hat.array() + eps).log() 
            ).mean();

            return loss;
        }

        void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int numEpochs, double lr){
            
            assert(X.cols() == W.size());
            assert(y.size() == X.rows());

            for(int i = 0; i < numEpochs; i++){
                Eigen::VectorXd y_hat = predict(X);
                
                Eigen::VectorXd dz = y_hat - y;
                double m = static_cast<double>(dz.size());
                Eigen::MatrixXd dW = (1/m) * (X.transpose() * dz);
                double db = (1/m) * dz.sum();
                
                W.array() -= lr * dW.array();
                b -= lr * db;
            }
        }
};


int main(){
    Eigen::MatrixXd X(6,2);
    X << 1,2,
         2,3,
         3,1,
         6,7,
         7,8,
         8,6;
    Eigen::VectorXd Y(6);
    Y << 0,0,0,1,1,1;
    LogisticRegression Model(2);
    Model.train(X,Y,100000,0.1);

    Eigen::VectorXd Y_hat = Model.predict(X);
    double loss = Model.loss(Y, Y_hat);

    std::cout << "Loss = " << loss << "\n";
    return 0;
}

