#include <Eigen/Dense>
#include <cassert>
#include <iostream>

class LinearRegression {
private:
    Eigen::VectorXd W;
    double b;

public:
    LinearRegression(int numFeatures) {
        W = Eigen::VectorXd::Zero(numFeatures);
        b = 0.0;
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& X) {
        assert(X.cols() == W.size());
        Eigen::VectorXd y_hat = X * W;
        y_hat.array() += b;
        return y_hat;
    }

    double Loss(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat) {
        assert(y.size() == y_hat.size());
        double mse = (y_hat - y).squaredNorm() / y.size();
        return mse;
    }

    void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int num_epochs, double lr) {
        assert(X.cols() == W.size());
        assert(y.size() == X.rows());

        for (int i = 0; i < num_epochs; i++) {
            Eigen::VectorXd y_hat = predict(X);

            Eigen::VectorXd e = y_hat - y;

            Eigen::VectorXd dW = (2.0 / X.rows()) * X.transpose() * e;
            double db = (2.0 / X.rows()) * e.sum();

            W -= lr * dW;
            b -= lr * db;
        }
    }

   
};

int main(){
    Eigen::MatrixXd X(8, 2);
    X << 1, 2,
        2, 1,
        3, 0,
        0, 3,
        4, 1,
        2, 2,
        5, 0,
        0, 4;

    
    Eigen::VectorXd Y(8);
    Y << 13, 12, 11, 14, 16, 15, 15, 17;

    LinearRegression Model(2);
    Model.train(X, Y, 100000,0.1);
    
    Eigen::VectorXd Y_hat = Model.predict(X);
    double loss = Model.Loss(Y, Y_hat);

    std::cout << "Loss = " << loss << "\n";
    return 0;
}
