#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <string>
#include <stdexcept>

class LinearRegression {
private:
    Eigen::VectorXd W;
    double b = 0.0;
    bool ridge = false;
    bool lasso = false;
    double lambda = 0.0;

public:
    LinearRegression(int numFeatures, bool ridge_ = false, bool lasso_ = false, double lambda_ = 0.0)
        : W(Eigen::VectorXd::Zero(numFeatures)),
        ridge(ridge_),
        lasso(lasso_),
        lambda(lambda_)
    {
        if((ridge || lasso) && lambda <= 0.0)
            throw std::invalid_argument("lambda must be > 0 when regualarization is enabled");
        if(!ridge && !lasso && lambda != 0.0)
            throw std::invalid_argument("lambda given but no regularization enabled");
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& X) {
        assert(X.cols() == W.size());
        Eigen::VectorXd y_hat = X * W;
        y_hat.array() += b;
        return y_hat;
    }
 
    double loss(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat){
        assert(y.size() == y_hat.size());
        double mse = (y_hat - y).squaredNorm()/y.size();
        
        if (ridge)
            mse += W.squaredNorm() * lambda;
        if (lasso)
            mse += W.lpNorm<1>() * lambda;
        return mse;
    }

    void train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int num_epochs, double lr) {
        assert(X.cols() == W.size());
        assert(y.size() == X.rows());
        
        const int N = X.rows();
        for (int i = 0; i < num_epochs; i++) {
            Eigen::VectorXd y_hat = predict(X);

            Eigen::VectorXd e = y_hat - y;

            Eigen::VectorXd dW = (2.0 / X.rows()) * X.transpose() * e;
            double db = (2.0 / X.rows()) * e.sum();
            
            if(ridge) 
                dW += 2.0 * lambda * W;
            if(lasso)
                dW += lambda * W.array().sign().matrix();
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

    LinearRegression Model(2, false, true, 0.01);
    Model.train(X, Y, 100000,0.1);
    
    Eigen::VectorXd Y_hat = Model.predict(X);
    double loss = Model.loss(Y, Y_hat);

    std::cout << "Loss = " << loss << "\n";
    return 0;
}
