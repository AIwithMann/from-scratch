#include<Eigen/Dense>
#include<iostream>
#include<cassert>
#include<stdexcept>
class LogisticRegression{
private:
    Eigen::VectorXd W;
    double b;
    bool ridge = false;
    bool lasso = false;
    double lambdaL1 = 0.0;
    double lambdaL2 = 0.0;

public:
    LogisticRegression(int numFeatures, bool ridge_ = false, bool lasso_ = false,  double lambdaL1_ = 0.0, double lambdaL2_ = 0.0)
        : W(Eigen::VectorXd::Zero(numFeatures)),
        b(0.0f),
        ridge(ridge_),
        lasso(lasso_),
        lambdaL1(lambdaL1_),
        lambdaL2(lambdaL2_)
    {
        if((ridge || lasso) && (lambdaL1 <= 0.0 && lambdaL2 <= 0.0))
            throw std::invalid_argument("lambda must be > 0 when regualarization is enabled");
        if(!ridge && !lasso && (lambdaL1 != 0.0 || lambdaL2 != 0.0))
            throw std::invalid_argument("lambda given but no regularization enabled");
        if(ridge && lasso && (lambdaL1 <= 0.0 || lambdaL2 <= 0.0))
            throw std::invalid_argument("both lambdaL2 (ridge) and lambdaL1 (lasso) must be > 0 for elastic net");
        if(ridge && !lasso && lambdaL2 <= 0.0)
            throw std::invalid_argument("lambdaL2 must be > 0 for ridge");
        if(!ridge && lasso && lambdaL1 <= 0.0)
            throw std::invalid_argument("lambdaL1 must be > 0 for lasso");       
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const{
        assert(X.cols() == W.size());
        Eigen::VectorXd z = X * W;
        z.array() += b;
        Eigen::VectorXd exponential = (-z.array()).exp();
        Eigen::VectorXd y_hat = (1.0/(1.0+exponential.array()));
        return y_hat;
    }

    double loss(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat) const{
        assert(y.size() == y_hat.size());
        double eps = 1e-15;

        double loss = -(
             y.array() * (y_hat.array() + eps).log()
             + (1.0 - y.array()) * (1.0 - y_hat.array() + eps).log() 
             ).mean();
        
        
        if (ridge)
            loss += W.squaredNorm() * (lambdaL2/(2.0 * y.size()));
        if (lasso)
            loss += W.lpNorm<1>() * (lambdaL1/y.size());

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
            if(ridge)
                dW += (lambdaL2/m) * W;
            if(lasso)
                dW += (lambdaL1/m) * W.array().sign().matrix();
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
    LogisticRegression Model(2,true, true, 0.01,0.01);
    Model.train(X,Y,100000,0.1);

    Eigen::VectorXd Y_hat = Model.predict(X);
    double loss = Model.loss(Y, Y_hat);

    std::cout << "Loss = " << loss << "\n";
    return 0;
}

