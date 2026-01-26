import numpy as np

class LogisticRegression:
    def __init__(self,numFeatures:int, ridge_:bool = False, lasso_:bool = False, lambda_l1_:float  = 0.0, lambda_l2_ = 0.0):
        self.W = np.zeros(shape = (numFeatures), dtype = np.float64)
        self.b = 0.0
        self.ridge = ridge_
        self.lasso = lasso_
        self.lambda_l1 = lambda_l1_
        self.lambda_l2 = lambda_l2_
        if (self.ridge or self.lasso) and (self.lambda_l1 <= 0.0 and self.lambda_l2 <= 0.0):
            raise Exception("lambda must be > 0 when regularization is enabled")
        if not self.ridge and not self.lasso and (self.lambda_l1 != 0.0 and self.lambda_l2 != 0.0):
            raise Exception("lambda given but no regularization enabled");
        if(self.ridge and self.lasso and (self.lambda_l1 <= 0.0 or self.lambda_l2 <= 0.0)):
            raise Exception("both lambda_l2 (ridge) and lambda_l1 (lasso) must be > 0 for elastic net")
        if(self.ridge and not self.lasso and self.lambda_l2 <= 0.0):
            raise Exception("lambda_l2 must be > 0 for ridge")
        if(not self.ridge and self.lasso and self.lambda_l1 <= 0.0):
            raise Exception("lambda_l1 must be > 0 for lasso")

        
    def predict(self,X:np.ndarray):
        assert X.shape[1] == self.W.size

        z = X @ self.W + self.b
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def loss(self, y:np.ndarray, y_hat:np.ndarray):
        assert y.shape == y_hat.shape
        
        loss = -np.mean(y * np.log(y_hat + 1e-15) + (1-y) * np.log(1 - y_hat + 1e-15) ) 

        
        if self.ridge:
            loss += (self.lambda_l2/(2 * y.size)) * np.sum(self.W ** 2)

        if self.lasso:
            loss += (self.lambda_l1/y.size) * np.sum(np.abs(self.W))


        return loss

    def train(self, X:np.ndarray, y:np.ndarray, num_epochs:int, lr:float):
        assert X.shape[0] == y.size 
        assert X.shape[1] == self.W.size

        for i in range(num_epochs):
            y_hat = self.predict(X)
            
            dz = y_hat - y
            dW = (1/dz.size) * (X.T  @ dz)
            db = (1/dz.size) * np.sum(dz)
            
            if self.ridge:
                dW += (self.lambda_l2/dz.size) * self.W 
            if self.lasso:
                dW += (self.lambda_l1/dz.size) * np.sign(self.W)

            self.W -= lr * dW
            self.b -= lr * db

X = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [6, 7],
    [7, 8],
    [8, 6]
])
y = np.array([0,0,0,1,1,1])

model = LogisticRegression(2, True, True, 0.01, 0.01)
model.train(X, y, 100000, 2)
y_hat = model.predict(X)
loss = model.loss(y, y_hat)
print("loss = ", loss)
