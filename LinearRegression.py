import numpy as np

class LinearRegression:
    def __init__(self,numFeatures:int, ridge_:bool = False, lasso_:bool = False, lambda_l1_:float  = 0.0, lambda_l2_:float = 0.0):
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
        assert X[0].size == self.W.size
        y_hat = X @ self.W + self.b 
        return y_hat

    def loss(self,y:np.ndarray, y_hat:np.ndarray):
        assert y.shape == y_hat.shape
        mse  = np.sum((y_hat - y) ** 2) /y.size 
        if(self.ridge and self.lasso):
            mse += self.lambda_l1 * np.sum(np.abs(self.W)) + self.lambda_l2 * np.sum(self.W ** 2)
        elif self.ridge:
            mse += self.lambda_l2 * np.sum(self.W ** 2)
        elif self.lasso:
            mse += self.lambda_l1 * np.sum(np.abs(self.W))

        return mse

    def train(self,X:np.ndarray, y:np.ndarray, num_epochs:int, lr:float):
        assert X[0].size == self.W.size
        assert X.shape[0] == y.size 
        
        for i in range(num_epochs):
            y_hat = self.predict(X)
            e = y_hat - y 
            dW = (2.0/X.shape[0]) * X.T @ e
            db = (2.0/X.shape[0]) * e.sum()
            
            if self.ridge and self.lasso:
                dW += 2.0 * self.lambda_l2 * self.W + self.lambda_l1 * np.sign(self.W)
            elif self.ridge:
                dW += 2.0 * self.lambda_l2 * self.W 
            elif self.lasso:
                dW += self.lambda_l1 * np.sign(self.W)
            self.W -= lr * dW 
            self.b -= lr * db 



X = np.array([[1,2],[2,1],[3,0],[0,3],[4,1],[2,2],[5,0],[0,4]])
y = np.array([13,12,11,14,16,15,15,17])

model = LinearRegression(2, True, True, 0.01, 0.01)
model.train(X,y, 10000, 0.1)

y_hat = model.predict(X)
loss = model.loss(y, y_hat)
 
print("Loss = ", loss) 
