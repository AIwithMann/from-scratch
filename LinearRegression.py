import numpy as np

class LinearRegression:
    def __init__(self,numFeatures:int, ridge_:bool = False, lasso_:bool = False, lambda_:float  = 0.0):
        self.W = np.zeros(shape = (numFeatures), dtype = np.float64)
        self.b = 0.0
        self.ridge = ridge_
        self.lasso = lasso_
        self.lmbda = lambda_
        if (self.ridge or self.lasso) and self.lmbda <= 0.0:
            raise Exception("lambda must be > 0 when regularization is enabled")
        if(not self.ridge and not self.lasso and self.lmbda !=0.0):
            raise Exception("lambda given but no regularization enabled");
        

    def predict(self,X:np.ndarray):
        assert X[0].size == self.W.size
        y_hat = X @ self.W + self.b 
        return y_hat

    def loss(self,y:np.ndarray, y_hat:np.ndarray):
        assert y.shape == y_hat.shape
        mse  = np.sum((y_hat - y) ** 2) /y.size 
        if self.ridge:
            mse += self.lmbda * np.sqrt(self.W ** 2)
        if self.lasso:
            mse += self.lmbda * np.sum(np.abs(self.W))

        return mse

    def train(self,X:np.ndarray, y:np.ndarray, num_epochs:int, lr:float):
        assert X[0].size == self.W.size
        assert X.shape[0] == y.size 
        
        for i in range(num_epochs):
            y_hat = self.predict(X)
            e = y_hat - y 
            dW = (2.0/X.shape[0]) * X.T @ e
            db = (2.0/X.shape[0]) * e.sum()
            
            if self.ridge:
                dW += 2.0 * self.lmbda * self.W 
            if self.lasso:
                dW += self.lmbda * np.sign(self.W)
            self.W -= lr * dW 
            self.b -= lr * db 



X = np.array([[1,2],[2,1],[3,0],[0,3],[4,1],[2,2],[5,0],[0,4]])
y = np.array([13,12,11,14,16,15,15,17])

model = LinearRegression(2, False, True, 0.01)
model.train(X,y, 10000, 0.1)

y_hat = model.predict(X)
loss = model.loss(y, y_hat)
 
print("Loss = ", loss) 
