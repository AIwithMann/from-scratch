import numpy as np

class LinearRegression:
    def __init__(self,numFeatures:int):
        self.W = np.zeros(shape = (numFeatures), dtype = np.float64)
        self.b = 0.0

    def predict(self,X:np.ndarray):
        assert X[0].size == self.W.size
        y_hat = X @ self.W + self.b 
        return y_hat

    def loss(self,y:np.ndarray, y_hat:np.ndarray):
        assert y.shape == y_hat.shape
        mse  = np.sum((y_hat - y) ** 2) /y.size 
        return mse 

    def train(self,X:np.ndarray, y:np.ndarray, num_epochs:int, lr:float):
        assert X[0].size == self.W.size
        assert X.shape[0] == y.size 
        
        for i in range(num_epochs):
            y_hat = self.predict(X)
            e = y_hat - y 
            dW = (2.0/X.shape[0]) * X.T @ e
            db = (2.0/X.shape[0]) * e.sum()

            self.W -= lr * dW 
            self.b -= lr * db 



X = np.array([[1,2],[2,1],[3,0],[0,3],[4,1],[2,2],[5,0],[0,4]])
y = np.array([13,12,11,14,16,15,15,17])

model = LinearRegression(2)
model.train(X,y, 10000, 0.1)

y_hat = model.predict(X)
loss = model.loss(y, y_hat)
 
print("Loss = ", loss) 
