import numpy as np

class LogisticRegression:
    def __init__(self, num_features:int):
        self.w = np.zeros(shape = (num_features))
        self.b = 0
        
    def predict(self,X:np.ndarray):
        assert X.shape[1] == self.w.size

        z = X @ self.w + self.b
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def loss(self, y:np.ndarray, y_hat:np.ndarray):
        assert y.shape == y_hat.shape
        
        loss = -np.mean(y * np.log(y_hat + 1e-15) + (1-y) * np.log(1 - y_hat + 1e-15) ) 
        return loss

    def train(self, X:np.ndarray, y:np.ndarray, num_epochs:int, lr:float):
        assert X.shape[0] == y.size 

        for i in range(num_epochs):
            y_hat = self.predict(X)
            
            dz = y_hat - y
            dw = (1/dz.size) * (X.T  @ dz)
            db = (1/dz.size) * np.sum(dz)

            self.w -= lr * dw
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

model = LogisticRegression(2)
model.train(X, y, 100000, 2)
y_hat = model.predict(X)
loss = model.loss(y, y_hat)
print("loss = ", loss)
