import numpy as np




class MLP :
    def __init__(self,input_size,hidden_size,output_size,x,y,batch_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Random weight initialization
        self.W1 = np.random.randn(input_size, hidden_size)   # (3×4)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size)  # (4×2)
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = 0.1
        self.X = x
        self.y = y
        self.batch_size=batch_size


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)
    def forWard(self,X_batch):
        
        


        self.z1 = X_batch.dot(self.W1) + self.b1        # (1×4)
        self.a1 = self.sigmoid(self.z1)           # (1×4)

        self.z2 = self.a1.dot(self.W2) + self.b2       # (1×2)
        self.a2 = self.sigmoid(self.z2)           # output (1×2)

        print("Output:", self.a2)
    def backWord(self, y_batch, X_batch):
        batch_size = X_batch.shape[0]

        # Output layer error
        error_output = (self.a2 - y_batch) * self.sigmoid_derivative(self.a2)

        # Hidden layer error
        error_hidden = error_output.dot(self.W2.T) * self.sigmoid_derivative(self.a1)

        # Weight updates
        self.W2 -= self.a1.T.dot(error_output) * self.learning_rate / batch_size
        self.b2 -= np.sum(error_output, axis=0, keepdims=True) * self.learning_rate / batch_size

        self.W1 -= X_batch.T.dot(error_hidden) * self.learning_rate / batch_size
        self.b1 -= np.sum(error_hidden, axis=0, keepdims=True) * self.learning_rate / batch_size

    

    def train(self):
        for i in range(0, len(self.X), self.batch_size):
            X_batch = self.X[i : i + self.batch_size]
            y_batch = self.y[i : i + self.batch_size]
            self.forWard(X_batch=X_batch)
            self.backWord(y_batch=y_batch,X_batch=X_batch)
    def predict(self,X):
         
         self.forWard(X)
         
















