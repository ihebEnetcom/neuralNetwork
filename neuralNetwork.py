import numpy as np




class MLP :
    def __init__(self,layers_size,x,y,batch_size):
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.output_size = output_size
        # # Random weight initialization
        # self.W1 = np.random.randn(input_size, hidden_size)   # (3×4)
        # self.b1 = np.zeros((1, hidden_size))

        # self.W2 = np.random.randn(hidden_size, output_size)  # (4×2)
        # self.b2 = np.zeros((1, output_size))
        self.layers_size=layers_size
        w=[]
        b=[]
        for i in range(0,len(layers_size)-1):
            w.append(np.random.randn(layers_size[i], layers_size[i+1]))
            b.append(np.zeros((1, layers_size[i+1])))
        self.w=w
        self.b=b
        self.learning_rate = 0.1
        self.X = x
        self.y = y
        self.batch_size=batch_size


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)
    def forWard(self, X_batch):
        self.a_values = [X_batch]   # a(0) = input
        self.z_values = []

        a = X_batch

        for i in range(len(self.w)):
            z = a.dot(self.w[i]) + self.b[i]
            a = self.sigmoid(z)

            self.z_values.append(z)
            self.a_values.append(a)

        return a


      

    def backWord(self, y_batch):

        m = y_batch.shape[0]   # batch size

        # Prepare gradients
        dw = [None] * len(self.w)
        db = [None] * len(self.b)

        # ------------------------
        # 1. Output layer delta
        # ------------------------

        a_L = self.a_values[-1]
        delta = (a_L - y_batch) * self.sigmoid_derivative(a_L)

        # ------------------------
        # 2. Backprop through layers
        # ------------------------

        for i in reversed(range(len(self.w))):

            a_prev = self.a_values[i]

            # Compute gradients
            dw[i] = a_prev.T.dot(delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m

            # Compute delta for next layer (unless we reached input layer)
            if i > 0:
                delta = delta.dot(self.w[i].T) * self.sigmoid_derivative(self.a_values[i])

        # ------------------------
        # 3. Update weights
        # ------------------------

        for i in range(len(self.w)):
            self.w[i] -= self.learning_rate * dw[i]
            self.b[i] -= self.learning_rate * db[i]

        

    def train(self):
        for i in range(0, len(self.X), self.batch_size):
            X_batch = self.X[i : i + self.batch_size]
            y_batch = self.y[i : i + self.batch_size]
            self.forWard(X_batch=X_batch)
            self.backWord(y_batch=y_batch,)
    def predict(self,X):
         
         return self.forWard(X)
         
















