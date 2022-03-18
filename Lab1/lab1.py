import numpy
import matplotlib.pyplot as plt


class linear:
    def __init__(self,in_size,out_size):
        self.prev_X = np.zeros(in_size)
        self.W = np.random.uniform(0,1,(in_size,out_size))
        self.gradient = np.zeros(in_size,out_size)

    #calculate the value W*X, and store X for gradient calculation
    def forward(self,X):
        #self.prev_X = X
        self.prev_X = np.copy(X)
        return self.W*X

    #calculate 
    def backword(self,derivative):
        self.gradient = derivative*self.prev_X
        return self.gradient

    def update_wight(self,learning_rate):
        self.W -= learning_rate*self.gradient

class My_NN:
    def __init__(self,learning_rate = 0.01):
        self.weights = []
        self.learning_rate = learning_rate
    def forward(self):
        pass
    def backword(self):
        pass
    def add_linear(self,n):
        self.weights.append()
        pass


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
    if pt[0] > pt[1]:
        labels.append(0)
    else:
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy(n=11):
    inputs = []
    labels = []
    step = 1/(n-1)
    for i in range(n):
        inputs.append([step*i, step*i])
        labels.append(0)
        
        if i == int((n-1)/2):
            continue
        
        inputs.append([step*i, 1 - step*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n*2 - 1,1)

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)
