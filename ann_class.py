import numpy as np


class Data:
    pass




class LinearData(Data):

    def __init__(self):
        
        X = self.gen(3)
        Y = X.sum(axis=1)
        Y = Y.reshape(-1,1).clip(0,1)
        self.train = (X,Y)
        
        X = self.gen(3)
        Y = X.sum(axis=1)
        Y = Y.reshape(-1,1)
        self.test = (X,Y)

    @staticmethod
    def gen(x):
        return np.random.random_integers(-1,1,(x,4))
    
    @staticmethod
    def normalise(X):
        # Input - Feature Normalise
        X = (X - X.mean()) / X.std()


    @staticmethod
    def denormalise(data):
        pass






class Neuron:
    def activate(x):
        """ Computes the activation of input x
        """
        raise NotImplementedError

    def derivative(x):
        raise NotImplementedError



class sigmoidNeuron(Neuron):
    descriptor = "Sigmoid Activation"
    @staticmethod
    def activate(x):
        """ Returns activate for inputs x
        """
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative(x):
        """ Returns derivate at point x
        """
        return x * (1 - x);

    @classmethod
    def info(cls):
        return cls.descriptor

class layer:
    def __init__(self, units, neuron):
        self.units = units
        self.neuron = neuron
        self.values = None
        """ Values after being activated"""

    def connect(self, back_layer):
        self.back_layer = back_layer
        self.weights = self.init_weights(back_layer.units + 1,self.units)

    def info(self):
        return("Number of units: ", self.units, self.neuron.info())

    @staticmethod
    def init_weights(N_in,N_out):
        """ Intialise to a small random variables formal normal districtuion with zero mean
        """
        std = np.sqrt(2/(N_in+N_out))
        return np.random.randn(N_in,N_out) * std

def output(self):
        self.values = self.neuron.activate( np.dot(self.back_layer.output(), self.weights) )
        # Add bais unit as additional input feature
        self.values = np.append(np.ones((self.values.shape[0],1)), self.values, axis=1)
        # need to strip bais for output layer
        return self.values

    def back_prop():
            # Equation 61 http://neuralnetworksanddeeplearning.com/chap3.html
            self.grad = np.dot(self.back_layer.values.T, self.error) / self.values.shape[0]

            # Backprogigate error from current self
            self.back_layer.error = np.multiply(np.dot(self.error, layer.weights[1:,:].T),
                                    layer.neuron.derivative(layer.back_layer.values[:,1:]))

class InputLayer(layer):
    def __init__(self, units):
        self.units = units
        self.values = None
        self.back_layer = None

    def output(self):
        return self.values

    def info(self):
        return ("Input Layer - Number of units: ", self.units)
        

class ann_model:
    def __init__(self, data = None):
        self.layers = [] 
        self.data = data

    def add(self,layer):
        if len(self.layers) > 0:
            layer.connect(self.layers[-1])

        self.layers.append(layer)
    
    def summary(self):
        for layer in self.layers:
            print(layer.info())

    def predict(self, data):
        self.layers[0].values = np.append(np.ones((data[0].shape[0],1)), data[0], axis=1)
        self.pred = self.layers[-1].output()
        self.pred = self.pred[:,1:] #strip bais unit
        return self.pred

    def train(self, epochs=300):
        for i in range(epochs):
            self.predict(self.data.train)
            print("Cost: ", self.cost())
            self.back_prop()
            self.learn()

    def back_prop(self):
        """ With cross entropy cost function, and sigmoid activation, dC/dw is a
            a function of the error. First backprogate the error on X, and then
            use it to calculate the grad
        """

        # Set error for output layer (special case)
        self.layers[-1].error = self.pred - self.data.train[1]
        
        # Backprogate errors to all layers
        for layer in reversed(self.layers[1:]): # Skip first/input layer
            layer.back_prop()
            

    def learn(self):
        lr = 0.0001
        for layer in self.layers[1:]: # Skip first/input layer
            layer.weights += -lr * layer.grad

    def cost(self):
        Y = self.data.train[1]
        h = self.pred
        C = np.multiply(Y,np.log(h)) + np.multiply(1-Y,np.log(1-h))
        C = -C.sum() / Y.shape[0]
        return C
