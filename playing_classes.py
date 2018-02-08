import numpy as np
import pandas as pd
import os
import struct

class Account:
    def __init__(self, balance):
        self._balance = balance
    def __del__(self):
        pass
    def balance(self):
        return self._balance
    def deposit(self, amount):
        self._balance += amount
    def withdraw(self, amount):
        if ( self._balance >= amount):
            self._balance -= amount
            return True
        else:
            raise ValueError("Insufficient account Funds")
            return False

class OverdraftAccount(Account):
    def __init__(self, balance, credit_line):
        super(OverdraftAccount, self).__init__(balance)
        self._credit_line = credit_line
        self._credit = credit_line
    def credit(self):
        return self._credit
    def withdraw(self, amount):
        if ( self._balance >= amount):
            self._balance -= amount
            return True
        elif ( self._balance + self._credit >= amount):
            self._credit -= amount - self._balance
            self._balance = 0
            return True
        else:
            raise ValueError("Insufficient account Funds")
            return False

class Nueron:
    @staticmethod
    def activation(x):
        raise NotImplementedError

class Network:
    def cost():
        Pass
    def gradient():
        Pass


class model:
    def __init__(self, data):
        self._data = data
    def __del__(self):
        pass
    def intialise(self):
        pass

        no_layers = 2


    def train(self):
        print(self._data)
        pass
    def test(self):
        pass
    def predict(self):
        pass


class data:
    def __init__(self):
        self.train_x = None
        self.train_y = None

        self.cv_x = None
        self.cv_y = None

        self.test_x = None
        self.test_y = None

        self.norm_input = False #Has the input been normalised
        self.norm_mean = False
        self.norm_std = False

    def normalise(self):
        if !self.norm_input
            self.norm_mean = self.train_x.mean()
            self.norm_std = self.train_x.std()

            for X in [self.test
        k = (X - X.mean()) / X.std()

    def shuffle(self):
        """Reblends training and cross validation data"""
        raise NotImplementedError
    def clean(self):
        X = data[0]
        Y = data[1]
        
        # Number of training examples
        m = len(Y)

        # Input Data - Reshape to 1D feature vector
        X = X.reshape(m,-1)

        # Input - Feature Normalise
        X = (X - X.mean()) / X.std()

        # Output - Binarise the output data
        Y = binarise(Y)

        return (X,Y)

class DataMNIST(data):
    fnames_train = ('./data/train-images-idx3-ubyte','./data/train-labels-idx1-ubyte')
    fnames_test = ('./data/t10k-images-idx3-ubyte','./data/t10k-labels-idx1-ubyte')

    def __init__(self):
        (self.test_x, self.test_y) = self._binaryRead(DataMNIST.fnames_test)
        (self.training, self.training) = self._binaryRead(DataMNIST.fnames_test)

    @staticmethod
    def _binaryRead(fnames):
        """Reads binary file as defined by source"""
        fname_img = fnames[0]
        fname_lbl = fnames[1]

        #Image file format
        #  [offset] [type]          [value]          [description] 
        #  0000     32 bit integer  0x00000803(2051) magic number 
        #  0004     32 bit integer  60000            number of images 
        #  0008     32 bit integer  28               number of rows 
        #  0012     32 bit integer  28               number of columns 
        #  0016     unsigned byte   ??               pixel 
        #  0017     unsigned byte   ??               pixel 
        #  ........ 
        #  xxxx     unsigned byte   ??               pixel
        with open(fname_img, 'rb') as fimg:
            #Read Metadata in big-endian format, as 4-byte unsigned int
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            #Read pixel data
            img = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows, cols)

        #Label file format
        #  [offset] [type]          [value]          [description] 
        #  0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
        #  0004     32 bit integer  60000            number of items 
        #  0008     unsigned byte   ??               label 
        #  0009     unsigned byte   ??               label 
        #  ........ 
        #  xxxx     unsigned byte   ??               label
        with open(fname_lbl, 'rb') as flbl:
            #Read Metadata in big-endian format, as 4-byte unsigned int
            magic, num = struct.unpack(">II", flbl.read(8))
            #Read label data
            lbl = np.fromfile(flbl, dtype=np.int8)

        return (img, lbl)

class ANNmodel(model):

    def __init__(self, **kwargs):

        if "nn_arch" in kwargs: #list of layer sizes
            self.nn_arch = kwargs['nn_arch'] 
        else:
            raise ValueError("Must supply Neural Network Architecture")
            

    def intialise(self):
        self.syn = [None]*len(self.nn_arch) #Intialise to null array

        for i in range(1,len(self.syn)): #skip input layer
            N_in = self.nn_arch[i-1] + 1
            N_out = self.nn_arch[i]
            epsilon = np.sqrt(6/(N_in+N_out))

            self.syn.append( np.random.random((N_out,N_in)) * 2 * epsilon - epsilon )

if __name__=="__main__":
    dat = DataMNIST()
    print(dat.test_x)
    #mod = ANNmodel(
    #                nn_arch = [1,1,2]
    #              )
    #mod.intialise()
    #print(mod.syn)

    #print(dat.train())
    #mdl = model( ([1,1],[2,2]) )
    #mdl.train()

    #acc = OverdraftAccount(5000,1500)
    #acc.withdraw(6000)
    #print("Account Balance: ",acc.balance(), "Account Credit:", acc.credit())
