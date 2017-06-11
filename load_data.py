import os
import struct
import numpy as np

import scipy.io

def load_data_mat():
    raw_mat = scipy.io.loadmat("./data/ex4data1.mat")
    X = raw_mat.get("X")
    Y = raw_mat.get("y").flatten()

    Y = (Y) % 10 #Covert from Matlab 1-indexing
    
    # Output - Binarise the output data
    Y = binarise(Y)

    y1 = Y[:,1:]
    y2 = (Y[:,0]).reshape(-1,1)
    
    Y = np.append(y1,y2,axis=1)


    return (X,Y)

#Loads training data and test data into nested tuple
def load_data():
    path = "./data/"

    fnames_train = ('train-images-idx3-ubyte','train-labels-idx1-ubyte')
    fnames_test = ('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')

    data_train = clean(read(path, fnames_train))
    data_test = clean(read(path, fnames_test))

    return (data_train, data_test)

def clean(data):
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

#Reads binary img and label data and returns tuple
def read(path, fnames):
    fname_img = os.path.join(path, fnames[0])
    fname_lbl = os.path.join(path, fnames[1])

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

def binarise(Y):
    # Number of training examples
    m = len(Y)

    #Number of outputs
    o = Y.max() - Y.min() + 1

    # Reshape to column indice matrix
    Y = Y.reshape((-1,1))

    #Indice matrix with column index as element value
    I = np.indices((m,o))[1]


    Y = (Y == I) * 1 #Multiply by 1 to convert to int array

    return Y

if __name__ == "__main__":
    load_data_mat()
    #data_train, data_test = load_data()
