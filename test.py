import numpy as np
import pandas as pd
import os.path
import datetime
from pickle import dump,load


from load_data import load_data
from ann_model import train_model, test_model



def main():

    data_train, data_test = load_data()

    model = np.load('./model/model.npy')

    test_model(model, data_test)

    #for i in range(0,5000,250):
    #    show(data_train[0][i,:], data_train[1][i])


def show(X,Y):
    import matplotlib.pyplot as pyplot
    import matplotlib as mpl

    #Reshape from flatten pixel data
    sq_size = int(np.sqrt(np.size(X)))
    image = X.reshape(sq_size,sq_size)

    #Compute output
    Y = np.dot(Y,np.indices((1,10))[1].transpose()).sum()
    

    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    ax.text(2, 2, Y, bbox={'facecolor':'red', 'pad':10})
    pyplot.show()



if __name__ == "__main__":
    main()
