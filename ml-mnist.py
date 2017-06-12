from load_data import load_data
from ann_model import train_model, test_model

def main():

    #Load data
    data_train, data_test = load_data()

    #Limit number of training examples for development speed
    #data_train = data_trim(1000,data_train)

    #Define parameters of neural network
    nn_arch = (784,30,10) #layer sizes (without bias term)
    reg_param = 0 #regularisation parameter

    model = train_model(data_train, nn_arch, reg_param)
    test_model(model, data_test, nn_arch)

    #model_file = 'trained_model.npy'
    #np.save(model_file, model)
    #model = np.load(model_file)


    #for i in range(0,5000,250):
    #    show(data_train[0][i,:], data_train[1][i])


def data_trim(m,data):
    X = data[0]
    Y = data[1]

    if m < len(Y):
        X = X[0:m,:]
        Y = Y[0:m,:]

    return (X,Y)


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
