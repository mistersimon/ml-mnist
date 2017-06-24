import numpy as np
import pandas as pd
import os.path
import datetime
from pickle import dump,load


from load_data import load_data
from ann_model import train_model, test_model


def CVsplit(data,split):
    np.random.seed(0) #keep determinstic
    X = data[0]
    Y = data[1]

    m = len(Y)

    y = (np.random.random(m) < split)

    data_train = (X[y,:],Y[y,:])
    data_cv = (X[~y,:],Y[~y,:])

    return (data_train,data_cv)


def randomisedSearch(data, model_search_params):

    data_train, data_cv = CVsplit(data,0.8)

    best_score = 1 #Maximum Error is 1
    best_model = [None]*2
    max_iter = 100

    file_trails = './model/trails.csv'
    file_state = './model/state.obj'
    file_model = './model/model.npy'

    # Check if state file exits
    if os.path.exists(file_state):
        with open(file_state, 'rb') as f:
            np.random.set_state(load(f))
    else:
        print('Setting Seed')
        np.random.seed(0) #Make calculations determinstic

    # Check if trails file exits
    if os.path.exists(file_trails):
        trails = pd.read_csv(file_trails,index_col=0)
    else:
        trails = pd.DataFrame()

    #Repeat until stopped
    #while True:
    for i in range(max_iter):

        model_params = model_search_params()
        print(model_params)

        model = train_model(data_train, model_params)
        score = test_model(model, data_cv)

        #Save run information to trail file
        trail = {'accuracy': score,
                 'hidden layers': model_params['nn_arch'][1],
                 'reg param': model_params['reg_param'],
                 'datetime': datetime.datetime.now()
                }
        trails = trails.append(trail,ignore_index=True)
        trails.to_csv(file_trails)


        #Save state to restart
        with open(file_state, 'wb') as f:
            dump(np.random.get_state(),f)


        if score < best_score:
            best_score = score
            best_model = model
            np.save(file_model,best_model)
        
    #print(best_parameters)
    #return best_model


def main():

    #Load data
    data_train, data_test = load_data()

    #Limit number of training examples for development speed
    m = min(10,2)
    #data_train = data_trim(1000,data_train)
    #data_train = mnist_data_reduce(m,data_train)
    #data_test = mnist_data_reduce(m,data_test)



    input_nodes = data_train[0].shape[1]
    output_nodes = data_train[1].shape[1]

    params = lambda: {'reg_param': float(np.power(10,np.random.uniform(np.log10(0.001),np.log10(1000),1))),
                      'nn_arch': np.hstack(([input_nodes],
                                            int(np.random.uniform(output_nodes,input_nodes)),
                                            [output_nodes])),
                     }


    model = randomisedSearch(data_train, params)

    test_model(model, data_test)



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


#Reduces number of numbers from 10 to m
def mnist_data_reduce(m,data):
    X = data[0]
    Y = data[1]

    y = np.argmax(Y,axis=1) < (m-1)

    
    X = X[y,:]
    Y = Y[y,:m]


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
