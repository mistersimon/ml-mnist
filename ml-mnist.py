import numpy as np
from scipy.optimize import minimize
import scipy.io

from load_data import load_data, load_data_mat, binarise

def split_synVec(synVec, nn_arch):
    #Split synVec vector
    len_syn0 = (nn_arch[0]+1)*nn_arch[1]
    len_syn1 = (nn_arch[1]+1)*nn_arch[2]
    syn0 = synVec[:len_syn0]
    syn1 = synVec[-len_syn1:]

    #reshape synVec vector
    syn0 = syn0.reshape(nn_arch[1],nn_arch[0]+1)
    syn1 = syn1.reshape(nn_arch[2],nn_arch[1]+1)

    return (syn0, syn1)

#Cost function
#Feed forward network to find cost
def fcost(synVec, X, Y, nn_arch, reg_param):
    #Number of training examples
    m = len(Y)

    syn1, syn2 = split_synVec(synVec, nn_arch)


    ####Feed foward

    #Zero Layer (also input layer)
    lay0 = np.append(np.ones((m,1)),X, axis=1)

    #First Layer
    firing1 = np.dot(lay0, syn1.T)
    lay1 = activation(firing1)
    lay1 = np.append(np.ones((m,1)), lay1, axis=1)

    #Second Layer (Output Layer)
    firing2 = np.dot(lay1,syn2.T)
    lay2 = activation(firing2)

    #Hypothesis
    h = lay2

    #Calculative cost
    J = np.multiply(-Y,np.log(h)) - np.multiply(1-Y,np.log(1-h))

    J = J.sum() / m

    if reg_param > 0:
        J += (syn1[:,1:]**2).sum() * reg_param / 2 / m
        J += (syn2[:,1:]**2).sum() * reg_param / 2 / m

    #Calculate gradient
    delta2 = h - Y
    grad2 = np.dot(delta2.T,lay1) / m

    delta1 = np.dot(delta2,syn2[:,1:])

    delta1 = np.multiply(delta1, activation_deriv(firing1))

    grad1 = np.dot(delta1.T,lay0) / m

    #Add Regularisation
    if reg_param > 0:
        grad1[:,1:] += grad1[:,1:] * reg_param / m
        grad2[:,1:] += grad2[:,1:] * reg_param / m
    
    #Collapse vector
    gradVec = np.append(grad1,grad2).reshape(-1)

    return (J, gradVec)

def intialise_weights(N_in, N_out):

    # intialise random weights
    epsilon_init = 0.08

    synapse = np.random.random((N_out,N_in))
    synapse = synapse*2*epsilon_init - epsilon_init

    return synapse

#train model with single hidden layer
def train_model(data):
    #Seed random numbers to make calculation deterministic
    np.random.seed(0)

    #Split data
    X = data[0]
    Y = data[1]

    #n number of features
    n = len(X[1,:]) 

    #number of training examples
    m = len(Y)

    #number of outputs
    o = len(Y[1,:]) 

    # Hidden layer size
    w = 25

    reg_param = 0
    nn_arch = (n,w,o)

    #raw_params = scipy.io.loadmat("./data/ex4weights.mat")
    #syn0 = raw_params.get("Theta1")
    #syn1 = raw_params.get("Theta2")

    syn0 = intialise_weights(nn_arch[0]+1,nn_arch[1])
    syn1 = intialise_weights(nn_arch[1]+1,nn_arch[2])
    
    # Unroll Parameters
    initial_nn_params = np.append(syn0,syn1).reshape(-1)

    #Train
    #print(fcost(initial_nn_params, X, Y, nn_arch, reg_param))
    results = minimize(fcost, 
                       initial_nn_params,
                       args=(X,Y, nn_arch, reg_param), 
                       method = "CG",
                       #method = "BFGS",
                       jac=True,
                       options={'maxiter':400})


    print('Model Found')

    final_nn_params = results.x

    syn1, syn2 = split_synVec(final_nn_params, nn_arch)
    return (syn1, syn2);

def test(model, data):
    X = data[0]
    Y = data[1]

    m = len(Y)

    ####Feed foward
    syn1 = model[0]
    syn2 = model[1]

    #Zero Layer (also input layer)
    lay0 = np.append(np.ones((m,1)),X, axis=1)

    #First Layer
    firing1 = np.dot(lay0, syn1.T)
    lay1 = activation(firing1)
    lay1 = np.append(np.ones((m,1)), lay1, axis=1)

    #Second Layer (Output Layer)
    firing2 = np.dot(lay1,syn2.T)
    lay2 = activation(firing2)

    #Hypothesis
    h = lay2
    p = np.argmax(h,axis=1)
    y = np.argmax(Y,axis=1)

    print(np.equal(p,y).sum()/m)
    
#=======================================================
def main():

    model_file = 'trained_model.npy'

    data_train, data_test = load_data()
    #data_train = load_data_mat()

    #Limit number of training examples for development speed
    data_train = data_trim(10000,data_train)

    model = train_model(data_train)

    #np.save(model_file, model)
    #model = np.load(model_file)

    test(model, data_test)

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


def activation(x):
    return 1.0 / (1.0 + np.exp(-x))


def activation_deriv(x):
    x = activation(x)
    return x * (1 - x);


if __name__ == "__main__":
    main()
