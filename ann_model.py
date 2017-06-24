import numpy as np
from scipy.optimize import minimize

minimise_iter = 0;

#train model 
def train_model(data, model_params):

    nn_arch = model_params['nn_arch']

    # Intialise synapses
    syn = intialise_weights(nn_arch)

    # Unroll Parameters
    initial_nn_params = collapse_syn(syn)

    global minimise_iter
    minimise_iter = 0

    #Train using minimisation function
    results = minimize(ann, 
                       initial_nn_params,
                       args=(data, model_params), 
                       method = "CG",
                       jac=True,
                       callback=callback,
                       options={'maxiter':400})

    #return split results vector
    return (nn_arch, results.x)

def test_model(model, data):

    syn = model[1]

    model_params = {'nn_arch': model[0],
                    'reg_param' : 0}

    #Extract results
    Y = data[1]

    #Number of test_model sample points
    m = len(Y)

    #Calculate hypothesis
    h = ann(syn, data, model_params, 'predict')

    #Extract highest predication as value
    p = np.argmax(h,axis=1)
    y = np.argmax(Y,axis=1)

    error = (1 - np.equal(p,y).sum() / m)

    print('Model Error:',  error * 100, "%")
    print('Digits wrong: ', int(error * m), "/", m)

    return error
    

# Display current progress
def callback(synVec):
    global minimise_iter
    minimise_iter += 1
    print('Iterations = %d\r'%minimise_iter, end="") 


#Cost function
#Feed forward network to find cost
def ann(synVec, data, model_params, mode='minimise'):

    nn_arch = model_params['nn_arch']

    #Check if reguralisation parameter given, otherwise disable
    if 'reg_param' in model_params:
        reg_param = model_params['reg_param']
    else:
        reg_param = 0


    #Split data
    X = data[0]
    Y = data[1]
    syn = split_synVec(synVec, nn_arch)

    #Number of training examples
    m = len(Y)

    #Number of layers
    nl = nn_arch.size

    # Intialise empty lists, sometimes index 0 is unused
    lay = [None]*nl
    delta = [None]*nl
    grad = [None]*nl

    #Zero Layer (also input layer)
    lay[0] = np.append(np.ones((m,1)),X, axis=1)

    ####Feed foward
    for i in range(1,nl): #skip input layer
        #Calculate layer i values
        lay[i] = activation( np.dot(lay[i-1], syn[i].T) )
        #Add bais unit
        lay[i] = np.append(np.ones((m,1)), lay[i], axis=1)

    #Hypothesis
    h = lay[nl-1][:,1:] #strip bais unit

    if mode == 'predict':
        return h

    #Calculative cost
    J = np.multiply(-Y,np.log(h)) - np.multiply(1-Y,np.log(1-h))
    J = J.sum() / m

    #Backprogate Errors
    delta[nl-1] = h - Y #output layer
    for i in reversed(range(1,nl-1)): #skip input and output layer
        delta[i] = np.multiply( np.dot(delta[i+1],syn[i+1][:,1:]),
                                activation_self_deriv(lay[i][:,1:])) #use the fact activation_deriv(firing1) = activation_self_deriv(lay2)

    # Calculate gradients
    for i in range(1, nl): # skip input layer
        grad[i] = np.dot(delta[i].T,lay[i-1]) / m

    #Add Regularisation
    if reg_param > 0:
        for i in range(1, nl): # skip input layer
            J += (syn[i][:,1:]**2).sum() * reg_param / 2 / m
            grad[i][:,1:] += grad[i][:,1:] * reg_param / m

    return ( J, collapse_syn(grad) )

#===============================================================================
# Utility Functions
#===============================================================================

#Activation function for neural network
def activation(x):
    return 1.0 / (1.0 + np.exp(-x))

#Derivative of activation function, input as activiation(x)
def activation_self_deriv(x):
    return x * (1 - x);

#Intialises weights of nn
def intialise_weights(nn_arch):


    syn = [None]*len(nn_arch) #Intialise to null array

    for i in range(1,len(syn)): #skip input layer
        N_in = nn_arch[i-1] + 1
        N_out = nn_arch[i]
        epsilon = np.sqrt(6/(N_in+N_out))

        syn.append( np.random.random((N_out,N_in)) * 2 * epsilon - epsilon )

    return syn

def split_synVec(synVec, nn_arch):
    syn = [None]*nn_arch.size #Intialise to null array

    split_start = 0
    split_end = 0

    #Split and reshape into syn matrices
    for i in range(1,len(syn)): #skip input layer
        split_end += nn_arch[i]*(nn_arch[i-1]+1)

        syn[i] = synVec[split_start:split_end].reshape(nn_arch[i],nn_arch[i-1]+1)

        split_start = split_end

    return syn

# Collapses synapses into 1d array for minimisation
def collapse_syn(syn):

    synVec = np.empty(0)

    for i in range(len(syn)):
        if syn[i] is not None:
            synVec = np.append(synVec,syn[i].reshape(-1))

    return synVec

