import numpy as np

from ann_class import *

# X = input data
np.random.seed(0) #Make calculations determinstic
#neuron = sigmoidNeuron()

data = LinearData()
#print(data.train[1])

model = ann_model(data)

model.add(InputLayer(4))

model.add(layer(4, sigmoidNeuron() ))
model.add(layer(1, sigmoidNeuron() ))


model.train()


print(model.pred,data.train[1])
#print(model.predict(data.test))



#model.summary()
