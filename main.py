from keras.models import Sequential
from keras.layers import Dense
import numpy

#Fix random seed
numpy.random.seed(7)

#load data from data.csv
dataset = numpy.loadtxt("data.csv", delimiter=",")

#Split into input and output variables
input = dataset[:,0:8]
output = dataset[:,8]

#Define neural network model
model = Sequential()
#Three-layer network
model.add(Dense(12, input_dim=8,activation="relu")) #12 neurons
model.add(Dense(8, activation="relu")) #8 neurons
model.add(Dense(1,activation="sigmoid"))#1 neuron