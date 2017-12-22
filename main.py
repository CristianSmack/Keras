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

#Compile model
#loss= function to evaluate a set of weights
#optimizer= adam, optimization algorithm
#metrics= metrics used to train the model
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

# Fit the model
model.fit(input, output, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(input, output)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))