import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
mpl.use('TkAgg')

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(x)
y = np.sin(x)



# approximating with the help of 10 neurons
neuron_centers = x
neuron_outputs = np.zeros((10, 10))
neuron_outputs2 = np.zeros((10,100))
def activation(input, center):
    output = math.exp(-(center-input)**2)
    return output

def sumNeurons(output_array, weights):
    output_vector = np.sum(output_array*weights, axis=0)
    return output_vector

def calcIdealWeights(Phi, datapoints):
    weights = np.linalg.inv(Phi).dot(np.transpose(datapoints))
    return weights

for i, center in enumerate(neuron_centers):
    for j, check in enumerate(x):
        neuron_outputs[i, j] = activation(check, center)

idealWeights = calcIdealWeights(neuron_outputs, y)

#hihger res evaluation
for i, center in enumerate(neuron_centers):
    for j, check in enumerate(np.linspace(0,10,100)):
        neuron_outputs2[i, j] = activation(check, center)

calculated_output = sumNeurons(neuron_outputs2)


print(x)
plt.plot(np.linspace(0,10,100), calculated_output)
plt.plot(x, y)
