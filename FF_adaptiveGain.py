"""
This script is a working implementation of fitting an rbf neural net to a given 3d dataset by computing the weights in one step
The parameter determining the quality here is the width of the rbfs which can be adjusted in the activatino function. higher values
result in a more precise representation, as all rbfs are fitted to their respective datapoints (K=N) so ideally there should be no overlap.
"""
import random

from mpl_toolkits import mplot3d
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
mpl.use('TkAgg')

Cm   = []
alpha= []
beta = []

with open('F16traindata_CMabV_2022.csv', newline='') as csvfile:

    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
         Cm.append(float(row[0]))
         alpha.append(float(row[1]))
         beta.append(float(row[2]))

# Creating dataset
class FFnet:
    def __init__(self, name, number, n_input, n_output, input, desired_output):
        self.name: str  = name
        self.n_input    = n_input
        self.n_output   = n_output
        self.input      = input
        self.desired_output = desired_output
        self.outputs     = np.zeros((np.shape(input)[0], self.n_output))
        self.neurons    = number
        self.activations = np.zeros((self.neurons, 10001))
        self.ffInput    = np.zeros((self.neurons, 10001))
        self.ffInputRaw = np.zeros((2, self.neurons, 10001))
        self.ffOutput   = np.zeros_like(self.ffInput)
        self.ffOutputWeighed = np.zeros_like(self.ffOutput)
        self.Jacobian    = np.zeros((np.shape(input)[1], self.neurons))
        self.Jacobian2   = np.zeros((np.shape(input)[1], self.neurons, 2))
        self.J21 = np.zeros((np.shape(input)[1], self.neurons))
        self.J22 = np.zeros((np.shape(input)[1], self.neurons))
        self.learningRate = 1e-6
        self.error_collection = []
        self.initialiseWeights()
        self.computeActivationAndOutput()
        self.helper1 = np.zeros((30,1))


    def initialiseWeights(self):
        self.inputWeights  = np.random.random((self.neurons, self.n_input))*6
        self.outputWeights = np.random.random((self.neurons, self.n_output))

    def computeActivationAndOutput(self):
        for i in range(self.neurons):
            ffInput = np.square(self.inputWeights[i, :]) * self.input.T

            self.ffInput[i, :] = np.sum(ffInput, axis=1)


        for i in range(self.neurons):
            self.ffOutput[i, :] = (2 / (1 + np.exp(-2*self.ffInput[i, :]))) - 1
            self.ffOutputWeighed[i, :] = self.outputWeights[i] * self.ffOutput[:, i]
        self.outputs = np.sum(self.ffOutputWeighed, axis=0)
        # print(self.outputWeights)
        self.sum_sq_errors = np.sum(0.5 * np.square(Cm - self.outputs))
        self.error_collection.append(self.sum_sq_errors)



    def trainLM(self):
        pass
        # # Potentially still a problem with summing over the entire dataset? Weve got a mistake ther somewher....
        sq_errors = 0.5 * np.square(Cm - self.outputs)
        self.errors    = np.subtract(Cm, self.outputs)
        for i, error in enumerate(self.errors):
            self.Jacobian[i, :] = error * -1 * self.ffOutput[:, i]
            self.Jacobian2[i, :, :] = error * -1 * self.outputWeights * self.input[:, i] * 4 * np.exp(-2*self.input[:, i])/(1+np.exp(-2*self.input[:, i]))**2
        # THere is an error here. To get the error gradients wrt the different weights, they should be summed across the entire dataset
        # self.helper1 = np.array(np.sum(self.Jacobian, axis=0)[..., np.newaxis])
        # self.helper2 = np.sum(self.Jacobian2, axis=0)
        self.update  = np.linalg.inv(self.Jacobian.T.dot(self.Jacobian) + self.learningRate*np.eye(self.neurons)).dot(self.Jacobian.T.dot(sq_errors)) #
        # self.update21 = np.linalg.inv(self.Jacobian2[:,:,0].T.dot(self.Jacobian2[:,:,0]) + self.learningRate*np.eye(self.neurons)).dot(self.Jacobian2[:,:,0].T.dot(sq_errors)) #
        # self.update22 = np.linalg.inv(
        #     self.Jacobian2[:,:,1].T.dot(self.Jacobian2[:,:,1]) + self.learningRate * np.eye(self.neurons)).dot(
        #     self.Jacobian2[:,:,1].T.dot(sq_errors))  #
        self.outputWeights = self.outputWeights - self.learningRate * self.update
        # self.inputWeights  = self.inputWeights - np.array([self.update21, self.update22]).T
        self.computeActivationAndOutput()


input = np.array([alpha, beta]).T
desired_output = np.array(Cm).reshape(-1, 1)


self = FFnet("self", 30, 2, 1, np.array([alpha, beta]), Cm)
# self_lr = FFnet("two", 1, 2, 1, np.array([alpha, beta]), Cm)
# self_hr = FFnet("tre", 1, 2, 1, np.array([alpha, beta]), Cm)
i_count = 0

# self_lr.learningRate = 0.8
# self_hr.learningRate = 500

while self.sum_sq_errors > 1e-6:
    previous_error = self.sum_sq_errors
    self.trainLM()
    # self_lr.trainLM()
    # self_hr.trainLM()
    if self.sum_sq_errors < previous_error:
        self.learningRate = 1.2*self.learningRate
        print(f"Iteration:  [{i_count+1}]\n")
        print(f"Summed MSE: [{self.sum_sq_errors}]\n")
        print(f"Current LR: [{self.learningRate}]\n\n")
    else:
        self.learningRate = self.learningRate*0.7
    if i_count == 500 or self.learningRate < 1e-40:
        break
    i_count += 1

# Creating figure

# Creating plot
fig = plt.figure(figsize=(14, 9))
ax1 = plt.axes(projection='3d')
ax1.scatter(alpha, beta, self.outputs)

fig2 = plt.figure(figsize=(14, 9))
ax2 = plt.axes(projection='3d')
ax2.scatter(alpha, beta, Cm)
#ax2.scatter(cents[0,:], cents[1,:])

fig3 = plt.figure(figsize=(15,15))
ax3 = plt.axes(projection="3d")
ax3.axis("equal")
ax3.set(xlim=(-0.22, 0.94), ylim=(-0.21, 0.2), zlim=(-0.12, -0.03))
ax3.scatter(alpha[::100], beta[::100], Cm[::100], "blue")
ax3.scatter(alpha[::100], beta[::100], self.outputs[::100], "green")
# ax3.bar3d(alpha[::100], beta[::100], -0.12*np.ones_like(beta[::100]), 0.015, 0.005, abs(self.error_collection[::100]))
# show plot
plt.show()


# fig5 = plt.figure(figsize=(14, 9))
# ax4 = plt.axes()
# ax4.set_yscale('log')
# ax4.plot(np.arange(len(self.error_collection)), self.error_collection)
# ax4.plot(np.arange(len(self_lr.error_collection)), self_lr.error_collection)
# ax4.plot(np.arange(len(self_hr.error_collection)), self_hr.error_collection)

#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#fig.suptitle('Sharing x per column, y per row')
#ax1 = plt.axes(projection='3d')
#ax1.plot(alpha, beta, Cm)
#ax2 = plt.axes(projection='3d')
#ax2.plot(alpha, beta, self.outputs)
#ax3 = plt.axes(projection='3d')
#ax3.plot(alpha[::100], beta[::100], Cm[::100])
#ax3.plot(alpha[::100], beta[::100], self.outputs[::100])
#ax4 = plt.axes(projection='3d')
#ax4.plot(x, -y**2, 'tab:red')

#for ax in fig.get_axes():
#    ax.label_outer()