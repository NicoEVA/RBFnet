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

Cm          = []
alpha       = []
beta        = []
CmValid     = []
alphaValid  = []
betaValid   = []


with open('F16traindata_CMabV_2022.csv', newline='') as csvfile:

    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
         Cm.append(float(row[0]))
         alpha.append(float(row[1]))
         beta.append(float(row[2]))

# Creating dataset
with open("F16validationdata_CMab_2022.csv", newline='') as csvfile:
    validation = csv.reader(csvfile, delimiter=",", quotechar='|')
    for row in validation:
        CmValid.append(float(row[0]))
        alphaValid.append(float(row[1]))
        betaValid.append(float(row[2]))


# Cut down the to be processed data for testing!
alpha = alpha[::30]
beta  = beta[::30]
Cm    =  Cm[::30]


def k_means(datapoints_x, datapoints_y, k):
    datapoints_x = np.array(datapoints_x)
    datapoints_y = np.array(datapoints_y)
    previous_centers = np.zeros((2, k))
    new_centers = np.zeros_like(previous_centers)
    distances = np.zeros((k, len(datapoints_x)))
    closest = np.zeros((2, len(datapoints_x)))
    itercounter = 0
    #Select random datapoints as initial k:
    exclude = []
    for neuron in range(k):
        select = random.randint(0, len(datapoints_x))
        exclude.append(select)
        previous_centers[0, neuron] = datapoints_x[select]
        previous_centers[1, neuron] = datapoints_y[select]

    plt.scatter(previous_centers[0, :], previous_centers[1, :])
    print((previous_centers!=new_centers).all())

    while not (previous_centers==new_centers).all():
        if itercounter != 0:
            previous_centers = new_centers
        for i, point_x in enumerate(datapoints_x):
            for j in range(k):
                #Compute the distances of each datapoint to each center
                distances[j, i] = math.sqrt((point_x - previous_centers[0, j])**2 +
                                            (datapoints_y[i] - previous_centers[1, j])**2)

        #Find the closest centroid and appoint to it.
        for i, distance in enumerate(distances.T):
            min_index = np.where(distance == min(distance))[0][0]
            closest[0, i] = min_index
            closest[1, i] = min(distance)

        #Compute new centroid based on assignments,
        #this need the closest index and the x,y positions of the respective points:
        for i in range(k):
            n = np.shape(np.where(closest[0, :]==i))[1]
            if n != 0:
                average_x = np.sum(datapoints_x[np.where(closest[0, :] == i)]) / n
                average_y = np.sum(datapoints_y[np.where(closest[0, :] == i)]) / n
                new_centers[0, i] = average_x
                new_centers[1, i] = average_y
            else:
                new_centers[0, i] = previous_centers[0, i]
                new_centers[1, i] = previous_centers[1, i]

        itercounter += 1
        print(f"Iteration: {itercounter}")
        if itercounter > 3:
            print("Terminated @ 4!")
            plt.scatter(new_centers[0, :], new_centers[1, :], marker="x")
            break
        # Creating figure

        # Creating plot
        for j in range(k):
            plt.arrow(previous_centers[0, j], previous_centers[1, j],
                      new_centers[0, j] - previous_centers[0, j], new_centers[1, j] - previous_centers[1, j],
                      color="black")
        # show plot
        plt.show()
    plt.scatter(new_centers[0, :], new_centers[1, :], marker="x", color="black")
    #generate colors for k clusters and assign them to a vector
    colors = []

    for i in range(k):
        colors.append((random.random(), random.random(), random.random(), 0.5))
    colors = np.array(colors)
    plt.scatter(alpha, beta, marker='.', color=colors[np.int16(closest[0, :])])
    print("Clustered!")
    return distances, new_centers, closest, colors

class RBFnet:
    def __init__(self, name, locations, n_input, n_output, input, desired_output, validation ,validation_output):
        self.name: str  = name
        self.locations  = locations
        self.n_input    = n_input
        self.n_output   = n_output
        self.input      = input
        self.validation = validation
        self.validation_output = validation_output
        self.desired_output = desired_output
        self.outputs     = np.zeros((np.shape(input)[0], self.n_output))
        self.neurons    = np.shape(locations)[1]
        self.activations = np.zeros((self.neurons, np.shape(self.input)[1]))
        self.rbfInput    = np.zeros((self.neurons, np.shape(self.input)[1]))
        self.rbfInputRaw = np.zeros((2, self.neurons, np.shape(self.input)[1]))
        self.rbfOutput   = np.zeros_like(self.rbfInput)
        self.rbfOutputWeighed = np.zeros_like(self.rbfOutput)

        self.rbfInputV = np.zeros((self.neurons, 25))
        self.rbfInputSep = np.zeros((self.neurons, np.shape(self.input)[1], 2))
        self.rbfInputRawV = np.zeros((2, self.neurons, 25))
        self.rbfOutputV = np.zeros_like(self.rbfInputV)
        self.rbfOutputWeighedV = np.zeros_like(self.rbfOutputV)

        self.Jacobian    = np.zeros((np.shape(input)[1], self.neurons))
        self.J21         = np.zeros((np.shape(input)[1], self.neurons))
        self.J22         = np.zeros((np.shape(input)[1], self.neurons))
        self.learningRate = 0.1
        self.error_collection = []
        self.initialiseWeights()
        self.computeInitial()
        self.try1 = np.zeros((self.neurons, self.n_input))

    def initialiseWeights(self):
        self.inputWeights  = np.ones((self.neurons, self.n_input))
        self.outputWeights = np.random.random((self.neurons, self.n_output))
        self.copyOW = self.outputWeights
        self.copyIW = self.inputWeights
    def computeInitial(self):

        self.computeActivationAndOutput(notappend=False)

    def computeActivationAndOutput(self, notappend):
        for i in range(self.neurons):
            rbfInput = np.square(self.inputWeights[i, :]) * np.square(np.subtract(self.input.T, self.locations[:, i]))
            self.rbfInputSep[i, :, :] = rbfInput
            self.rbfInput[i, :] = np.sum(rbfInput, axis=1)

            # Something here might be going wrong? Cant really do input weight backprop if I only have one input!!

        for i in range(self.neurons):
            self.rbfOutput[i, :] = np.exp(-self.rbfInput[i, :])
            self.rbfOutputWeighed[i, :]  = self.outputWeights[i] * self.rbfOutput[i, :]

        self.outputs = np.sum(self.rbfOutputWeighed, axis=0)

        self.sum_sq_errors = np.sum(0.5 * np.square(Cm - self.outputs))
        if not notappend:
            self.error_collection.append(self.sum_sq_errors)


    def trainLM(self, notappend=False):

        sq_errors   = 0.5 * np.square(Cm - self.outputs)
        self.errors = np.subtract(Cm, self.outputs)

        for i, error in enumerate(self.errors):
            self.Jacobian[i, :] = error * -1 * self.rbfOutput[:, i]

            self.J21[i, :] = (error * 2 * self.input[0, i] * self.outputWeights * ((np.exp(-self.rbfInputSep[:, i, 0]**2).dot(self.rbfInputSep[:, i, 0])))).flatten()

            self.J22[i, :] = (error * 2 * self.input[1, i] * self.outputWeights * ((np.exp(-self.rbfInputSep[:, i, 1]**2).dot(self.rbfInputSep[:, i, 1])))).flatten()

            self.TotalJacobian = np.hstack((self.Jacobian, self.J21, self.J22))

        self.actualupdate = np.linalg.inv(self.TotalJacobian.T.dot(self.TotalJacobian) + self.learningRate * np.eye(3*self.neurons)).dot(self.TotalJacobian.T.dot(sq_errors))
        # print("Update Computed.")

        self.outputWeights = self.outputWeights - self.actualupdate[:self.neurons].reshape(-1, 1)
        self.inputWeights = self.inputWeights - np.array((self.actualupdate[self.neurons:2*self.neurons],
                                                          self.actualupdate[2*self.neurons:3*self.neurons])).T

        self.computeActivationAndOutput(notappend)



input = np.array([alpha, beta]).T
desired_output = np.array(Cm).reshape(-1, 1)

dis, cents, closest, colors = k_means(alpha, beta, 50)

test_net    = RBFnet("One", cents, 2, 1, np.array([alpha, beta]), Cm, np.array([alphaValid, betaValid]), CmValid)
i_count = 0


while test_net.sum_sq_errors > 1e-6:
    previous_error = test_net.sum_sq_errors
    bunet = test_net
    test_net.trainLM()
    #
    # if test_net.sum_sq_errors < previous_error*0.999:
    #     if test_net.learningRate <= 600:
    #         test_net.learningRate = 1.2*test_net.learningRate
    # else:
    #     test_net = bunet
    #     if test_net.learningRate >= 1e-4:
    #         test_net.learningRate = 0.9 * test_net.learningRate

    # if test_net.sum_sq_errors > previous_error:
    #     print("terminated due to wrong gradient, no improvement!")
    #     break

    if i_count%100 ==0:
        print(i_count)
        print(test_net.learningRate, test_net.sum_sq_errors)

    if i_count == 25 or test_net.learningRate < 1e-40:
        break
    i_count += 1


# Creating plot
fig = plt.figure(figsize=(14, 9))
ax1 = plt.axes(projection='3d')
ax1.scatter(alpha, beta, test_net.outputs)

fig2 = plt.figure(figsize=(14, 9))
ax2 = plt.axes(projection='3d')
ax2.scatter(alpha, beta, Cm)

# fig3 = plt.figure(figsize=(15,15))
# ax3 = plt.axes(projection="3d")
# ax3.axis("equal")
# ax3.set(xlim=(-0.22, 0.94), ylim=(-0.21, 0.2), zlim=(-0.12, -0.03))
# ax3.scatter(alpha[::100], beta[::100], Cm[::100], "blue")
# ax3.scatter(alpha[::100], beta[::100], test_net.outputs[::100], "green")
# ax3.bar3d(alpha[::100], beta[::100], -0.12*np.ones_like(beta[::100]), 0.015, 0.005, abs(test_net.errors[::100]))
# plt.show()

# #--------------------------- Enable Below if dataset actively cut! -----------------------------------------------###

fig3 = plt.figure(figsize=(15,15))
ax3 = plt.axes(projection="3d")
ax3.axis("equal")
ax3.set(xlim=(-0.22, 0.94), ylim=(-0.21, 0.2), zlim=(-0.12, -0.03))
ax3.scatter(alpha, beta, Cm, "blue")
ax3.scatter(alpha, beta, test_net.outputs, "green")
ax3.bar3d(alpha, beta, -0.12*np.ones_like(beta), 0.015, 0.005, abs(test_net.errors))
plt.show()
# ---------------------------------------------------------------------------------------------------------------------#

fig5 = plt.figure(figsize=(14, 9))
ax4 = plt.axes()
ax4.set_yscale('log')
ax4.plot(np.arange(len(test_net.error_collection)), test_net.error_collection, "r+")