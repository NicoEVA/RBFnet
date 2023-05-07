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
import GradientDescent as gd
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

def k_means(datapoints_x, datapoints_y, k):
    datapoints_x = np.array(datapoints_x)
    datapoints_y = np.array(datapoints_y)
    previous_centers = np.zeros((2, k))
    new_centers = np.zeros_like(previous_centers)
    distances = np.zeros((k, 10001))
    closest = np.zeros((2, 10001))
    itercounter = 0
    #Select random datapoints as initial k:
    exclude = []
    for neuron in range(k):
        select = random.randint(0, 10001)
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
                average_x = np.sum(datapoints_x[np.where(closest[0, :]==i)]) / n
                average_y = np.sum(datapoints_y[np.where(closest[0, :]==i)]) / n
                new_centers[0, i] = average_x
                new_centers[1, i] = average_y
            else:
                new_centers[0, i] = previous_centers[0,i]
                new_centers[1, i] = previous_centers[1,i]


        itercounter += 1
        print(f"Iteration: {itercounter}")
        if itercounter > 3:
            print("Terminated @ 4!")
            plt.scatter(new_centers[0, :], new_centers[1, :], marker="x")
            break

        # Creating figure



        # Creating plot
        #plt.scatter(new_centers[0, :], new_centers[1, :], marker=".")
        for j in range(k):
            plt.arrow(previous_centers[0, j], previous_centers[1, j],
                      new_centers[0, j] - previous_centers[0, j], new_centers[1, j] - previous_centers[1, j],
                      color="black")
        #plt.scatter(new_centers[0, :], new_centers[1, :])
        # show plot
        plt.show()
    plt.scatter(new_centers[0, :], new_centers[1, :], marker="x", color="black")
    #generate colors for k clusters and assign them to a vector
    colors = []

    for i in range(k):
        colors.append((random.random(), random.random(), random.random(), 0.5))
    colors = np.array(colors)

    plt.scatter(alpha, beta, marker='.', color=colors[np.int16(closest[0, :])])
    return distances, new_centers, closest, colors



class RBFnet:

    def __init__(self, name, locations, n_input, n_output, input, desired_output):
        self.name: str  = name
        self.locations  = locations
        self.n_input    = n_input
        self.n_output   = n_output
        self.input      = input
        self.desired_output = desired_output
        self.outputs     = np.zeros((np.shape(input)[0], self.n_output))
        self.neurons    = np.shape(locations)[1]
        self.activations = np.zeros((self.neurons, 10001))
        self.rbfInput    = np.zeros((self.neurons, 10001))
        self.rbfOutput   = np.zeros_like(self.rbfInput)
        self.rbfOutputWeighed = np.zeros_like(self.rbfOutput)
        self.Jacobian    = np.zeros((np.shape(input)[1], self.neurons))
        self.initialiseWeights()
        self.computeInitial()

    def initialiseWeights(self):
        self.inputWeights  = np.random.random((self.neurons, self.n_input))*10
        self.outputWeights = np.random.random((self.neurons, self.n_output))
    def computeInitial(self):
        for i in range(self.neurons):
            rbfInput = np.multiply(np.square(np.subtract(self.input.T, self.locations[:,i])), np.square(self.inputWeights[i,:]))

            self.rbfInput[i,:] = np.sum(rbfInput,axis=1)
        self.computeActivationAndOutput()

    def computeActivationAndOutput(self):
        for i in range(self.neurons):
            self.rbfOutput[i, :] = np.exp(-self.rbfInput[i, :])
            self.rbfOutputWeighed[i, :]  = self.outputWeights[i] * self.rbfOutput[i, :]
        self.outputs = np.sum(self.rbfOutputWeighed, axis=0)
        #print(self.outputWeights)
        self.sum_sq_errors = np.sum(0.5 * np.square(Cm - self.outputs))
        print(self.sum_sq_errors)

    def trainLM(self):

        # Potentially still a problem with summing over the entire dataset? Weve got a mistake ther somewher....
        sq_errors = 0.5 * np.square(Cm - self.outputs)
        errors    = np.subtract(Cm, self.outputs)
        for i, error in enumerate(errors):
            self.Jacobian[i, :] = error * -1 * self.rbfOutput[:, i]

        self.update = np.linalg.inv(self.Jacobian.T.dot(self.Jacobian) + 1e-10*np.eye(self.neurons)).dot(self.Jacobian.T.dot(sq_errors)) #
        #print(self.update)
        self.outputWeights = self.outputWeights - self.update.reshape(-1, 1)
        self.computeActivationAndOutput()

def create_net(locations):
    """This function is intended to create the network structure based on the initial locations computed by
    e.g. k-means"""
    centers = locations


input = np.array([alpha, beta]).T
desired_output = np.array(Cm).reshape(-1, 1)





dis, cents, closest, colors = k_means(alpha, beta, 100)






test_net = RBFnet("One", cents, 2, 1, np.array([alpha, beta]), Cm)

i_count = 0

while test_net.sum_sq_errors > 1e-4:
    test_net.trainLM()
    if i_count == 100:
        break
    i_count += 1

# Creating figure



# Creating plot
fig = plt.figure(figsize=(14, 9))
ax1 = plt.axes(projection='3d')
ax1.scatter(alpha, beta, test_net.outputs)

fig2 = plt.figure(figsize=(14, 9))
ax2 = plt.axes(projection='3d')
ax2.scatter(alpha, beta, Cm)
#ax2.scatter(cents[0,:], cents[1,:])

fig3 = plt.figure(figsize=(15,15))
ax3 = plt.axes(projection="3d")
ax3.scatter(alpha[::100], beta[::100], Cm[::100], "blue")
ax3.scatter(alpha[::100], beta[::100], test_net.outputs[::100], "green")
# show plot
plt.show()

#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#fig.suptitle('Sharing x per column, y per row')
#ax1 = plt.axes(projection='3d')
#ax1.plot(alpha, beta, Cm)
#ax2 = plt.axes(projection='3d')
#ax2.plot(alpha, beta, test_net.outputs)
#ax3 = plt.axes(projection='3d')
#ax3.plot(alpha[::100], beta[::100], Cm[::100])
#ax3.plot(alpha[::100], beta[::100], test_net.outputs[::100])
#ax4 = plt.axes(projection='3d')
#ax4.plot(x, -y**2, 'tab:red')

#for ax in fig.get_axes():
#    ax.label_outer()