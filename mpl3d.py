"""
This script is a working implementation of fitting an rbf neural net to a given 3d dataset by computing the weights in one step
The parameter determining the quality here is the width of the rbfs which can be adjusted in the activatino function. higher values
result in a more precise representation, as all rbfs are fitted to their respective datapoints (K=N) so ideally there should be no overlap.
"""

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
x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
y = x.copy().T  # transpose
z = (np.sin(x ** 2) + np.cos(y ** 2))

x2 = alpha
y2 = beta
z2 = Cm


#Establish basis for the neural net
weights = np.ones((32, 32))
activations = np.zeros((32**2,32,32))
outputs     = np.zeros((32**2,32,32))
outputsums  = np.zeros((32**2,32,32))
x_coordinates = x[:, 0]
y_coordinates = y[0, :]
neurondist = np.zeros((32, 32, 32, 32)) # xc_index, yc_index,  xref_index, yre_index, carries distances
neuronactiv = np.zeros_like(neurondist)
outputs     = np.zeros_like(neurondist)

#------------------------------------------

weights2 = np.ones(10001)
activations2 = np.zeros((10001, 10001))
outputs2     = np.zeros((10001, 10001))
outputsums2  = np.zeros((10001, 1))
x_coordinates2 = alpha
y_coordinates2 = beta
neurondist2 = np.zeros((10001, 10001)) # xc_index, yc_index,  xref_index, yre_index, carries distances
neuronactiv2 = np.zeros_like(neurondist2)
outputs2     = np.zeros_like(neurondist2)

#------------------------------------------------------------------------------------

#compute the activations
def activation(distance):
    activation = np.exp(-2000000*(distance))
    return activation


for i, x_coordinate in enumerate(x_coordinates):
    for j, y_coordinate in enumerate(y_coordinates):
        center = [x_coordinate, y_coordinate]
        helper = (np.subtract(x, center[0])**2 + np.subtract(y, center[1])**2)
        neurondist[i,j,:,:] = helper
        neuronactiv[i,j,:,:] = activation(neurondist[i,j,:,:])
        outputs[i, j, :, :] =  weights[i,j] * neuronactiv[i, j, :, :]
f = np.sum(neuronactiv,axis=(0,1))
weights = z / f

for i, x_coordinate in enumerate(x_coordinates):
    for j, y_coordinate in enumerate(y_coordinates):
        center = [x_coordinate, y_coordinate]
        helper = (np.subtract(x, center[0])**2 + np.subtract(y, center[1])**2)
        neurondist[i,j,:,:] = helper
        neuronactiv[i,j,:,:] = activation(neurondist[i,j,:,:])
        outputs[i, j, :, :] =  weights[i,j] * neuronactiv[i, j, :, :]


final = np.sum(outputs,axis=(0,1))

# ------------------------------------------------------------------------------------------

for i, x_coordinate2 in enumerate(x_coordinates2):
    center = [x_coordinate2, y_coordinates2[i]]
    helper = (np.subtract(x2, center[0])**2 + np.subtract(y2, center[1])**2)
    neurondist2[i, :] = helper
    neuronactiv2[i, :] = activation(neurondist2[i, :])
    outputs2[i, :] =  weights2[i] * neuronactiv2[i, :]
f2 = np.sum(neuronactiv2,axis=0)
errors = 0.5*(f2-Cm)**2

testJacobian = gd.calculateJacobian(weights2, neuronactiv2, errors)
testUpdate   = gd.applyUpdate(testJacobian, weights2, errors, 0.2)


weights2 = weights2 - np.transpose(testUpdate)

for i, x_coordinate2 in enumerate(x_coordinates2):
    center = [x_coordinate2, y_coordinates2[i]]
    helper = (np.subtract(x2, center[0]) ** 2 + np.subtract(y2, center[1]) ** 2)
    neurondist2[i, :] = helper
    neuronactiv2[i, :] = activation(neurondist2[i, :])
    outputs2[i, :] = weights2[i] * neuronactiv2[i, :]
final2 = np.sum(outputs2,axis=0)



#--------------------------------------------------------------------------------


#Trim the datasets for plotting purposes!
alpha_plot = alpha[::10]
beta_plot  = beta[::10]
Cm_plot    = Cm[::10]
final2_plot= final2[::10]


# Creating figure
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')

# Creating plot
ax.plot_surface(x, y, z)
ax.scatter(x,y, final)
ax.scatter(alpha_plot,beta_plot,Cm_plot)
ax.scatter(alpha_plot,beta_plot,final2_plot)

# show plot
plt.show()