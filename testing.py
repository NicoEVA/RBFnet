import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

x = np.array([0, 1, 2, 3, 4])
y = np.sin(x)
print(y)

w = np.ones((5, 1))
print(w)
centers = x
activations = np.zeros((5,5))
for i, center in enumerate(centers):
    activations[i, :] = np.exp(-0.5*(x-centers[i])**2)*w[i,:]
    #plt.plot(x, activations[i,:])
print(activations)
# trying to see if datapoints are correctly reconstructed
point_values = np.sum(activations, axis=0)
e = 0.5 * (point_values - y)**2
e2 = point_values - y
#Fucking LM algorithm for only the weights
jacobian_ow = e2 * -1 * activations

#update_1 = jacobian_ow.T * jacobian_ow #+ 0.02 * np.eye(5)
#update_2 = np.linalg.inv(update_1).dot(jacobian_ow.dot(np.atleast_2d(y).T))

print(w)
w = activations**(-1) * y
print(w)

for i, center in enumerate(centers):
    activations[i, :] = np.exp(-0.5*(x-centers[i])**2)*w[i,:]
    plt.plot(x, activations[i,:])

plt.scatter(x, y)



