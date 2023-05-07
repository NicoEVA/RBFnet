"""This file is trying to implement a gradient descent lerning for the associated NN problem."""
import numpy as np


def calculateJacobian(outputWeights, layerOutput, errors):
    J = np.ones((len(errors), len(outputWeights)))
    for i, error in enumerate(errors):
        J[i, :] = -1 * layerOutput[i, :] * errors[i]
    return J

def applyUpdate(Jacobian, outputWeights, errors, rate):
    print(np.shape(outputWeights))
    w_updated_p1 = np.linalg.inv(Jacobian.T.dot(Jacobian) + rate*np.eye(np.shape(Jacobian.T.dot(Jacobian))[0]))
    w_updated_p2 = Jacobian.T.dot(errors.reshape(-1,1))
    w_updated_comp =  w_updated_p1.dot(w_updated_p2)
    w_updated  = np.subtract(outputWeights, w_updated_comp)
    print(np.shape(w_updated_p1))
    print(np.shape(w_updated_p2))
    print(np.shape(w_updated_comp))
    print(np.shape(w_updated))
    return w_updated_comp


