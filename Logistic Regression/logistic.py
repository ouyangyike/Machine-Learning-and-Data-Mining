""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    N,M = data.shape
    bias_data = np.ones((N,1))
    new_data = np.concatenate((data, bias_data), axis=1)
    y = sigmoid(np.dot(new_data, weights))

    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    ce=0
    for index in xrange(len(targets)):
        ce += targets[index]*(-np.log(y[index]))+(1-targets[index])*(-np.log(1-y[index]))
    j=0
    y = (y >= 0.5).astype(np.int)
    for i in xrange(len(y)):
        if targets[i]==y[i]:
            j+=1
    frac_correct = (j*1.0)/len(y)   
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        f=0
        for index in xrange(len(targets)):
            f += (-targets[index]*np.log(y[index])-(1-targets[index])*np.log(1-y[index]))
        N,M = data.shape
        bias_data = np.ones((N,1))
        new_data = np.concatenate((data, bias_data), axis=1).T
        df = np.zeros(((M+1),1))
        df = np.dot(new_data,y-targets)
#        for index2 in xrange(len(weights)):
#            for index3 in range(len(y)):
#                df[index2] += new_data[index2][index3]*(y[index3]-targets[index3])
            
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    y = logistic_predict(weights, data)
    f=0
    for index in xrange(len(targets)):
        f += (-targets[index]*np.log(y[index])-(1-targets[index])*np.log(1-y[index]))
    f += 0.5*hyperparameters['weight_decay']*np.dot(weights.T,weights)[0][0]
    N,M = data.shape
    bias_data = np.ones((N,1))
    new_data = np.concatenate((data, bias_data), axis=1).T
    df = np.zeros(((M+1),1))
    df = np.dot(new_data,y-targets)+hyperparameters['weight_decay']*weights
    
    return f, df
