from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt


def run_logistic_regression(hyperparameters):
    # TODO specify training data
    train_inputs, train_targets = load_train()

    valid_inputs, valid_targets = load_valid()

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = 0.1*np.random.randn(M+1,1)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log posterior and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # Find the negative log posterior of validation set
        f_valid, df_valid, prediction_valid = logistic(weights, valid_inputs, valid_targets, hyperparameters)

        logging[t] = [f/N, f, frac_correct_train*100, f_valid, frac_correct_valid*100]
    return logging

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':

    plt.figure(1)
    
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.5,
                    'weight_regularization':True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 40,
                    'weight_decay': 0.001 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    plt.plot(logging[:,1],marker='+',label='weight decay=0.001')

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.5,
                    'weight_regularization':True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 40,
                    'weight_decay': 0.01 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    plt.plot(logging[:,1],marker='H',label='weight decay=0.01')

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.5,
                    'weight_regularization':True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 40,
                    'weight_decay': 0.1 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    plt.plot(logging[:,1],marker='d',label='weight decay=0.1')

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.5,
                    'weight_regularization':True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 40,
                    'weight_decay': 1.0 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    plt.plot(logging[:,1],marker='h',label='weight decay=1.0')
    
    plt.legend(loc='upper right')
    plt.title('Plot of Loss vs. Iteration Times on training set')
    plt.xlabel('Iteration Times')
    plt.ylabel('Loss of training set')

    plt.figure(2)
        # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.5,
                    'weight_regularization':True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 40,
                    'weight_decay': 0.001 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    plt.plot(logging[:,2],marker='+',label='weight decay=0.001')

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.5,
                    'weight_regularization':True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 40,
                    'weight_decay': 0.01 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    plt.plot(logging[:,2],marker='H',label='weight decay=0.01')

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.5,
                    'weight_regularization':True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 40,
                    'weight_decay': 0.1 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    plt.plot(logging[:,2],marker='d',label='weight decay=0.1')

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.5,
                    'weight_regularization':True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 40,
                    'weight_decay': 1.0 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    plt.plot(logging[:,2],marker='h',label='weight decay=1.0')
    
    plt.legend(loc='lower right')
    plt.title('Plot of Fraction Correct vs. Iteration Times on training set')
    plt.xlabel('Iteration Times')
    plt.ylabel('Fraction Correct of training set')
    
    plt.show()
 
