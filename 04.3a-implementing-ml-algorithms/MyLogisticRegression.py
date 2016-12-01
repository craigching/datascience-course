
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(Y, X, theta):
    # Call the hyphothesis function to turn our
    # data into probabilities
    p = sigmoid(X.dot(theta))

    # Calculate the log likelihood for each observation
    loglikelihood = Y * np.log(p + 1e-24) + (1 - Y) * np.log(1 - p + 1e-24)

    # Return the sum of the negative log likelihood
    return -1 * loglikelihood.sum()

def gradient(Y, X, theta):
    # Calculate the derivative of our likelihood function
    return ((Y - sigmoid(X.dot(theta))) * X).sum(axis = 0).reshape(theta.shape)

def gradient_descent(Y, X, theta, costf, gradientf, alpha=1e-7, max_iterations=1e4, epsilon=1e-5):
    # Find the maximum likelihood using gradient descent
    prev = costf(Y, X, theta)
    diff = epsilon+1
    i = 0

    while (diff > epsilon) and (i < max_iterations):

        # Gradient descent update rule
        theta = theta + alpha * gradientf(Y, X, theta)

        temp = costf(Y, X, theta)
        diff = np.abs(temp-prev)
        prev = temp
        i += 1
        if i % 1000 == 0:
            print('iteration {}'.format(i))

    print("number of iterations: {}".format(i))

    return theta


class MyLogisticRegression:

    def __init__(self, X, y, epsilon=1e-5):
        self.y = y.reshape(y.size,1)
        self.epsilon = epsilon
        self.mean_x = X.mean(axis=0)
        self.std_x = X.std(axis=0)
        self.X = np.ones((X.shape[0],X.shape[1]+1))
        self.X[:,1:] = (X-self.mean_x)/self.std_x
        #create weights equal to zero with an intercept coefficent at index 0
        self.theta = np.zeros((X.shape[1]+1,1))

    def optimize(self, alpha = 1e-7, max_iterations = 1e4):
        self.theta = gradient_descent(self.y, self.X, self.theta, cost, gradient, alpha, max_iterations, self.epsilon)

        new_coef = self.theta.T[0]/np.hstack((1,self.std_x))
        new_coef[0] = self.theta.T[0][0]-(self.mean_x*self.theta.T[0][1:]/self.std_x).sum()
        print(-new_coef)
