"""
    This program implements linear regression with multiple variables.
    By: Rahul Golhar
"""
import numpy
import matplotlib.pyplot as plt
from ipython_genutils.py3compat import xrange
from numpy.linalg import inv


def h(theta, X):
    """
        This function returns the hypothesis value. (x*theta)
    :param theta:   the theta vector
    :param X:       the X matrix
    :return:        hypothesis value vector
    """
    return numpy.dot(X, theta)


def costFunction(theta, X, y):
    """
        This function returns the values calculated by the
        cost function.
    :param theta:   theta vector to be used
    :param X:       X matrix
    :param y:       y Vector
    :return:        Cost Function results
    """
    # note to self: *.shape is (rows, columns)
    m = len(y)
    return float((1. / (2 * m)) * numpy.dot((h(theta, X) - y).T, (h(theta, X) - y)))


def gradientDescent(X, y, alpha, iterations, initialTheta=numpy.zeros(2)):
    """
            This function minimizes the gradient descent.
    :param X:                   X matrix
    :param y:                   y vector
    :param alpha:               value of learning rate
    :param iterations:          no of iterations to run
    :param initialTheta:        initial values of theta vector to start with
    :return: theta, calculatedTheta, jTheta
    """
    # initialize theta values with initial theta vector
    theta = initialTheta
    # values of cost functions will be stored here
    jTheta = []
    # calculated Theta values will be stored here
    calculatedTheta = []
    # no of training examples used
    m = y.size

    print("\n\tMinimizing the cost function and finding the optimal theta values.")
    for itr in xrange(iterations):
        thetaTemp = theta

        jTheta.append(costFunction(theta, X, y))

        calculatedTheta.append(list(theta[:, 0]))

        # Update values of theta simultaneously
        for j in xrange(len(thetaTemp)):
            thetaTemp[j] = theta[j] - (alpha / m) * numpy.sum((h(theta, X) - y) * numpy.array(X[:, j]).reshape(m, 1))
        theta = thetaTemp

    print("\tDone with minimizing of the cost function and finding the optimal theta values.")

    return theta, calculatedTheta, jTheta


def plotInitialData(X):
    """
        This function plots the initial data and saves it.
    :param X: X matrix
    :return: None
    """
    print("\n\tPlotting the initial data.")
    plt.grid(True)
    plt.xlim([-100, 5000])
    colorDescription = plt.hist(X[:, 0], label='col1')
    colorDescription = plt.hist(X[:, 1], label='col2')
    colorDescription = plt.hist(X[:, 2], label='col3')
    plt.title('Initial data before normalization.')
    plt.xlabel('Column Value')
    plt.ylabel('Counts')
    colorDescription = plt.legend()
    plt.savefig("initialDataPlot.jpg")
    print("\tSaved the initial data plotted to initialDataPlot.jpg.")


def plotHypothesis(X,y,theta):
    """
        This function plots the hypothesis line
        using X, y and the calculated theta.
    :param X:       X matrix
    :param y:       y vector
    :param theta:   theta vector
    :return:        None
    """
    print("\n\tPlotting the hypothesis.")
    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 1], y[:, 0], 'rx', markersize=10, label='Training Data')
    plt.plot(X[:, 1], predictValue(X[:, 1], theta), 'b-', label='Hypothesis: h(x) = %0.2f + %0.2fx' % (theta[0], theta[1]))
    plt.grid(True)
    plt.title("Graph with hypothesis for the data")
    plt.ylabel('Price of the house')
    plt.xlabel('Parameters')
    plt.savefig("hypothesis.jpg")
    print("\tSaved the hypothesis plotted to hypothesis.jpg.")


def predictValue(X, theta):
    """
        This function returns the predicted value
        using theta values calculated.
    :param X:    X vector
    :param theta:   theta vector
    :return:        predicted value
    """
    return theta[0] + theta[1] * X


def normalizeData(X):
    print("\n\tNormalizing the data.")
    meansOfFeatures, standardDeviationsOfFeatures = [], []
    normalizedX = X.copy()
    for ithFeature in xrange(normalizedX.shape[1]):
        meansOfFeatures.append(numpy.mean(normalizedX[:, ithFeature]))
        standardDeviationsOfFeatures.append(numpy.std(normalizedX[:, ithFeature]))
        # Skip for the first column
        if not ithFeature:
            continue
        # store mean and standard deviations
        normalizedX[:, ithFeature] = (normalizedX[:, ithFeature] - meansOfFeatures[-1]) / standardDeviationsOfFeatures[-1]

    print("\tDone with normalization of data.")

    return normalizedX, meansOfFeatures, standardDeviationsOfFeatures


def plotNormalizedData(normalizedX):
    """
        This function plots the normalized data.
    :param normalizedX:     normalized features data
    :return: None
    """
    print("\n\tPlotting the Normalized data.")
    plt.grid(True)
    plt.xlim([-5, 5])
    dummy = plt.hist(normalizedX[:, 0], label='col1')
    dummy = plt.hist(normalizedX[:, 1], label='col2')
    dummy = plt.hist(normalizedX[:, 2], label='col3')
    plt.title('Data after feature normalization')
    plt.xlabel('Column Value')
    plt.ylabel('Counts')
    dummy = plt.legend()
    plt.savefig("normalizedData.jpg")
    print("\tSaved the normalized data plot to normalizedData.jpg.")


def normalEquationPredict(X, y):
    """
        This function implements the normal equation to linear regression.
    :param X:   X matrix
    :param y:   y vector
    :return:    normal value
    """
    return numpy.dot(numpy.dot(inv(numpy.dot(X.T, X)), X.T), y)


def plotConvergence(jTheta,iterations):
    """
        This function plots the convergence graph.
    :param jTheta:  the minimized cost function value
    :return:        None
    """

    print("\n\tPlotting the convergence.")
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(jTheta)), jTheta, 'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    dummy = plt.xlim([-0.05 * iterations, 1.05 * iterations])
    plt.savefig("convergenceOfCostFunction.jpg")
    print("\tSaved the convergence graph to convergenceOfCostFunction.jpg.")


def main():
    """
        This is the main function.
    :return: None
    """
    # Read the data
    data = numpy.loadtxt('data/housingPrices.txt', delimiter=',', usecols=(0, 1, 2), unpack=True)

    print("******************* Starting execution **********************")

    print("\nSuccessfully read the data.")

    # ***************************************** Step 1: Initial data *****************************************
    print("Getting the data ready.")

    # X matrix
    X = numpy.transpose(numpy.array(data[:-1]))
    # y vector
    y = numpy.transpose(numpy.array(data[-1:]))
    # no of training examples
    m = y.size
    # Insert a column of 1's into the X matrix
    X = numpy.insert(X, 0, 1, axis=1)
    # plot initial data on screen
    plotInitialData(X)


    # ***************************************** Step 2: Normalized the data *****************************************

    # normalize the data
    normalizedX, meansOfFeatures, standardDeviationsOfFeatures = normalizeData(X)

    # plot normalized data
    plotNormalizedData(normalizedX)

    # ******************* Step 3: calculate the cost function values***********************

    # set number of iterations
    iterations = 1500
    # set the learning rate
    alpha = 0.01

    # run gradient descent algorithm to find best theta values
    initial_theta = numpy.zeros((normalizedX.shape[1], 1))
    theta, calculatedTheta, jTheta = gradientDescent(normalizedX, y, alpha, iterations, initial_theta)

    # Plot the Convergence graph
    plotConvergence(jTheta,iterations)

    # Plot the hypothesis line
    plotHypothesis(normalizedX, y, theta)

    # ******************* Step 4: test the calculated theta values ***********************

    testSet = numpy.array([1650., 3.])

    # normalize the testSet
    normalizeTestSet = [(testSet[x] - meansOfFeatures[x + 1]) / standardDeviationsOfFeatures[x + 1] for x in xrange(len(testSet))]
    normalizeTestSet.insert(0, 1)


    print("\n\t __________________________ Results __________________________")

    print ("\n\tFinal result theta parameters: ", theta[0][0], theta[1][0], theta[2][0])

    print("\n\tTesting value used: ", testSet)

    print("\tTesting value after normalization: ", normalizeTestSet)

    actual = float(h(theta, normalizeTestSet))
    print("\n\tPrice of house with 1650 square feet and 3 bedrooms must be around: $%0.2f" % actual)
    predicted = float(h(normalEquationPredict(X, y), [1, 1650., 3]))
    print("\n\tPrediction for price of house with 1650 square feet and 3 bedrooms: $%0.2f" % predicted)

    print("\n\tPercentage error: %0.3f" % float(100-(100*predicted/actual)),"%")
    print("\n******************* Exiting **********************")

if __name__ == '__main__':
    main()