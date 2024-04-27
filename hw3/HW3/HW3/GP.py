import numpy as np
import scipy

### -------------------------------- ###
# Question 2a

class GaussianProcess():
    def __init__(self, X1, Y1, kernel_func=None, noise=1e-2):
        # X1: (N x 3) inputs of training data
        # Y1: (N x 1) outputs of training data
        # kernel_func: (function) a function defining your kernel. It should take the form f(X1, X2) = K, where K is N x N if X1, X2 are N x k.
        # where k is the number of feature dimensions

        self.noise = noise
        self.X1 = X1
        self.Y1 = Y1

        self.kernel_func = kernel_func

        self.compute_training_covariance()

    def compute_training_covariance(self):
        # Computes the training covariance matrix Σ11(X, X) using self.kernel_func and your input data self.X1

        ### STUDENT CODE BEGINS ###

        # Kernel of the observations
        self.Σ11 = ...

        ### STUDENT CODE ENDS ###

    def compute_posterior(self, X):
        # X: (N x k) set of inputs used to predict
        # μ2: (N x 1) GP means at inputs X
        # Σ2: (N x N) GP means at inputs X

        ### STUDENT CODE BEGINS ###
        # Students should make use of the training outputs self.Y1 at some point, as well as self.kernel_func

        # Compute posterior mean
        μ2 = ...

        # Compute the posterior covariance
        Σ2 = ...

        ### STUDENT CODE ENDS ###


        return μ2, Σ2  # posterior mean, covariance

# Question 2b
def plot_GP(mu, Sigma, X, ax):
    # mu: (N x 1) GP means
    # Sigma: (N x N) GP covariances
    # X: (N x k) id's for mu (the x-axis plot)
    # ax: (object) figure axes

    mu = mu.squeeze()
    X = X.squeeze()
    ax.plot(X, mu, label='mean')

    ### STUDENT CODE BEGINS ###
    std = ...

    confidence_interval_top = ...
    confidence_interval_bottom = ...

    ax.fill_between(X, confidence_interval_bottom, confidence_interval_top, color='green', alpha=0.3)

    ### STUDENT CODE ENDS ###

    return ax

###### KERNELS ######

# Question 2c
def radial_basis(X1, X2, sig=1., l=.1):
    # Implement the radial basis kernel, given two data matrices X1 and X2
    ### STUDENT CODE BEGINS ###

    K = ...
    ### STUDENT CODE ENDS ###

    return K

# Question 2d
def exponential_sine_squared(X1, X2, sig=1., l=.05, p=1.):
    # Implement the exponential sine squared kernel, given two data matrices X1 and X2
    ### STUDENT CODE BEGINS ###

    K = ...
    ### STUDENT CODE ENDS ###

    return K

# Question 2e
def combined_kernel(X1, X2, sig1=1, l1=.05, sig2=1., l2=.05, p=1.):
    # Combines the exponential sine squared kernel and radial basis kernel through multiplication.
    ### STUDENT CODE BEGINS ###

    K = ...
    ### STUDENT CODE ENDS ###

    return K
