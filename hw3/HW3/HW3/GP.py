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

        # Compute prior obseravations Σy1:N from kernel
        self.Σ11 = self.kernel_func(self.X1, self.X1) # (N X N)
        
        # Adding noise to the diagonal elements
        N = self.X1.shape[0]
        self.Σ11 += self.noise * np.eye(N)

        ### STUDENT CODE ENDS ###

    def compute_posterior(self, X):
        # X: (N x k) set of inputs used to predict
        # μ2: (N x 1) GP means at inputs X
        # Σ2: (N x N) GP means at inputs X 

        ### STUDENT CODE BEGINS ###
        # Students should make use of the training outputs self.Y1 at some point, as well as self.kernel_func
        N = self.X1.shape[0] # num of prior data points, 3-dimension
        M = X.shape[0] # num of new data points, k-dimesion
        
        # cov Σy_ (based on prior data)
        Σy_ = self.Σ11

        # cross cov, Σyy1:N
        Σyy_ = self.kernel_func(X, self.X1)
        # Σyy_ = np.zeros((M,N)) # MxN matrix
        # for i in range(M):
        #     for j in range(N):
        #         Σyy_[i, j] = self.kernel_func(X[i], self.X1[j]) 

        # cov Σy (based on new data)
        Σy = self.kernel_func(X, X)
        # Σy = np.zeros((M,M))
        # for i in range(M):
        #     for j in range(M):
        #         Σy[i, j] = self.kernel_func(X[i],X[j])

        # Compute posterior mean
        μy = np.zeros((M,1)) # initialize to be zero (?)
        μy_ = np.zeros((N,1)) # zero prior mean (also?)
        # print("μy shape:",μy.shape)
        # print("Σyy_ shape:",Σyy_.shape)
        # print("Σy_^-1 shape:",np.linalg.inv(Σy_).shape)
        # print(self.)
        # check if they are all PDE
        # print("Σyy_",Σyy_,"Σy_",Σy_, "Σy",Σy)
        μ2 = μy + Σyy_ @ np.linalg.inv(Σy_) @ (self.Y1 - μy_) # (Mx1) + (MxN)(NxN)

        # Compute the posterior covariance
        Σ2 = Σy - Σyy_ @ np.linalg.inv(Σy_) @ Σyy_.T # (MxM) + (MxN)(NXN)(NxM)

        ### STUDENT CODE ENDS ##
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
    # print(np.diag(Sigma))
    std = np.sqrt(np.diag(Sigma)) # square root of the diagonal entry in cov mat, cov(Yi,yj)
    # print("sigma:",Sigma)
    # print("std",std)
    # print("mu.shape:",mu.shape)
    # print("std.shape:",std.shape)
    confidence_interval_top = mu + std
    confidence_interval_bottom = mu - std

    ax.fill_between(X, confidence_interval_bottom, confidence_interval_top, color='green', alpha=0.3)

    ### STUDENT CODE ENDS ###

    return ax

###### KERNELS ######

# Question 2c
def radial_basis(X1, X2, sig=1., l=.1):
    # Implement the radial basis kernel, given two data matrices X1 and X2
    ### STUDENT CODE BEGINS ###

    # K = sig**2 * np.exp(-1/(2*l**2)* (X1-X2).T @ (X1-X2))

    # Reshape X1 and X2 for broadcasting
    X1_reshaped = X1[:, np.newaxis, :] # (N x 1 x k)
    X2_reshaped = X2[np.newaxis, :, :] # (1 x M x k)
    # print("X1:", X1_reshaped.shape, "X2:", X2_reshaped.shape)

    # Compute the difference between X1 and X2 for each pair of points
    diff = X1_reshaped - X2_reshaped # (N x M x k)
    squared_norm = np.linalg.norm((diff), axis=-1)**2 # compute it over the last axis (N x M)

    # Compute the Radial Basis Function kernel using the formula
    rbf_kernel = (sig**2) * np.exp(-squared_norm / (2 * l**2))

    ### STUDENT CODE ENDS ###

    return rbf_kernel

# Question 2d
def exponential_sine_squared(X1, X2, sig=1., l=.05, p=1.):
    # Implement the exponential sine squared kernel, given two data matrices X1 and X2
    ### STUDENT CODE BEGINS ###

    # Reshape X1 and X2 for broadcasting
    X1_reshaped = X1[:, np.newaxis, :] # (N x 1 x k)
    X2_reshaped = X2[np.newaxis, :, :] # (1 x M x k)

    # Compute the difference between X1 and X2 for each pair of points
    diff = X1_reshaped - X2_reshaped # (N x M x k)
    squared_norm = np.linalg.norm((diff), axis=-1)**2 # compute it over the last axis (N x M)

    # compute exp kernel
    exp_kernel = (sig**2) * np.exp(-np.sin((np.pi/p) * squared_norm) **2/ (2 * l**2))
    ### STUDENT CODE ENDS ###

    return exp_kernel

# Question 2e
def combined_kernel(X1, X2, sig1=1, l1=.05, sig2=1., l2=.05, p=1.):
    # Combines the exponential sine squared kernel and radial basis kernel through multiplication.
    ### STUDENT CODE BEGINS ###

    K1 = radial_basis(X1,X2,sig1,l1)
    K2 = exponential_sine_squared(X1,X2,sig2,l2,p)
    K = K1*K2
    ### STUDENT CODE ENDS ###

    return K
