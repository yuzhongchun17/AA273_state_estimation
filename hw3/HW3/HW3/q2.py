#%%
import numpy as np
import pandas as pd
from GP import GaussianProcess, plot_GP, radial_basis, exponential_sine_squared, combined_kernel
import matplotlib.pyplot as plt

np.random.seed(0)
# Goals: to construct a Gaussian Process object from scratch
# NOTE: YOU DON'T NEED TO CHANGE ANYTHING IN THIS FUNCTION! STUDENT CODE IS LOCATED IN GP.py!!!

# Here we are reading in the data, shifting the data to be approximately 0 mean and on a smaller scale.
data_filepath = 'prices.csv'
data = pd.read_csv(data_filepath)

max_prices = np.array(data['High'])
min_prices = np.array(data['Low'])
avg_prices = (max_prices + min_prices) / 2

max_prices -= np.mean(avg_prices)
min_prices -= np.mean(avg_prices)
avg_prices -= np.mean(avg_prices)

max_prices /= 20.
min_prices /= 20.
avg_prices /= 20.

id = np.linspace(0., 1., len(max_prices))

# Randomly sample a fraction of the dataset for training
frac = 0.8

random_indices = np.array(sorted(np.random.choice(np.arange(len(max_prices)), int(frac*len(max_prices)), replace=False)))

avg_prices_train = avg_prices[random_indices]
id_train = id[random_indices]

gp = GaussianProcess(id_train.reshape(-1, 1), avg_prices_train.reshape(-1, 1), kernel_func=radial_basis)

mu, Sigma = gp.compute_posterior(id.reshape(-1, 1))

fig, ax = plt.subplots(1, figsize=(15, 15))

ax = plot_GP(mu, Sigma, id, ax)
ax.plot(id, avg_prices, label='avg')
ax.plot(id, max_prices, label='max')
ax.plot(id, min_prices, label='min')

ax.legend()
plt.show()


gp = GaussianProcess(id_train.reshape(-1, 1), avg_prices_train.reshape(-1, 1), kernel_func=exponential_sine_squared)

mu, Sigma = gp.compute_posterior(id.reshape(-1, 1))

fig, ax = plt.subplots(1, figsize=(15, 15))

ax = plot_GP(mu, Sigma, id, ax)
ax.plot(id, avg_prices, label='avg')
ax.plot(id, max_prices, label='max')
ax.plot(id, min_prices, label='min')

ax.legend()
plt.show()

gp = GaussianProcess(id_train.reshape(-1, 1), avg_prices_train.reshape(-1, 1), kernel_func=combined_kernel)

mu, Sigma = gp.compute_posterior(id.reshape(-1, 1))

fig, ax = plt.subplots(1, figsize=(15, 15))

ax = plot_GP(mu, Sigma, id, ax)
ax.plot(id, avg_prices, label='avg')
ax.plot(id, max_prices, label='max')
ax.plot(id, min_prices, label='min')

ax.legend()
plt.show()

#%%