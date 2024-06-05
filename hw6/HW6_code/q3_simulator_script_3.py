###############
# Script to work with the simulator for the mobile robot system in 
# HW6 Question 3.
###############

import numpy as np

import q3_simulator_class as sim

# Define the parameters
n_steps = 100
pose0 = [0, 0, 0]

# Initialize the simulator
# Arguments:
# q_mat: np.array - process noise covariance matrix
# r_mat: np.array - measurement noise covariance matrix
# dt: float - time step
# base_station_locs: np.array - locations of the base stations
# rng_seed: int - for reproducibility
base_station_locs = np.array([[30, -5], [-10, 20]])
dt = 0.5
rng_seed = 10
q_mat = 0.1 * dt * np.eye(3)
r_mat = 0.1 * np.eye(2)

robot = sim.MobileRobotSimulator(q_mat=q_mat, r_mat=r_mat, dt=dt, base_station_locs=base_station_locs, rng_seed=rng_seed)

pose_hist, y_hist, u_hist = robot.simulate(pose0, n_steps)

fig = robot.plot_position_history(pose_hist, show_plot=True)
#fig.savefig("robot_position_history.png", dpi=300, bbox_inches='tight')

fig = robot.plot_pose_history(pose_hist, show_plot=True)
#fig.savefig("robot_pose_history.png", dpi=300, bbox_inches='tight')

fig = robot.plot_measurement_history(y_hist, show_plot=True)
#fig.savefig("robot_measurement_history.png", dpi=300, bbox_inches='tight')

fig = robot.plot_control_history(u_hist, show_plot=True)
#fig.savefig("robot_control_history.png", dpi=300, bbox_inches='tight')

# Just like in the quadrotor simulator, you can run a single simulation
# using the simulate method (i.e., robot.simulate(...))
# Arguments:
# pose0: np.array - initial pose
# n_steps: int - number of steps to simulate
#
# Returns:
# pose_hist: np.array - pose history
# y_hist: np.array - measurement history
# u_hist: np.array - control history

# We offer very similar plotting functions to visualize the results as in the
# quadrotor simulator:
# 1. robot.plot_position_history(...) - plots the 2D position over time
# 2. robot.plot_pose_history(...) - plots the 2D pose over time [a bit slower]
# 3. robot.plot_control_history(...) - plots each control input against time
# 4. robot.plot_measurement_history(...) - plots each measurement against time
# 5. robot.plot_2D_errors(...) - plots the 2D errors for covariance analysis

# If you want to run multiple simulations, you can use the 
# simulate_multiple_runs method (i.e., robot.simulate_multiple_runs(...))
# Arguments:
# pose0: np.array - initial pose
# n_steps: int - number of steps to simulate
# n_sims: int - number of simulations to run
#
# Returns:
# pose_run_hists: np.array - pose history for each run
# y_run_hists: np.array - measurement history for each run
# u_run_hists: np.array - control history for each run

# We, again, offer similar plotting functions to visualize the results:
# 1. robot.plot_position_histories(...) - plots the 2D positions across runs
# 2. robot.plot_measurement_histories(...) - plots the measurements across runs

# %% Interfacing with your EKF, UKF, or particle filter

# As in the quadrotor simulator, you can either generate the _whole_ simulation
# data and then pass it to your filter, or you can run the filter and simulator
# in lockstep. The latter is more realistic, but the former is even more
# convient in this case than the previous case becuase you need to make sure
# That the EKF, UKF, and PF are operating on the _same_ run!

# If you are using the first method, you will use robot.simulate(...)
# But, since u_hist_to_replace and y_hist_to_replace are defined at the start
# and fixed, you can run each filter in a separate for loop, if you want.

import scipy
import matplotlib.pyplot as plt

# Function to calculate 2D error ellipse
def plot_ellipse(P, mu, Sigma, fig, label="", color="C2"):
    r = np.sqrt(-2*np.log(1-P))
    
    theta = np.linspace(0, 2*np.pi)
    w = np.stack((r*np.cos(theta), r*np.sin(theta)))
    
    x = scipy.linalg.sqrtm(Sigma) @ w + mu

    plt.figure(fig.number)
    plt.plot(x[0,:], x[1,:], label=label, c=color)

###### EKF Implementation ######

u_hist_to_replace = u_hist
y_hist_to_replace = y_hist

m = base_station_locs
dt = 0.5
q_mat = 0.1 * dt * np.eye(3)
r_mat = 0.1 * np.eye(2)
s_t = 1.
phi_t = 0.
Q = q_mat
R = r_mat
p = 0.95

# Initial robot states
mu_0 = np.zeros(3).reshape(-1, 1)
sigma_0 = 0.01 * np.eye(3)
mu_t = mu_0
sigma_t = sigma_0

A = np.eye(3)
C = np.zeros((2, 3))

state_est = [mu_0]

fig1 = plt.figure()
plot_ellipse(p, mu_0[:2], sigma_0[:2, :2], fig1, label='Error Ellipse')

for t_index, (u, y) in enumerate(zip(u_hist_to_replace, y_hist_to_replace)):
    ### Predict Step ###

    # Update Jacobian matrix A
    A[0, 2] = -dt * s_t * np.sin(mu_t[2])
    A[1, 2] = dt * s_t * np.cos(mu_t[2])

    #phi_t = np.sin(t_index)
    phi_t = u[1]
    mu_t = np.array([mu_t[0] + dt * s_t * np.cos(mu_t[2]), mu_t[1] + dt * s_t * np.sin(mu_t[2]), mu_t[2] + dt * phi_t]).reshape(-1, 1)
    sigma_t = A @ sigma_t @ A.T + Q
    
    ### Update Step ###
    diff = m - mu_t[:2].reshape(1, -1) # Find difference between current position estimation and each landmark
    norms = np.linalg.norm(diff, axis=1)

    # Update Jacobian Matrix C
    C[0, 0] = -diff[0, 0] / norms[0]
    C[0, 1] = -diff[0, 1] / norms[0]
    C[1, 0] = -diff[1, 0] / norms[1]
    C[1, 1] = -diff[1, 1] / norms[1]

    K_t = sigma_t @ C.T @ np.linalg.inv(C @ sigma_t @ C.T + R)

    g = norms.reshape(-1, 1)

    mu_t = mu_t + K_t @ (y.reshape(-1, 1) - g)
    sigma_t = sigma_t - K_t @ C @ sigma_t

    state_est.append(mu_t)

    plot_ellipse(p, mu_t[:2], sigma_t[:2, :2], fig1)

state_est_array = np.array(state_est).reshape(-1, 3)

#### Postion Plot for EKF ####
plt.figure(fig1.number)
plt.scatter(mu_0[0], mu_0[1], color='black', marker='*', s=100, label='Initial Position')
plt.plot(state_est_array[:,0], state_est_array[:,1], label='Extended Kalman Filter Position')
plt.plot(pose_hist[:, 0], pose_hist[:, 1], label='Ground-Truth Trajectory')

plt.legend(loc='lower right')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Means and Error Ellipses of the Position Estimate for EKF Over the True Trajectory')
plt.savefig('Mean and error ellipses of the position estimate for EKF over the true trajectory.png', dpi=300, bbox_inches='tight')


# %%
###### UKF Implementation ######

m = base_station_locs
dt = 0.5
q_mat = 0.1 * dt * np.eye(3)
r_mat = 0.1 * np.eye(2)
s_t = 1.
phi_t = 0.
Q = q_mat
R = r_mat

# Initial robot states
mu_0 = np.zeros(3)
sigma_0 = 0.01 * np.eye(3)
mu_t = mu_0
sigma_t = sigma_0

n = 3 # dimension of state space
lambda_param = 2.
num_x_sampled = 2 * n + 1
w_1 = 1. / (2. * (lambda_param + n))
w_0 = lambda_param / (lambda_param + n)

p = 0.95

state_est_UKF = [mu_0.reshape(-1, 1)]

def ukf_state_transition(mu, u):
    mu = mu.flatten()
    u = u.flatten()
    px, py, theta = mu
    s_t, phi_t = u

    px = px + dt * s_t * np.cos(theta)
    py = py + dt * s_t * np.sin(theta)
    theta = theta + dt * phi_t
    return np.array([px, py, theta])


def ukf_predict(mu, sigma, u):
    sigma_square = scipy.linalg.sqrtm(sigma)
    sigma_square = np.sqrt(lambda_param + n) * sigma_square

    x_sampled = np.zeros((n, num_x_sampled))
    sigma_new = np.zeros((n, n))

    for i in range(n + 1):
        if i == 0:
            mu_new= ukf_state_transition(mu, u)
            x_sampled[:, i] = mu_new
        else:
            
            x_sampled[:, i] = mu + sigma_square[:, i - 1] # x_i
            x_sampled[:, i + n] = mu - sigma_square[:, i - 1] # symmetric counterpart of x_i
            x_sampled[:, i] = ukf_state_transition(x_sampled[:, i], u)
            x_sampled[:, i + n] = ukf_state_transition(x_sampled[:, i + n], u)

    mu_weighted = w_1 * x_sampled
    mu_weighted[:, 0] = w_0 * x_sampled[:, 0]
    
    mu_new = np.sum(mu_weighted, axis=1)
    

    for i in range(num_x_sampled):
        if i == 0:
            sigma_new = sigma_new + w_0 * ((x_sampled[:, i] - mu_new).reshape(-1, 1)) @ ((x_sampled[:, i] - mu_new).reshape(-1, 1)).T
        else:
            sigma_new = sigma_new + w_1 * ((x_sampled[:, i] - mu_new).reshape(-1, 1)) @ ((x_sampled[:, i] - mu_new).reshape(-1, 1)).T

    sigma_new = sigma_new + Q

    return mu_new.flatten(), sigma_new


def find_measurement(x_i):
    diff = m - x_i[:2].reshape(1, -1)
    norms = np.linalg.norm(diff, axis=1)
    return norms


def ukf_update(mu_x, sigma, y):
    sigma_square = scipy.linalg.sqrtm(sigma)
    sigma_square = np.sqrt(lambda_param + n) * sigma_square

    y_sampled = np.zeros((n - 1, num_x_sampled))
    x_sampled = np.zeros((n, num_x_sampled))
    sigma_xy_new = np.zeros((n, n - 1))
    sigma_y_new = np.zeros((n - 1, n - 1))
    
    for i in range(n + 1):
        if i == 0:
            x_sampled[:, i] = mu_x
            y_sampled[:, i] = find_measurement(mu_x)
        else:
            x_sampled[:, i] = mu_x + sigma_square[:, i - 1] # x_i
            x_sampled[:, i + n] = mu_x - sigma_square[:, i - 1] # symmetric counterpart of x_i
            y_sampled[:, i] = find_measurement(x_sampled[:, i])
            y_sampled[:, i + n] = find_measurement(x_sampled[:, i + n])

    
    mu_y_weighted = w_1 * y_sampled
    mu_y_weighted[:, 0] = w_0 * y_sampled[:, 0]
    
    mu_y_new = np.sum(mu_y_weighted, axis=1)

    for i in range(num_x_sampled):
        if i == 0:
            sigma_y_new = sigma_y_new + w_0 * ((y_sampled[:, i] - mu_y_new).reshape(-1, 1)) @ ((y_sampled[:, i] - mu_y_new).reshape(-1, 1)).T
            sigma_xy_new = sigma_xy_new + w_0 *(x_sampled[:, i] - mu_x).reshape(-1, 1) @ ((y_sampled[:, i] - mu_y_new).reshape(-1, 1)).T
        else:
            sigma_y_new = sigma_y_new + w_1 * ((y_sampled[:, i] - mu_y_new).reshape(-1, 1)) @ ((y_sampled[:, i] - mu_y_new).reshape(-1, 1)).T
            sigma_xy_new = sigma_xy_new + w_1 *(x_sampled[:, i] - mu_x).reshape(-1, 1) @ ((y_sampled[:, i] - mu_y_new).reshape(-1, 1)).T
    
    sigma_y_new = sigma_y_new + R

    mu_x_new = mu_x.reshape(-1, 1) + sigma_xy_new @ np.linalg.inv(sigma_y_new) @ (y.reshape(-1, 1) - mu_y_new.reshape(-1, 1))
    sigma_new = sigma - sigma_xy_new @ np.linalg.inv(sigma_y_new) @ sigma_xy_new.T

    return mu_x_new.flatten(), sigma_new
    
   
fig2 = plt.figure()
mu_0_ = mu_0.reshape(-1, 1)
plot_ellipse(p, mu_0_[:2], sigma_0[:2, :2], fig1, label='Error Ellipse')

for t_index, (u, y) in enumerate(zip(u_hist_to_replace, y_hist_to_replace)):   
    ### Predict Step ###
    mu_t, sigma_t = ukf_predict(mu_t, sigma_t, u)
    
    ### Update Step ###
    mu_t, sigma_t = ukf_update(mu_t, sigma_t, y)

    state_est_UKF.append(mu_t.reshape(-1, 1))

    mu_t_ = mu_t.reshape(-1, 1)

    plot_ellipse(p, mu_t_[:2], sigma_t[:2, :2], fig2)

state_est_array_UKF = np.array(state_est_UKF).reshape(-1, 3)

#### Postion Plot for UKF ####
plt.figure(fig2.number)
plt.scatter(mu_0[0], mu_0[1], color='black', marker='*', s=100, label='Initial Position')
plt.plot(state_est_array_UKF[:,0], state_est_array_UKF[:,1], label='Uscented Kalman Filter Position')
plt.plot(pose_hist[:, 0], pose_hist[:, 1], label='Ground-Truth Trajectory')

plt.legend(loc='lower right')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Means and Error Ellipses of the Position Estimate for UKF Over the True Trajectory')
plt.savefig('Mean and error ellipses of the position estimate for UKF over the true trajectory.png', dpi=300, bbox_inches='tight')



# %%
###### PF Implementation ######
m = base_station_locs
dt = 0.5
q_mat = 0.1 * dt * np.eye(3)
r_mat = 0.1 * np.eye(2)
s_t = 1.
phi_t = 0.
Q = q_mat
R = r_mat

# Initialize particle filter
mu_0 = pose0
sigma_0 = 0.01 * np.eye(3)
num_particles = 1000 # Number of particles sampled
particles = np.random.multivariate_normal(mu_0, sigma_0, num_particles) # sample from the prior distribution
weights = np.ones(num_particles) / num_particles # initially, assume uniform weights


def pf_find_measurement(particles):
    diff = m - particles[:, None, :2]
    norms = np.linalg.norm(diff, axis=2)
    return norms

def pf_state_transition(particles, u):
    u = u.flatten()
    s_t, phi_t = u

    particles[:, 0] += dt * s_t * np.cos(particles[:, 2])
    particles[:, 1] += dt * s_t * np.sin(particles[:, 2])
    particles[:, 2] += dt * phi_t

    particles += np.random.multivariate_normal([0, 0, 0], Q, num_particles)
    return particles

def pf_predict(particles, u):
    particles = pf_state_transition(particles, u)
    return particles

def pf_update(particles, weights, y):
    g_t = pf_find_measurement(particles) # observation for each particle
    y_diff = g_t - y.flatten()
    
    measurement_likelihoods = np.exp(-0.5 * np.diagonal(y_diff @ np.linalg.inv(R) @ y_diff.T))
    #measurement_likelihoods = np.zeros(num_particles)
    #for i in range(num_particles):
    #    g_i = find_measurement(particles[i, :])
    #    y_diff = (y.flatten() - g_i).reshape(-1, 1)
    #    measurement_likelihoods[i] = np.exp(-0.5 * y_diff.T @ np.linalg.inv(R) @ y_diff) 
   
    if np.random.uniform(0, 1) <= 0.9:
        weights = measurement_likelihoods / np.sum(measurement_likelihoods)
        particles, weights = importance_resampling(particles, weights)
    else:
        weights = measurement_likelihoods * weights
        weights = weights / np.sum(weights)
    #weights = measurement_likelihoods / np.sum(measurement_likelihoods)
    #particles, weights = importance_resampling(particles, weights)
    return particles, weights

def importance_resampling(particles, weights):
    chosen_indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    particles = particles[chosen_indices]
    weights = np.ones(num_particles) / num_particles
    return particles, weights

fig3 = plt.figure()
plt.scatter(particles[:, 0], particles[:, 1], s=1, alpha=0.05)    
for t_index, (u, y) in enumerate(zip(u_hist_to_replace, y_hist_to_replace)):   
    ### Predict Step ###
    particles = pf_predict(particles, u)
    
    ### Update Step ###
    particles, weights = pf_update(particles, weights, y)
    
    plt.scatter(particles[:, 0], particles[:, 1], s=1, alpha=0.05)
    
#### Postion Plot for UKF ####
plt.figure(fig3.number)
plt.scatter(mu_0[0], mu_0[1], color='black', marker='*', s=100, label='Initial Position')
plt.plot(pose_hist[:, 0], pose_hist[:, 1], label='Ground-Truth Trajectory')

plt.legend(loc='lower right')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Estimated Trajectory with Covariance Ellipses of the Position Estimate for Particle Filter Over the True Trajectory')
plt.savefig('Estimated Trajectory with Covariance Ellipses of the Position Estimate for PF over the true trajectory.png', dpi=300, bbox_inches='tight')
    