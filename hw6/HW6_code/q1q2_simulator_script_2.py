###############
# Script to work with the simulator for the quadrotor system in 
# HW6 Question 1 and Question 2.
###############

import numpy as np

import q1q2_simulator_class as sim

# Define the parameters
n_steps = 100
p0 = [1000, 0]
s0 = [0, 50]

# %% Question 1

# Initialize the simulator
# Arguments:
# sensor: str - "GPS" for Question 1, "Velocity" for Question 2
# q_mat: np.array - process noise covariance matrix
# r_mat: np.array - measurement noise covariance matrix
# dt: float - time step
# rng_seed: int - for reproducibility

q_mat = 1. * np.eye(2)
r_mat = 9. * np.eye(2)
dt = 1
rng_seed = 1

quadrotor = sim.QuadrotorSimulator(sensor="GPS", q_mat=q_mat, r_mat=r_mat, dt=dt, rng_seed=rng_seed)

phist, shist, yhist, uhist = quadrotor.simulate(p0, s0, n_steps)

fig = quadrotor.plot_position_history(phist, yhist, show_plot=True)
fig.savefig("position_history_1.png", dpi=300, bbox_inches="tight")

fig = quadrotor.plot_velocity_history(shist, show_plot=True)
fig.savefig("velocity_history_1.png", dpi=300, bbox_inches="tight")

# If you want to run a single simulation, use the simulate method
# i.e., quadrotor.simulate(...)
# Arguments:
# p0: np.array - initial position
# s0: np.array - initial velocity
# n_steps: int - number of steps to simulate
#
# Returns:
# phist: np.array - position history
# shist: np.array - velocity history
# yhist: np.array - measurement history
# uhist: np.array - control history

# We offer several plotting functions to visualize the results:
# 1. quadrotor.plot_position_history(...) - plots the 2D position over time
# 2. quadrotor.plot_velocity_history(...) - plots the 2D velocity over time
# 3. quadrotor.plot_control_history(...) - plots the 2D control input over time
#
# These functions return a matplotlib figure object, which you can save using
# fig.savefig(...). If you want to display the plot, you can use show_plot=True.
# For example:
# fig = quadrotor.plot_position_history(phist, yhist, show_plot=True)
# fig.savefig("position_history.png", dpi=300, bbox_inches="tight")
# Of course, you can also plot the results yourself using matplotlib.
#
# To debug your filter, you may also want to plot the errors:
# i.e., quadrotor.plot_2D_errors(...) 
# For example, to plot the process errors, use:
# s_diff = shist[1:-1, :] - shist[:-2, :]  (i.e., s_t+1 - s_t)
# quadrotor.plot_2D_errors(uhist[:-2], s_diff, 
#                           err_cov=quadrotor.q_mat, 
#                           title="Control Process Error")
# If you want to plot the measurement errors, use:
# quadrotor.plot_2D_errors(phist, yhist,
#                           err_cov=quadrotor.r_mat,
#                           title="GPS Measurement Error")


# If you want to run multiple simulations, use the simulate_multiple_runs method
# i.e., quadrotor.simulate_multiple_runs(...)
#
# Arguments:
# p0: np.array - initial position
# s0: np.array - initial velocity
# n_steps: int - number of steps to simulate
# n_sims: int - number of simulations to run
# Returns:
# prun_hists: np.array - array of position histories
# srun_hists: np.array - array of velocity histories
# yrun_hists: np.array - array of measurement histories
# urun_hists: np.array - array of control histories

# We, again, offer several plotting functions to visualize the results:
# 1. quadrotor.plot_position_histories(...) - plots the 2D position across runs
# 2. quadrotor.plot_velocity_histories(...) - plots the 2D velocity across runs
#
# To plot the errors, use the same functions as above, but you will want to
# reshape the data to be 2D. 
# For example, to plot the process errors across runs:
# s_diffs = srun_hists[:, 1:-1, :] - srun_hists[:, :-2, :]
# s_diffs = s_diffs.reshape(-1, 2)
# quadrotor.plot_2D_errors(urun_hists[:, :-2].reshape(-1, 2), s_diffs,
#                          err_cov=quadrotor.q_mat,
#                          title="Control Process Error (Multi-run)")


# %% Interfacing with your Kalman Filter implementation

# You will use the simulator to generate data to test your Kalman Filter 
# implementation. There are two ways to do this:
# 1. [Recommended] Run the _whole_ simulation in the simulator and pass the
#    generated measurements to your Kalman Filter implementation one-by-one.
#    This works since we are _not_ using active control (akin to using data 
#    from a previously collected dataset). It will make sure that the 
#    measurements you are using are consistent with the simulator ground
#    truth, and that the ground truth state does not end up in your filter.
# 2. Run the simulator and Kalman Filter in lockstep. This is more realistic
#    but can be harder to debug. 

# If you are using the first method, you can use the simulator as described
# above. 
# i.e., use quadrotor.simulate(...) to generate the data, and then loop through
# the measurements to pass them to your Kalman Filter implementation.

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


q_mat = 1. * np.eye(2)
r_mat = 9. * np.eye(2)
dt = 1

mu_0 = np.array([1500., 100., 0., 55.]).reshape(-1, 1)
sigma_0 = np.array([[250000., 0., 0., 0.], [0., 250000., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

A = np.array([[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]])
B = np.array([[0., 0.], [0., 0.], [dt, 0.], [0., dt]])
C = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])

Q = np.zeros((4, 4))
Q[2, 2] = 1.
Q[3, 3] = 1.

R = q_mat

u_hist_to_replace = uhist
y_hist_to_replace = yhist

mu_t = mu_0
sigma_t = sigma_0

state_est = [mu_0]
cov_est = [sigma_0]

p = 0.95

fig1 = plt.figure()
fig2 = plt.figure()

plot_ellipse(p, mu_0[:2], sigma_0[:2, :2], fig1, label='Error Ellipse')
plot_ellipse(p, mu_0[2:], sigma_0[2:, 2:], fig2, label='Error Ellipse')

for t_index, (u, y) in enumerate(zip(u_hist_to_replace, y_hist_to_replace)):    
    # Predict Step
    mu_t = A @ mu_t + B @ u.reshape(-1, 1)
    sigma_t = A @ sigma_t @ A.T + Q
    
    # Update Step
    K_t = sigma_t @ C.T @ np.linalg.inv(C @ sigma_t @ C.T + R)
    mu_t = mu_t + K_t @ (y.reshape(-1, 1) - C @ mu_t)
    sigma_t = sigma_t - K_t @ C @ sigma_t

    state_est.append(mu_t)
    cov_est.append(sigma_t)

    plot_ellipse(p, mu_t[:2], sigma_t[:2, :2], fig1)
    plot_ellipse(p, mu_t[2:], sigma_t[2:, 2:], fig2)

state_est_array = np.array(state_est).reshape(-1, 4)

##### (c) #####
plt.figure(fig1.number)
plt.scatter(mu_0[0], mu_0[1], color='black', marker='*', s=100, label='Initial Position')
plt.plot(state_est_array[:,0], state_est_array[:,1], label='Kalmen Filter Position')
plt.plot(phist[:, 0], phist[:, 1], label='Ground-Truth Trajectory')
plt.scatter(yhist[:,0], yhist[:,1], s=5, color='red', label='GPS Measurement')

plt.legend(loc='lower right')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Means and Error Ellipses of the Position Estimate Over the True Trajectory')
#plt.xlim(900, 1100)
#plt.ylim(0, 200)
plt.savefig('Mean and error ellipses of the position estimate over the true trajectory.png', dpi=300, bbox_inches='tight')

##### (d) #####
plt.figure(fig2.number)
plt.scatter(mu_0[-2], mu_0[-1], color='black', marker='*', s=100, label='Initial Velocity')
plt.plot(state_est_array[:, 2], state_est_array[:, 3], label='Kalmen Filter Velocity')
plt.plot(shist[:, 0], shist[:, 1], label='Ground-Truth Velocity Trajectory')

plt.legend(loc='upper right')
plt.xlabel('V_x (m/s)')
plt.ylabel('V_y (m/s)')
plt.title('Means and Error Ellipses of the Velocity Along the Trajectory of the Drone')
plt.savefig('Means and Error Ellipses of the Velocity.png', dpi=300, bbox_inches='tight')
plt.show()


# %% Question 2

q_mat = 1. * np.eye(2)
r_mat = 9. * np.eye(2)
dt = 1
rng_speed = 1

# Initialize the simulator
quadrotor = sim.QuadrotorSimulator(sensor="Velocity", q_mat=q_mat, r_mat=r_mat, dt=dt, rng_seed=rng_speed)

phist, shist, yhist, uhist = quadrotor.simulate(p0, s0, n_steps)

fig = quadrotor.plot_position_history(phist, show_plot=True)
fig.savefig("position_history_inertial_1.png", dpi=300, bbox_inches="tight")

fig = quadrotor.plot_velocity_history(shist, yhist, show_plot=True)
fig.savefig("velocity_history_inertial_1.png", dpi=300, bbox_inches="tight")

# %%

import scipy
import matplotlib.pyplot as plt

q_mat = 1. * np.eye(2)
r_mat = 9. * np.eye(2)
dt = 1

mu_0 = np.array([1000., 0., 0., 50.]).reshape(-1, 1)
sigma_0 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

A = np.array([[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]])
B = np.array([[0., 0.], [0., 0.], [dt, 0.], [0., dt]])
C = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.]])

Q = np.zeros((4, 4))
Q[2, 2] = 1.
Q[3, 3] = 1.

R = q_mat

u_hist_to_replace = uhist
y_hist_to_replace = yhist

mu_t = mu_0
sigma_t = sigma_0

state_est = [mu_0]
cov_est = [sigma_0]

p = 0.95

fig1 = plt.figure()
fig2 = plt.figure()

plot_ellipse(p, mu_0[:2], sigma_0[:2, :2], fig1, label='Error Ellipse')
plot_ellipse(p, mu_0[2:], sigma_0[2:, 2:], fig2, label='Error Ellipse')

for t_index, (u, y) in enumerate(zip(u_hist_to_replace, y_hist_to_replace)):   
    # Predict Step
    mu_t = A @ mu_t + B @ u.reshape(-1, 1)
    sigma_t = A @ sigma_t @ A.T + Q
    
    # Update Step
    K_t = sigma_t @ C.T @ np.linalg.inv(C @ sigma_t @ C.T + R)
    mu_t = mu_t + K_t @ (y.reshape(-1, 1) - C @ mu_t)
    sigma_t = sigma_t - K_t @ C @ sigma_t

    state_est.append(mu_t)
    cov_est.append(sigma_t)

    plot_ellipse(p, mu_t[:2], sigma_t[:2, :2], fig1)
    plot_ellipse(p, mu_t[2:], sigma_t[2:, 2:], fig2)

state_est_array = np.array(state_est).reshape(-1, 4)

##### Position Plot #####
plt.figure(fig1.number)
plt.scatter(mu_0[0], mu_0[1], color='black', marker='*', s=100, label='Initial Position')
plt.plot(state_est_array[:,0], state_est_array[:,1], label='Kalman Filter Position')
plt.plot(phist[:, 0], phist[:, 1], label='Ground-Truth Trajectory')

plt.legend(loc='lower right')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Means and Error Ellipses of the Position Estimate for Inertial Systems Over the True Trajectory')
#plt.xlim(900, 1100)
#plt.ylim(0, 200)
plt.savefig('Mean and error ellipses of the position estimate for Inertial Systems over the true trajectory.png', dpi=300, bbox_inches='tight')

##### Velocity Plot #####
plt.figure(fig2.number)
plt.scatter(mu_0[-2], mu_0[-1], color='black', marker='*', s=100, label='Initial Velocity')
plt.plot(state_est_array[:, 2], state_est_array[:, 3], label='Kalman Filter Velocity')
plt.plot(shist[:, 0], shist[:, 1], label='Ground-Truth Velocity Trajectory')
plt.scatter(yhist[:,0], yhist[:,1], s=10, color='red', label='Velocity Measurement')

plt.legend(loc='upper right')
plt.xlabel('V_x (m/s)')
plt.ylabel('V_y (m/s)')
plt.title('Means and Error Ellipses of the Velocity for Inertial System Along the Trajectory of the Drone')
plt.savefig('Means and Error Ellipses of the Velocity for Inertial System.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
