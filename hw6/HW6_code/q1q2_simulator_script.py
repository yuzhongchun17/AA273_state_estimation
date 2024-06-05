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

quadrotor = sim.QuadrotorSimulator(sensor="GPS", rng_seed=42) # defualt noise

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

pos_hist, vel_hist, meas_pose_hist, uhist = quadrotor.simulate(p0=p0, s0=s0, num_t=n_steps)
print(uhist.shape)
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

# fig = quadrotor.plot_position_history(pos_hist, meas_pose_hist, show_plot=True)
# fig.savefig("position_history_1.png", dpi=300, bbox_inches="tight")

# fig = quadrotor.plot_velocity_history(vel_hist, show_plot=True)
# fig.savefig("velocity_history_1.png", dpi=300, bbox_inches="tight")

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
def EKF(u_arr, y_arr):
    """ Implement Extended Kalman filtering (Q2E)
    Input:
      observations: (N,2) numpy array, the sequence of observations. From T=1.
      mu_0: (6,) numpy array, the mean of state belief after T=0
      sigma_0: (6,6) numpy array, the covariance matrix for state belief after T=0.
    Output:
      state_mean: (N,6) numpy array, the filtered mean state at each time step. Not including the
                  starting state mu_0.
      state_sigma: (N,6,6) numpy array, the filtered state covariance at each time step. Not including
                  the starting state covarance matrix sigma_0.
      predicted_observation_mean: (N,2) numpy array, the mean of predicted observations. Start from T=1
      predicted_observation_sigma: (N,2,2) numpy array, the covariance matrix of predicted observations. Start from T=1
    Note:
      Keep in mind this function will be reused for Q2 by inheritance.
    """
    mu_0 = np.array([0.5, 0.0, 5.0, 0.0, 0.0, 0.0])
    sigma_0 = np.eye(6)*0.01
    sigma_0[3:,3:] = 0.0

    # 1. define matrix, A, B, C
    dt = 1.
    # state transition matrix
    A = np.array([[1., 0., dt, 0.], 
                [0., 1., 0., dt], 
                [0., 0., 1., 0.], 
                [0., 0., 0., 1.]]) # 4x4

    B = np.array([[0., 0.], 
                [0., 0.], 
                [dt, 0.], 
                [0., dt]]) # 4x2

    # Jacobian matrix (meausrment model w.r.t state)
    C = np.array([[1., 0., 0., 0.], 
                [0., 1., 0., 0.]]) # 2x4

    # noise matrix
    Q = np.array([[0., 0., 0., 0.], 
                [0., 0., 0., 0.],
                [0., 0., 1., 0.], 
                [0., 0., 0., 1.]]) # Process noisy on velocity (4x4)

    R = np.eye(2) * 9.

    mu_0 = np.array([1500., 100., 0., 55.]) # 4x1
    sigma_0 = np.array([[250000., 0., 0., 0.], 
                        [0., 250000., 0., 0.], 
                        [0., 0., 1., 0.], 
                        [0., 0., 0., 1.]]) # 4x4

    state_mean = [mu_0]
    state_sigma = [sigma_0]
    predicted_observation_mean = []
    predicted_observation_sigma = []

    for t, (u, y) in enumerate(zip(u_arr, y_arr)):
        # Prediction Step
        # print(u.shape, B.shape)
        mu_bar_next = A @ state_mean[-1] + B @ u # Predict the next state, 4x1
        sigma_bar_next = A @ state_sigma[-1] @ A.T + Q  # Predict the next covariance, 4x4

        # Update Step
        K_t_numerator = sigma_bar_next @ C.T # (4x4)(4x2) = (4x2)
        K_t_denominator = C @ sigma_bar_next @ C.T + R # 2x2
        K_t = K_t_numerator @ np.linalg.inv(K_t_denominator) # 4x2

        # observation
        expected_y = C @ mu_bar_next # pass the expected state through the measurement model --> the expected position
        mu_next = mu_bar_next + K_t @ (y - expected_y)  # Update state mean
        sigma_next = (np.eye(4) - K_t @ C) @ sigma_bar_next  # Update state covariance
        
        state_mean.append(mu_next)
        state_sigma.append(sigma_next)
        predicted_observation_mean.append(expected_y)
        predicted_observation_sigma.append(K_t_denominator)
    return np.array(state_mean[1:]), np.array(state_sigma[1:]), np.array(predicted_observation_mean), np.array(predicted_observation_sigma)
                                                                         
filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = EKF(u_arr=uhist, y_arr=meas_pose_hist)

print(pos_hist.shape)
# fig = quadrotor.plot_position_history(pos_hist, predicted_observation_mean, show_plot=True)
# fig.savefig("position_history_1.png", dpi=300, bbox_inches="tight")

# PLOT for (C)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plot_helper import draw_2d, draw_3d

np.random.seed(402)
# # solution = Q2_solution()
# # states, observations = solution.simulation()
# # plotting
# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='2d')
# plt.scatter(pos_hist[:,0], pos_hist[:,1], c=np.arange(pos_hist.shape[0]))
# plt.show()

# fig = plt.figure()
# plt.scatter(observations[:,0], observations[:,1], c=np.arange(states.shape[0]), s=4)
# plt.xlim([0,640])
# plt.ylim([0,480])
# plt.gca().invert_yaxis()
# plt.show()

# observations = np.load('./data/Q2E_measurement.npy')
# filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = \
#     solution.EKF(observations)
# # print(filtered_state_mean)
# # plotting
# true_states = np.load('./data/Q2E_state.npy')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pos_hist[:,0], pos_hist[:,1], c='C0')
# for mean, cov in zip(filtered_state_mean, filtered_state_sigma):
#     draw_3d(ax, cov[:3,:3], mean[:3])
# ax.view_init(elev=10., azim=30)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(pos_hist[:,0], pos_hist[:,1], s=4)
for mean, cov in zip(filtered_state_mean, filtered_state_sigma):
    draw_2d(ax, cov, mean)
# plt.xlim([0,640])
# plt.ylim([0,480])
plt.gca().invert_yaxis()
plt.show()

# %% Question 2

# Initialize the simulator
quadrotor = sim.QuadrotorSimulator(sensor="Velocity")

# You can use the same functions as above to run and visualize the simulation.
