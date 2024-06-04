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
robot = sim.MobileRobotSimulator()

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

u_hist_to_replace = np.zeros((n_steps, 2))
y_hist_to_replace = np.zeros((n_steps, 2))

for t_index, (u, y) in enumerate(zip(u_hist_to_replace, y_hist_to_replace)):
    # Use (u, y) to step your filter
    # For example:
    # x_est_ekf, P_est_ekf = ekf.step(u, y)
    # x_est_ukf, P_est_ukf = ukf.step(u, y)
    # x_est_samples_pf, weights_pf = pf.step(u, y)
    pass

# If you are using the second method, you will need to run filters in the same
# for loop or use a separate rng_generator for the particle filter to avoid
# conflicting with the noise sampling in the simulator.

for t_index in range(n_steps):
    # Get the true state from a simulator dynamics step.
    # i.e., pose_true, u_true = robot.noisy_dynamics_step(
    #                 pose_true, t_index * robot.dt)
    # Then, use the true state to generate a measurement.
    # i.e., y_curr = robot.noisy_measurement_step(pose_true)

    # Use (u_true, y_curr) to step your filters, but they should never see
    # the true state (pose_true).
    # For example:
    # x_est_ekf, P_est_ekf = ekf.step(u_true, y_curr)
    # x_est_ukf, P_est_ukf = ukf.step(u_true, y_curr)
    # x_est_samples_pf, weights_pf = pf.step(u_true, y_curr)
    #
    # Make sure to store the estimated state and covariance for analysis
    # as well as the true state and measurement for comparison.

    # Continue the loop
    pass
