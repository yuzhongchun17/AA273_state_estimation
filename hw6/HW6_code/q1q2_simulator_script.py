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
quadrotor = sim.QuadrotorSimulator(sensor="GPS")

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

u_hist_to_replace = np.zeros((n_steps, 2))  # Replace with actual controls
y_hist_to_replace = np.zeros((n_steps, 2))  # Replace with actual measurements

for t_index, (u, y) in enumerate(zip(u_hist_to_replace, y_hist_to_replace)):
    # Use (u, y) to step your Kalman Filter (note that you will run)
    # For example:
    # x_est, P_est = kalman_filter.step(u, y)
    #
    # Or if you want to be more sequential:
    # x_est, P_est = kalman_filter.predict(u)
    # x_est, P_est = kalman_filter.update(y)
    #
    # Make sure to store the estimated state and covariance for analysis

    # Continue the loop
    pass

# If you are using the second method, you can run the simulator and Kalman
# Filter in lockstep.

for t_index in range(n_steps):
    # Get the true state from a simulator dynamics step.
    # i.e., p_true, s_true, u_true = quadrotor.noisy_dynamics_step(
    #              p_true, s_true, t_index * quadrotor.dt)
    # Then, use the true state to generate a measurement.
    # i.e., y_curr = quadrotor.noisy_measurement_step(p_true, s_true)

    # Use (u_true, y_curr) to step your Kalman Filter, but it should never see 
    # the true state (p_true, s_true).
    # For example:
    # x_est, P_est = kalman_filter.step(u_true, y_curr)
    #
    # Or if you want to be more sequential:
    # x_est, P_est = kalman_filter.predict(u_true)
    # x_est, P_est = kalman_filter.update(y_curr)
    #
    # Make sure to store the estimated state and covariance for analysis
    # as well as the true state and measurement for comparison.

    # Continue the loop
    pass

# %% Question 2

# Initialize the simulator
quadrotor = sim.QuadrotorSimulator(sensor="Velocity")

# You can use the same functions as above to run and visualize the simulation.
