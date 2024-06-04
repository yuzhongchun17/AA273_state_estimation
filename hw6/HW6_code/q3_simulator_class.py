##############################
# Simulate the mobile robot
##############################

import numpy as np
import matplotlib.pyplot as plt

class MobileRobotSimulator:
    # A simulator for the non-holonomic mobile robot dynamics and sensor 
    # measurements

    def __init__(self, 
                  q_mat = 0.1 * 0.5 * np.identity(3),
                  r_mat = 0.1 * np.identity(2),
                  dt = 0.5,
                  base_station_locs = np.array([[30, -5], [-10, 20]]),
                  rng_seed = 273):
        """
        Set up the mobile robot simulator. 
        
        Parameters
        ----------
        q_mat : np.ndarray
            The process noise covariance matrix, Q (m^2/s^2, m^2/s^2, rad^2/s^2).
        r_mat : np.ndarray
            The measurement noise covariance matrix, R (m^2, m^2).
        dt : float
            The time step for the simulation (s).
        base_station_locs : np.ndarray
            The locations of the base stations (m, m). [num_stations, 2]
        rng_seed : int
            The random seed for the simulation for reproducibility
        """

        # Check that the matrices are square and positive definite
        assert q_mat.shape[0] == q_mat.shape[1], "Q matrix must be square"
        assert r_mat.shape[0] == r_mat.shape[1], "R matrix must be square"
        assert np.all(np.linalg.eigvals(q_mat) > 0), \
            "Q matrix must be positive definite"
        assert np.all(np.linalg.eigvals(r_mat) > 0), \
            "R matrix must be positive definite"
        
        self.q_mat = q_mat
        self.r_mat = r_mat
        self.dt = dt
        self.rng_seed = rng_seed

        # Check that the base station locations are 2D
        assert base_station_locs.shape[1] == 2, \
            "Base station locations must be 2D"
        assert r_mat.shape[0] == base_station_locs.shape[0], \
            "R matrix must dimensions matching the number of base stations"
        self.base_station_locs = base_station_locs
        self.num_stations = base_station_locs.shape[0]

    def default_control(self, t):
        """
        Default control input for the mobile robot. 
        
        Parameters
        ----------
        t : float
            The current time (s).
        
        Returns
        -------
        u : np.ndarray
            The control input (s, phi) (m/s, rad/s).
        """
        s = 1.0
        phi = np.sin(t)
        return np.array([s, phi])
    
    def noiseless_dynamics_step(self, pose, t, u=None):
        """
        Simulate one step of the mobile robot dynamics without noise.
        
        Parameters
        ----------
        pose : np.ndarray
            The current pose of the mobile robot (x, y, theta) (m, m, rad)
        t : float
            The current time (s)
        u : function handle
            Function that returns the control input at time t
            
        Returns
        -------
        np.ndarray
            The new pose of the mobile robot (x, y, theta) (m, m, rad)
        """
        # Get the current control input. If none is provided, use the default
        if u is None:
            u = self.default_control
        u_curr = u(t)
        # Check the size of the control input
        assert u_curr.shape == (2,), \
            f"Control input must be 2D, got {u_curr.shape}"

        # Check the size of the pose
        assert pose.shape == (3,), \
            f"pose must be 3D (x, y, theta), got {pose.shape}"
        
        # Unpack the pose
        px, py, theta = pose

        # Compute the new pose
        px_new = px + self.dt * u_curr[0] * np.cos(theta)
        py_new = py + self.dt * u_curr[0] * np.sin(theta)
        theta_new = theta + self.dt * u_curr[1]

        return np.array([px_new, py_new, theta_new]), u_curr
    
    def noiseless_measurement_step(self, pose):
        """
        Simulate one step of the mobile robot measurement without noise.
        
        Parameters
        ----------
        pose : np.ndarray
            The current pose of the mobile robot (x, y, theta) (m, m, rad)
        
        Returns
        -------
        np.ndarray
            The measurement of the mobile robot (d1, d2) (m, m)
        """
        # Check the size of the pose
        assert pose.shape == (3,), \
            f"pose must be 3D (x, y, theta), got {pose.shape}"
        
        # Separate the position part of the pose
        p = pose[:2]

        # Compute the measurements (vectorized)
        y_meas = np.linalg.norm(self.base_station_locs - p, axis=1)
        
        # Check the size of the measurements
        assert y_meas.shape == (self.num_stations,), \
            f"Expected {self.num_stations} measurements, got {y_meas.shape}"

        return y_meas
    
    def noisy_dynamics_step(self, pose, t, u=None):
        """
        Simulate one step of the mobile robot dynamics with noise (see 
        self.noiseless_dynamics_step for more details).
        """
        noiseless_pose, noiseless_u = self.noiseless_dynamics_step(pose, t, u)

        # add process noise
        w_noise = np.random.multivariate_normal(np.zeros(3), self.q_mat)

        return noiseless_pose + w_noise, noiseless_u
    
    def noisy_measurement_step(self, pose):
        """
        Simulate one step of the mobile robot measurement with noise (see
        self.noiseless_measurement_step for more details).
        """
        noiseless_measurement = self.noiseless_measurement_step(pose)

        # add measurement noise
        v_noise = np.random.multivariate_normal(np.zeros(self.num_stations), 
                                                self.r_mat)

        return noiseless_measurement + v_noise
    
    def simulate(self, pose0, num_t, rng_seed=None, u=None, noisy=True):
        """
        Simulate the mobile robot dynamics and sensor measurements for `num_t` 
        steps.

        Parameters
        ----------
        pose0 : np.ndarray
            The initial pose of the mobile robot (x, y, theta) (m, m, rad)
        num_t : int
            The number of time steps to simulate
        rng_seed : int
            The random seed to use for the simulation
        u : function handle
            Function that returns the control input at time t (use None for
            the default control given in the problem)
        noisy : bool
            Whether to include noise in the simulation
        
        Returns
        -------
        np.ndarray
            The true pose of the mobile robot at each time step
        np.ndarray
            The sensor measurements at each time step
        np.ndarray
            The control inputs at each time step
        """
        # Set the random seed for reproducibility
        if rng_seed is None:
            np.random.seed(self.rng_seed)
        else:
            np.random.seed(rng_seed)

        # Initialize the histories
        true_poses_history = []
        measurements_history = []
        true_controls_history = []

        # Set the initial pose to the current values
        pose_curr = np.array(pose0)

        for t_ind in range(num_t):
            # Get the current time
            t = t_ind * self.dt

            if noisy:
                pose_curr, u_curr = self.noisy_dynamics_step(
                    pose_curr, t, u)
                y_meas = self.noisy_measurement_step(pose_curr)
            else:
                pose_curr, u_curr = self.noiseless_dynamics_step(
                    pose_curr, t, u)
                y_meas = self.noiseless_measurement_step(pose_curr)

            # Save the current values
            true_poses_history.append(pose_curr)
            measurements_history.append(y_meas)
            true_controls_history.append(u_curr)

        # Return the histories as numpy arrays
        return np.array(true_poses_history), \
               np.array(measurements_history), \
               np.array(true_controls_history)

    def simulate_multiple_runs(self, pose0, num_t, num_runs, 
                               rng_seeds=None, u=None, noisy=True):
        """
        Simulate the mobile robot dynamics and sensor measurements for `num_t` 
        steps for `num_runs` runs using different random seeds.
        
        Parameters
        ----------
        pose0 : np.ndarray
            The initial pose of the mobile robot (x, y, theta) (m, m, rad)
        num_t : int
            The number of time steps to simulate
        num_runs : int
            The number of runs to simulate
        rng_seeds : list
            The random seeds to use for each run
        u : function handle
            Function that returns the control input at time t (use None for
            the default control given in the problem)
        noisy : bool
            Whether to include noise in the simulation
        
        Returns
        -------
        np.ndarray
            The true pose of the mobile robot at each time step for each run
        np.ndarray
            The sensor measurements at each time step for each run
        np.ndarray
            The control inputs at each time step for each run
        """
        # Initialize the histories as zero'ed numpy arrays of the correct size
        all_true_poses = np.zeros((num_runs, num_t, 3))
        all_measurements = np.zeros((num_runs, num_t, self.num_stations))
        all_true_controls = np.zeros((num_runs, num_t, 2))
        
        # Set up the random seeds if not provided
        if rng_seeds is None:
            rng_seeds = np.arange(num_runs)
        
        for run_ind in range(num_runs):
            # Simulate the run
            true_poses, measurements, true_controls = \
                self.simulate(
                    pose0, num_t, 
                    rng_seed=rng_seeds[run_ind], u=u, noisy=noisy)
            
            # Save the results
            all_true_poses[run_ind, :, :] = true_poses
            all_measurements[run_ind, :, :] = measurements
            all_true_controls[run_ind, :, :] = true_controls

        return all_true_poses, all_measurements, all_true_controls
    
    def plot_position_history(self, poses, show_plot=True):
        """
        Plot the position history of the mobile robot. The result is one figure
        showing one rollout. (This is faster than the full pose history plot.)

        Parameters
        ----------
        poses : np.ndarray
            The pose history of the mobile robot (x, y, theta) (m, m, rad)
        show_plot : bool
            Whether to display the plot

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        # Check that we have only one run
        assert poses.ndim == 2, \
            "Data contains multiple runs. Must have 2 dimensions, " \
            f"but got {poses.ndim} dimensions."
        
        # Plot the pose history and the sensor measurements
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("pose History")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # Plot the initial pose with a black star at the highest z-order
        plt.scatter(poses[0, 0], poses[0, 1], 
                    color='black', marker='*', s=100, zorder=3,
                    label="Initial pose")

        # Plot the pose history
        plt.plot(poses[:, 0], poses[:, 1], label="pose")

        # Plot the base stations
        for station_ind in range(self.num_stations):
            plt.scatter(self.base_station_locs[station_ind, 0], 
                        self.base_station_locs[station_ind, 1],
                        label=f"Base Station {station_ind}")

        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()

        return fig
    
    def plot_pose_history(self, poses, show_plot=True):
        """
        Plot the pose history of the mobile robot. The result is one figure
        showing one rollout. (This is slower than the position history plot.)

        Parameters
        ----------
        poses : np.ndarray
            The pose history of the mobile robot (x, y, theta) (m, m, rad)
        show_plot : bool
            Whether to display the plot

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        # Check that we have only one run
        assert poses.ndim == 2, \
            "Data contains multiple runs. Must have 2 dimensions, " \
            f"but got {poses.ndim} dimensions."
        
        # Plot the pose history and the sensor measurements
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("Pose History")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        theta_offset = -90  # Offset for the triangle marker

        # Plot the initial pose with a black star at the highest z-order
        plt.scatter(poses[0, 0], poses[0, 1], 
                    color='black', s=100, zorder=3,
                    marker=(3, 0, theta_offset + np.rad2deg(poses[0, 2])),
                    label="Initial pose")
        
        # Plot a line going through the pose history
        plt.plot(poses[:, 0], poses[:, 1], color='cyan', label="Position History")

        # Plot the pose history
        labeled = False
        for px, py, theta in poses:
            plt.scatter(px, py, 
                        edgecolors='blue', facecolors='none',
                        s=50, zorder=2,
                        marker=(3, 0, theta_offset + np.rad2deg(theta)),
                        label="pose" if not labeled else None)
            labeled = True

        # Plot the base stations
        for station_ind in range(self.num_stations):
            plt.scatter(self.base_station_locs[station_ind, 0], 
                        self.base_station_locs[station_ind, 1],
                        label=f"Base Station {station_ind}")
            
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()

        return fig
    
    def plot_position_histories(self, all_poses, show_plot=True):
        """
        Plot the position histories for each run. The result is one figure 
        showing multiple rollouts.
        
        Parameters
        ----------
        all_poses : np.ndarray
            The pose of the mobile robot at each time step for each run.
            If multiple runs, the shape is (num_runs, num_t, 3).
        measurement_size : int
            The size of the scatter points for the measurements
        show_plot : bool
            Whether to display the plot
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        # Check that we have multiple runs
        assert all_poses.ndim == 3, \
            "Data does not contain multiple runs. Must have 3 dimensions, " \
            f"but got {all_poses.ndim} dimensions."
        num_runs = all_poses.shape[0]
        
        # Plot the position histories and the sensor measurements
        fig, _ = plt.subplots(figsize=(10, 6))
        plt.title("Position Histories")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # Plot the initial position with a black star at the highest z-order
        # We assume that all runs start at the same initial position
        plt.scatter(all_poses[0, 0, 0], all_poses[0, 0, 1], 
                    color='black', marker='*', s=100, zorder=3,
                    label="Initial Position")

        # The positions will be in solid lines and the measurements are scatter
        # points of the same color
        for run_ind in range(num_runs):
            # Plot the position history
            positions = all_poses[run_ind, :, :]
            plt.plot(positions[:, 0], positions[:, 1], 
                     label=f"Position for Run {run_ind}")
            
        # Plot the base stations
        for station_ind in range(self.num_stations):
            plt.scatter(self.base_station_locs[station_ind, 0], 
                        self.base_station_locs[station_ind, 1],
                        label=f"Base Station {station_ind}")
        
        # Place the legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()

        return fig
    
    def plot_control_history(self, controls, show_plot=True):
        """
        Plot the control history. The result is one figure showing one rollout.
        
        Parameters
        ----------
        controls : np.ndarray
            The control input at each time step.
        show_plot : bool
            Whether to display the plot
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        # Check that we have only one run
        assert controls.ndim == 2, \
            "Data contains multiple runs. Must have 2 dimensions, " \
            f"but got {controls.ndim} dimensions."
        
        # Plot the control history
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("Control History")
        plt.xlabel("Time (s)")
        plt.ylabel("s (m/s), phi (rad/s)")

        # Plot the control history
        plt.plot(controls[:, 0], label="Speed Input")
        plt.plot(controls[:, 1], label="Rotation Rate Input")
        
        plt.legend()
        plt.grid(True)
        # plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()
        
        return fig
    
    def plot_measurement_history(self, measurements, show_plot=True):
        """
        Plot the measurement history. The result is one figure showing one 
        rollout.
        
        Parameters
        ----------
        measurements : np.ndarray
            The sensor measurements at each time step.
        show_plot : bool
            Whether to display the plot
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        # Check that we have only one run
        assert measurements.ndim == 2, \
            "Data contains multiple runs. Must have 2 dimensions, " \
            f"but got {measurements.ndim} dimensions."
        
        # Plot the measurement history
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("Measurement History")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance to Base Station (m)")

        # Plot the measurement history
        for station_ind in range(self.num_stations):
            plt.plot(measurements[:, station_ind], 
                     label=f"Distance to Base Station {station_ind}")
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if show_plot:
            plt.show()
        
        return fig
    
    def plot_measurement_histories(self, all_measurements, show_plot=True):
        """
        Plot the measurement histories for each run. The result is one figure 
        showing multiple rollouts.
        
        Parameters
        ----------
        all_measurements : np.ndarray
            The sensor measurements at each time step for each run.
            If multiple runs, the shape is (num_runs, num_t, num_stations).
        show_plot : bool
            Whether to display the plot
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        # Check that we have multiple runs
        assert all_measurements.ndim == 3, \
            "Data does not contain multiple runs. Must have 3 dimensions, " \
            f"but got {all_measurements.ndim} dimensions."
        num_runs = all_measurements.shape[0]
        
        # Plot the measurement histories
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title("Measurement Histories")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance to Base Station (m)")

        colors = plt.get_cmap('tab10')

        # The measurements are in solid lines
        for run_ind in range(num_runs):
            # Plot the measurement history
            measurements = all_measurements[run_ind, :, :]

            # Use the same color, but different line styles for each station
            line_styles = ['-', '--', '-.', ':']
            for station_ind in range(self.num_stations):
                plt.plot(measurements[:, station_ind], 
                         label="Measurement to Base Station " + \
                            f"{station_ind} for Run {run_ind}",
                         linestyle=line_styles[station_ind % len(line_styles)],
                         color=colors(run_ind))
        
        # Place the legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        if show_plot:
            plt.show()

        return fig

    def plot_2D_errors(self, true_vals, comparison_vals,
                       err_cov=None,
                       title="2D Errors",
                       x_label="Error in x (m)",
                       y_label="Error in y (m)",
                       measurement_size=10, show_plot=True):
        """
        Plot the 2D errors between the true values and the comparison values.
        
        Parameters
        ----------
        true_vals : np.ndarray
            The true values to compare against.
        comparison_vals : np.ndarray
            The comparison values to plot the errors.
        err_cov : np.ndarray
            The error covariance matrix for the comparison values.
        title : str
            The title of the plot
        measurement_size : int
            The size of the scatter points for the measurements
        show_plot : bool
            Whether to display the plot
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        # Check that the dimensions match
        assert true_vals.shape == comparison_vals.shape, \
            "True and comparison values must have the same shape." + \
            f"Got True values with shape {true_vals.shape} and " + \
                f"Comparison values with shape {comparison_vals.shape}."
        
        # Compute the errors
        errors = comparison_vals - true_vals
        
        # Plot the errors
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # If we have an error covariance, plot the error ellipse (note that
        # we only plot axis-aligned ellipses, students can modify this to
        # plot more general error ellipse with their code from HW2)
        if err_cov is not None:
            assert err_cov.shape == (2, 2), \
                "Error covariance must be a 2x2 matrix."
            assert np.isclose(err_cov[1, 0], 0.0), \
                f"Error covariance must be diagonal, but got {err_cov}." + \
                "Students can modify this to plot general error ellipses."

            # Compute the error ellipse for 1 and 2 standard deviations
            for std_dev, std_ls in zip([1, 2], ['--', ':']):
                width = 2 * std_dev * np.sqrt(err_cov[0, 0])
                height = 2 * std_dev * np.sqrt(err_cov[1, 1])

                error_ellipse = plt.matplotlib.patches.Ellipse(
                    xy=[0, 0], width=width, height=height,
                    edgecolor='black', facecolor='none', linestyle=std_ls,
                    label=f"{std_dev} Std Dev Error Ellipse")
                ax.add_patch(error_ellipse)

        # Plot the errors
        plt.scatter(errors[:, 0], errors[:, 1],
                    s=measurement_size, marker='x', color='red',
                    label="Errors")
        
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()
        
        return fig
