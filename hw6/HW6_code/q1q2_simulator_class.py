###########################
# Simulate the quadrotor
###########################

import numpy as np
import matplotlib.pyplot as plt

class QuadrotorSimulator:
    # A simulator for the quadrotor dynamics and sensor measurements

    def __init__(self, sensor,
                 q_mat= np.identity(2), # process noise
                 r_mat = 9 * np.identity(2), # measurement noise covariance matrix
                 dt = 1,
                 rng_seed = 273):
        """
        Set up the quadrotor simulator. 

        Parameters
        ----------
        sensor : str
            Selection of "GPS" or "Velocity" sensor
        q_mat : np.ndarray
            The process noise covariance matrix, Q (m^2/s^2)
        r_mat : np.ndarray
            The measurement noise covariance matrix, R (m^2)
        dt : float
            The time step for the simulation (s)
        rng_seed : int
            The random seed for the simulation for reproducibility
        """
        assert sensor in ["GPS", "Velocity"], \
            "Sensor must be 'GPS' or 'Velocity'"
        self.sensor = sensor

        # Check that the matrices are square and positive definite.
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

    def default_control(self, t):
        """
        This is the quadrotor control defined in the problem (units are
        m/s^2).

        Parameters
        ----------
        t : float
            The current time (s)

        Returns
        -------
        np.ndarray
            The control input
        """
        return -2.5 * np.array([np.cos(0.05 * t), np.sin(0.05 * t)])
    
    def noiseless_dynamics_step(self, p, s, t, u=None):
        """
        Simulate one step of the quadrotor dynamics without noise.

        Parameters
        ----------
        p : np.ndarray
            The current position of the quadrotor (m)
        s : np.ndarray
            The current velocity of the quadrotor (m/s)
        t : float
            The current time (s)
        u : function handle
            Function that returns the control input at time t

        Returns
        -------
        np.ndarray
            The new position of the quadrotor
        np.ndarray
            The new velocity of the quadrotor
        """
        # Get the current control input. If none is provided, use the default
        if u is None:
            u = self.default_control
        u_curr = u(t)
        # Check the size of the control input
        assert u_curr.shape == (2,), \
            f"Control input must be 2D, got {u_curr.shape}"

        # Check the size of the position and velocity
        assert p.shape == (2,), f"Position must be 2D, got {p.shape}"
        assert s.shape == (2,), f"Velocity must be 2D, got {s.shape}"

        # Compute the new position and velocity
        p_new = p + self.dt * s
        v_new = s + self.dt * u_curr

        return p_new, v_new, u_curr
    
    def noiseless_gps_step(self, p):
        """
        Simulate one step of the GPS sensor measurement without noise.

        Parameters
        ----------
        p : np.ndarray
            The current position of the quadrotor (m)

        Returns
        -------
        np.ndarray
            The sensor measurement
        """
        # The noiseless GPS sensor measurement is the quadrotor position
        return p
    
    def noiseless_velocity_step(self, s):
        """
        Simulate one step of the velocity sensor measurement without noise.

        Parameters
        ----------
        s : np.ndarray
            The current velocity of the quadrotor (m/s)

        Returns
        -------
        np.ndarray
            The sensor measurement
        """
        # The noiseless velocity sensor measurement is the quadrotor velocity
        return s
    
    def noiseless_measurement_step(self, p, s):
        """
        Select the appropriate sensor measurement based on the sensor type.
        """
        # Check the size of the position and velocity
        assert p.shape == (2,), "Position must be 2D."
        assert s.shape == (2,), "Velocity must be 2D."

        if self.sensor == "GPS":
            return self.noiseless_gps_step(p)
        elif self.sensor == "Velocity":
            return self.noiseless_velocity_step(s)
        else:
            raise NotImplementedError(f"{self.sensor} sensor is not supported.")
        
    def noisy_dynamics_step(self, p, s, t, u=None):
        """
        Simulate one step of the quadrotor dynamics with noise (see 
        self.noiseless_dynamics_step for more details).
        """
        noiseless_p, noiseless_s, noiseless_u = \
            self.noiseless_dynamics_step(p, s, t, u)

        # Add process noise (only to the velocity)
        w_noise = np.random.multivariate_normal(np.zeros(2), self.q_mat)

        return noiseless_p, noiseless_s + w_noise, noiseless_u
    
    def noisy_gps_step(self, p):
        """
        Simulate one step of the GPS sensor measurement with noise (see
        self.noiseless_gps_step for more details).
        """
        noiseless_gps = self.noiseless_gps_step(p)

        # Add measurement noise (to the position measurement) 
        v_noise = np.random.multivariate_normal(np.zeros(2), self.r_mat)

        return noiseless_gps + v_noise
    
    def noisy_velocity_step(self, s):
        """
        Simulate one step of the velocity sensor measurement with noise (see
        self.noiseless_velocity_step for more details).
        """
        noiseless_velocity = self.noiseless_velocity_step(s)

        # Add measurement noise (to the velocity measurement)
        v_noise = np.random.multivariate_normal(np.zeros(2), self.r_mat)

        return noiseless_velocity + v_noise
    
    def noisy_measurement_step(self, p, s):
        """
        Select the appropriate sensor measurement based on the sensor type.
        """
        # Check the size of the position and velocity
        assert p.shape == (2,), "Position must be 2D."
        assert s.shape == (2,), "Velocity must be 2D."

        if self.sensor == "GPS":
            return self.noisy_gps_step(p)
        elif self.sensor == "Velocity":
            return self.noisy_velocity_step(s)
        else:
            raise NotImplementedError(f"{self.sensor} sensor is not supported.")
        
    def simulate(self, p0, s0, num_t, rng_seed=None, u=None, noisy=True):
        """
        Simulate the quadrotor dynamics and sensor measurements for `num_t` 
        steps.
        
        Parameters
        ----------
        p0 : np.ndarray
            The initial position of the quadrotor (m)
        s0 : np.ndarray
            The initial velocity of the quadrotor (m/s)
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
            The true position of the quadrotor at each time step
        np.ndarray
            The true velocity of the quadrotor at each time step
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
        true_positions_history = []
        true_velocities_history = []
        measurements_history = []
        true_controls_history = []

        # Set the initial position and velocity to the current values
        p_curr = np.array(p0)
        s_curr = np.array(s0)

        for t_ind in range(num_t):
            # Get the current time
            t = t_ind * self.dt

            if noisy:
                p_curr, s_curr, u_curr = self.noisy_dynamics_step(
                    p_curr, s_curr, t, u)
                y_curr = self.noisy_measurement_step(p_curr, s_curr)
            else:
                p_curr, s_curr, u_curr = self.noiseless_dynamics_step(
                    p_curr, s_curr, t, u)
                y_curr = self.noiseless_measurement_step(p_curr, s_curr)
            
            # Save the current values
            true_positions_history.append(p_curr)
            true_velocities_history.append(s_curr)
            measurements_history.append(y_curr)
            true_controls_history.append(u_curr)

        # Return the histories as numpy arrays
        return np.array(true_positions_history), \
               np.array(true_velocities_history), \
               np.array(measurements_history), \
               np.array(true_controls_history) 
    
    def simulate_multiple_runs(self, p0, s0, num_t, num_runs, 
                               rng_seeds=None, u=None, noisy=True):
        """
        Simulate the quadrotor dynamics and sensor measurements for `num_t` 
        steps for `num_runs` runs using different random seeds.
        
        Parameters
        ----------
        p0 : np.ndarray
            The initial position of the quadrotor (m)
        s0 : np.ndarray
            The initial velocity of the quadrotor (m/s)
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
            The true position of the quadrotor at each time step for each run
        np.ndarray
            The true velocity of the quadrotor at each time step for each run
        np.ndarray
            The sensor measurements at each time step for each run
        np.ndarray
            The control inputs at each time step for each run
        """
        # Initialize the histories as zero'ed numpy arrays of the correct size
        all_true_positions = np.zeros((num_runs, num_t, 2))
        all_true_velocities = np.zeros((num_runs, num_t, 2))
        all_measurements = np.zeros((num_runs, num_t, 2))
        all_true_controls = np.zeros((num_runs, num_t, 2))
        
        # Set up the random seeds if not provided
        if rng_seeds is None:
            rng_seeds = np.arange(num_runs)
        
        for run_ind in range(num_runs):
            # Simulate the run
            true_positions, true_velocities, measurements, true_controls = \
                self.simulate(
                    p0, s0, num_t, 
                    rng_seed=rng_seeds[run_ind], u=u, noisy=noisy)
            
            # Save the results
            all_true_positions[run_ind, :, :] = true_positions
            all_true_velocities[run_ind, :, :] = true_velocities
            all_measurements[run_ind, :, :] = measurements
            all_true_controls[run_ind, :, :] = true_controls

        return all_true_positions, all_true_velocities, all_measurements, \
                all_true_controls
    
    def plot_position_history(self, positions, measurements=None,
                              measurement_size=10, show_plot=True):
        """
        Plot the position history and the sensor measurements (if "GPS"). 
        The result is one figure showing one rollout. 
        
        Parameters
        ----------
        positions : np.ndarray
            The position of the quadrotor at each time step.
        measurements : np.ndarray
            The sensor measurements at each time step.
            Set to None to only plot the position history.
        measurement_size : int
            The size of the scatter points for the measurements
        show_plot : bool
            Whether to display the plot
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        # Check that we have only one run
        assert positions.ndim == 2, \
            "Data contains multiple runs. Must have 2 dimensions, " \
            f"but got {positions.ndim} dimensions."
        
        # Plot the position history and the sensor measurements
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("Position History")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # Plot the initial position with a black star at the highest z-order
        plt.scatter(positions[0, 0], positions[0, 1], 
                    color='black', marker='*', s=100, zorder=3,
                    label="Initial Position")

        # Plot the position history
        plt.plot(positions[:, 0], positions[:, 1], label="Position")
        
        # Reset the color cycle for the measurements
        ax.set_prop_cycle(None)
        # Plot the sensor measurements if using GPS
        if self.sensor == "GPS" and measurements is not None:
            plt.scatter(measurements[:, 0], measurements[:, 1], 
                        s=measurement_size,
                        label="GPS Measurements")
        
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()

        return fig

    
    def plot_position_histories(self, all_positions, all_measurements=None,
                                measurement_size=10, show_plot=True):
        """
        Plot the position histories and the sensor measurements for each 
        run (if "GPS"). The result is one figure showing multiple rollouts.
        
        Parameters
        ----------
        all_positions : np.ndarray
            The position of the quadrotor at each time step for each run.
            If multiple runs, the shape is (num_runs, num_t, 2).
        all_measurements : np.ndarray
            The sensor measurements at each time step for each run.
            If multiple runs, the shape is (num_runs, num_t, 2).
            Set to None to only plot the position histories.
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
        assert all_positions.ndim == 3, \
            "Data does not contain multiple runs. Must have 3 dimensions, " \
            f"but got {all_positions.ndim} dimensions."
        num_runs = all_positions.shape[0]
        
        # Plot the position histories and the sensor measurements
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title("Position Histories")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # Plot the initial position with a black star at the highest z-order
        # We assume that all runs start at the same initial position
        plt.scatter(all_positions[0, 0, 0], all_positions[0, 0, 1], 
                    color='black', marker='*', s=100, zorder=3,
                    label="Initial Position")

        # The positions will be in solid lines and the measurements are scatter
        # points of the same color
        for run_ind in range(num_runs):
            # Plot the position history
            positions = all_positions[run_ind, :, :]
            plt.plot(positions[:, 0], positions[:, 1], 
                     label=f"Position for Run {run_ind}")
        
        # Reset the color cycle for the measurements
        ax.set_prop_cycle(None)
        # Plot the sensor measurements if using GPS
        if self.sensor == "GPS" and all_measurements is not None:
            for run_ind in range(num_runs):
                measurements = all_measurements[run_ind, :, :]
                plt.scatter(measurements[:, 0], measurements[:, 1], 
                            s=measurement_size,
                            label=f"GPS Measurements for Run {run_ind}")
        
        # Place the legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()

        return fig

    def plot_velocity_history(self, velocities, measurements=None,
                              measurement_size=10, show_plot=True):
        """
        Plot the velocity history and the sensor measurements (if 
        "Velocity"). The result is one figure showing one rollout.

        Parameters
        ----------
        velocities : np.ndarray
            The velocity of the quadrotor at each time step.
        measurements : np.ndarray
            The sensor measurements at each time step.
            Set to None to only plot the velocity history.
        measurement_size : int
            The size of the scatter points for the measurements
        show_plot : bool
            Whether to display the plot
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot
        """
        # Check that we have only one run
        assert velocities.ndim == 2, \
            "Data contains multiple runs. Must have 2 dimensions, " \
            f"but got {velocities.ndim} dimensions."
        
        # Plot the velocity history and the sensor measurements
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("Velocity History")
        plt.xlabel("$v_x$ (m/s)")
        plt.ylabel("$v_y$ (m/s)")

        # Plot the initial velocity with a black star at the highest z-order
        plt.scatter(velocities[0, 0], velocities[0, 1], 
                    color='black', marker='*', s=100, zorder=3,
                    label="Initial Velocity")

        # Plot the velocity history
        plt.plot(velocities[:, 0], velocities[:, 1], label="Velocity")

        # Reset the color cycle for the measurements
        ax.set_prop_cycle(None)
        # Plot the sensor measurements if using Velocity
        if self.sensor == "Velocity" and measurements is not None:
            plt.scatter(measurements[:, 0], measurements[:, 1], 
                        s=measurement_size,
                        label="Velocity Measurements")
        
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        if show_plot:
            plt.show()
        
        return fig
    
    def plot_velocity_histories(self, all_velocities, all_measurements=None,
                                measurement_size=10, show_plot=True):
        """
        Plot the velocity histories and the sensor measurements for each 
        run (if "Velocity"). The result is one figure showing multiple rollouts.
        
        Parameters
        ----------
        all_velocities : np.ndarray
            The velocity of the quadrotor at each time step for each run.
            If multiple runs, the shape is (num_runs, num_t, 2).
        all_measurements : np.ndarray
            The sensor measurements at each time step for each run.
            If multiple runs, the shape is (num_runs, num_t, 2).
            Set to None to only plot the velocity histories.
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
        assert all_velocities.ndim == 3, \
            "Data does not contain multiple runs. Must have 3 dimensions, " \
            f"but got {all_velocities.ndim} dimensions."
        num_runs = all_velocities.shape[0]
        
        # Plot the velocity histories and the sensor measurements
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title("Velocity Histories")
        plt.xlabel("$v_x$ (m/s)")
        plt.ylabel("$v_y$ (m/s)")

        # Plot the initial velocity with a black star at the highest z-order
        # We assume that all runs start at the same initial velocity
        plt.scatter(all_velocities[0, 0, 0], all_velocities[0, 0, 1], 
                    color='black', marker='*', s=100, zorder=3,
                    label="Initial Velocity")

        # The velocities will be in solid lines and the measurements are scatter
        # points of the same color
        for run_ind in range(num_runs):
            # Plot the velocity history
            velocities = all_velocities[run_ind, :, :]
            plt.plot(velocities[:, 0], velocities[:, 1], 
                     label=f"Velocity for Run {run_ind}")
        
        # Reset the color cycle for the measurements
        ax.set_prop_cycle(None)
        # Plot the sensor measurements if using Velocity
        if self.sensor == "Velocity" and all_measurements is not None:
            for run_ind in range(num_runs):
                measurements = all_measurements[run_ind, :, :]
                plt.scatter(measurements[:, 0], measurements[:, 1], 
                            s=measurement_size,
                            label=f"Velocity Measurements for Run {run_ind}")
            
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
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("Control History")
        plt.xlabel("$u_x$ (m/s$^2$)")
        plt.ylabel("$u_y$ (m/s$^2$)")

        # Plot the initial control with a black star at the highest z-order
        plt.scatter(controls[0, 0], controls[0, 1], 
                    color='black', marker='*', s=100, zorder=3,
                    label="Initial Control")

        # Plot the control history
        plt.plot(controls[:, 0], controls[:, 1], label="Control")
        
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
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
            "True and comparison values must have the same shape."
        
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
