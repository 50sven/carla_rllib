"""Polynomial Trajectory Planning

This script allows the user to calculate polynomial trajectories based on jerk minimization.
The trajectories are planned in frenet coordinates.

Classes:
    * Trajectory - basic trajectory class
    * PolynomialGenerator - trajectory calculation based on jerk minimization
"""
import numpy as np


class Trajectory(object):

    def __init__(self, deltaT, dt, t_steps):
        # Trajectory variables
        self.deltaT = deltaT
        self.dt = dt
        self.t_steps = t_steps
        self.ref_vector = np.array([1.0, 0.0])
        self.s_coordinates = []
        self.d_coordinates = []
        self.s_velocity = []
        self.d_velocity = []
        self.s_acceleration = []
        self.d_acceleration = []

        # Action validation
        self.wheelbase = 2.2
        self.max_steering_angle = 0.22  # rad
        self.max_acceleration = 9.81
        # TODO: action validation
        self.curvature = []
        self.heading = []
        self.steering_angle = []
        self.total_acceleration = []
        self.invalid_action = False

    def _get_unit_vector(self, v):
        """Returns the unit vector of the vector"""
        return v / np.linalg.norm(v)

    def get_angle(self, t_step, degree=True):
        """Calculates the angle between two vectors"""
        v_vector = np.array([self.s_velocity[t_step], self.d_velocity[t_step]])
        u_v1 = self._get_unit_vector(v_vector)
        rad = np.arccos(np.clip(np.dot(self.ref_vector, u_v1), -1.0, 1.0))
        if degree:
            return np.rad2deg(rad) * np.sign(self.d_velocity[t_step])
        else:
            return rad


class PolynomialGenerator(object):

    def __init__(self, deltaT, dt, fraction=1.0):
        self.deltaT = deltaT
        self.dt = dt
        self.t_steps = int(np.around(self.deltaT / self.dt * fraction)) + 1

        # Preliminary calculation of pre-factors
        self.helper_p = np.array([[1, i, i**2, i**3, i**4, i**5]
                                  for i in np.arange(0, self.deltaT + self.dt, self.dt)])
        self.helper_v = np.array([[0, 1, 2 * i, 3 * i**2, 4 * i**3, 5 * i**4]
                                  for i in np.arange(0, self.deltaT + self.dt, self.dt)])
        self.helper_a = np.array([[0, 0, 2, 6 * i, 12 * i**2, 20 * i**3]
                                  for i in np.arange(0, self.deltaT + self.dt, self.dt)])

    def calculate_coefficients(self, start, end):
        """Solves the polynomial and returns the coefficients

        Parameters
        ----------
        start: array-like (list, triple, numpy-array)
            start conditions consisting of position, velocity and acceleration
        end: array-like (list, triple, numpy-array)
            end conditions consisting of position, velocity and acceleration

        Returns
        ----------
        coeff: numpy-array
            coefficients (a0-a5) of the polynomial trajectory that minimizes the jerk
        """
        A = np.array([
            [self.deltaT**3, self.deltaT**4, self.deltaT**5],
            [3 * self.deltaT**2, 4 * self.deltaT**3, 5 * self.deltaT**4],
            [6 * self.deltaT, 12 * self.deltaT**2, 20 * self.deltaT**3],
        ])

        a_0, a_1, a_2 = start[0], start[1], start[2] / 2.0
        c_0 = a_0 + a_1 * self.deltaT + a_2 * self.deltaT**2
        c_1 = a_1 + 2 * a_2 * self.deltaT
        c_2 = 2 * a_2

        B = np.array([
            end[0] - c_0,
            end[1] - c_1,
            end[2] - c_2
        ])

        a_3_4_5 = np.linalg.solve(A, B)
        coeff = np.concatenate((np.array([a_0, a_1, a_2]), a_3_4_5))

        return coeff

    def calculate_trajectory(self, position, velocity, acceleration,
                             deltaV, deltaD):
        """Returns the trajectory in frenet coordinates

        Parameters
        ----------
        position: array-like (s, d)
        velocity: array-like (vs, vd)
        acceleration: array-like (as, ad)
        deltaV: float
            Change of velocity
        deltaD: float
            Change of lateral position

        Returns
        ----------
        trajectory: Trajectory class
            see class above
        """
        # Initial/End conditions
        initS = (position[0], velocity[0], acceleration[0])
        initD = (position[1], velocity[1], acceleration[1])

        # TODO: find proper boundary condition for dS
        dS = ((2 * initS[1] + deltaV) / 2) * self.deltaT

        endS = (initS[0] + dS, initS[1] + deltaV, 0)
        endD = (initD[0] + deltaD, 0, 0)

        # Calculate coefficients
        coeff_S = self.calculate_coefficients(initS, endS)
        coeff_D = self.calculate_coefficients(initD, endD)

        # Calculate trajectory
        trajectory = self._get_trajectory(coeff_S, coeff_D)

        return trajectory

    def _get_trajectory(self, coeff_S, coeff_D):
        """Returns the trajectory calculated with given coefficients"""
        # Create trajectory
        trajectory = Trajectory(self.deltaT, self.dt, self.t_steps)
        coeff = np.array([coeff_S, coeff_D]).T
        p_array = self.helper_p.dot(coeff)
        v_array = self.helper_v.dot(coeff)
        a_array = self.helper_a.dot(coeff)

        # Truncate trajectory
        trajectory.s_coordinates = p_array[:self.t_steps, 0]
        trajectory.d_coordinates = p_array[:self.t_steps, 1]
        trajectory.s_velocity = v_array[:self.t_steps, 0]
        trajectory.d_velocity = v_array[:self.t_steps, 1]
        trajectory.s_acceleration = a_array[:self.t_steps, 0]
        trajectory.d_acceleration = a_array[:self.t_steps, 1]

        return trajectory

    def validate_trajectory(self):
        """Checks the trajectory for invalid actions"""
        raise NotImplementedError

    def transform_trajectory(self, trajectory, theta):
        """Rotates the trajectory by a given angle"""
        origin = np.array([trajectory.s_coordinates[0],
                           trajectory.d_coordinates[0]])

        for step in range(self.t_steps):
            point = np.array([trajectory.s_coordinates[step],
                              trajectory.d_coordinates[step]])
            point = self._rotate_vector(point - origin, theta) + origin
            trajectory.s_coordinates[step] = point[0]
            trajectory.d_coordinates[step] = point[1]

        return trajectory

    def _rotate_vector(self, v, theta):
        """Rotates a 2D-vector"""
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return R.dot(v)


if __name__ == "__main__":

    T, t, fraction = 1.4, 0.05, 0.5
    s, d = 47.7, 208.8
    vs, vd = 10.0, 0.0
    a_s, ad = 0.0, 0.0

    # Single trajectory
    # generator = PolynomialGenerator(T, t, fraction)

    # deltaV, deltaD = 5.0, 2.0
    # trajectory = generator.calculate_trajectory((s, d), (vs, vd), (a_s, ad),
    #                                             deltaV, deltaD)

    # s_list = trajectory.s_coordinates
    # d_list = trajectory.d_coordinates
    # vs_list = trajectory.s_velocity
    # vd_list = trajectory.d_velocity
    # as_list = trajectory.s_acceleration
    # ad_list = trajectory.d_acceleration

    # Multiple partial trajectories
    generator = PolynomialGenerator(T, t, fraction)
    deltas = [(5, 1.0), (4, 1.0), (3, -0.5), (-2, -0.5), (-1, 0), (0, 0)]

    s_list = np.array([s])
    d_list = np.array([d])
    vs_list = np.array([vs])
    vd_list = np.array([vd])
    as_list = np.array([a_s])
    ad_list = np.array([ad])

    for delta in deltas:

        deltaV, deltaD = delta[0], delta[1]

        trajectory = generator.calculate_trajectory((s, d), (vs, vd), (a_s, ad),
                                                    deltaV, deltaD)

        s_list = np.concatenate((s_list, trajectory.s_coordinates[1:]))
        d_list = np.concatenate((d_list, trajectory.d_coordinates[1:]))
        vs_list = np.concatenate((vs_list, trajectory.s_velocity[1:]))
        vd_list = np.concatenate((vd_list, trajectory.d_velocity[1:]))
        as_list = np.concatenate((as_list, trajectory.s_acceleration[1:]))
        ad_list = np.concatenate((ad_list, trajectory.d_acceleration[1:]))

        s, d = trajectory.s_coordinates[-1], trajectory.d_coordinates[-1]
        vs, vd = trajectory.s_velocity[-1], trajectory.d_velocity[-1]
        a_s, ad = trajectory.s_acceleration[-1], trajectory.d_acceleration[-1]

    import matplotlib.pyplot as plt

    # s
    plt.subplot(3, 1, 1)
    plt.plot(s_list, d_list, '-')
    plt.xlabel('s')
    plt.ylabel('d')

    # v
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, len(vs_list)), vs_list, '-')

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, len(vd_list)), vd_list, '-')
    plt.ylabel('v')

    plt.gca().legend(('s', 'd'), bbox_to_anchor=(1, 0.25))

    # a
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(0, len(as_list)), as_list, '-')

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(0, len(ad_list)), ad_list, '-')
    plt.xlabel('time')
    plt.ylabel('a')

    plt.show()
