"""CARLA Trajectory Environment

This environment is specific to a particular application and
does not represent a universal blueprint for different tasks!
It is designed for multi-agent scenarios but also works with single agent settings.

Classes:
    * TrajectoryWrapper - wrapper class for the discrete following of a trajectory
    * ObservationWrapper - wrapper class to handle discrete state information
    * RewardWrapper - wrapper class to handle cooperative multi-agent rewards
Functions:
    * make_env - creates the environment
"""
import gym
import carla
import numpy as np
from carla_rllib.utils.rendering import OpenDriveRenderer
from carla_rllib.utils.trajectory_planning import PolynomialGenerator
from carla_rllib.environments.carla_envs.base_env import BaseEnv
from carla_rllib.environments.carla_envs.env_wrapper import MultiAgentWrapper, ObstaclesWrapper


class TrajectoryWrapper(gym.Wrapper):

    def __init__(self, env, deltaT=1.0, dt=0.05, fraction=0.25):
        super(TrajectoryWrapper, self).__init__(env)
        self.renderer = {agent.id: OpenDriveRenderer(self.map, self.scenario, agent)
                         for agent in self.agents}
        self.generator = PolynomialGenerator(deltaT=deltaT, dt=dt, fraction=fraction)

    def step(self, action):
        """Executes each discrete timestep of a trajectory

        action = [deltaV, deltaL]
        """

        # Transform 2-dim action into trajectory
        action = self.get_trajectory(action)

        # Initialize buffers
        sub_action = dict.fromkeys(action.keys(), None)
        reward_buffer = dict.fromkeys(action.keys(), None)
        previous_state = dict.fromkeys(action.keys(), None)
        done_buffer = dict.fromkeys(action.keys(), 0)
        # Prepare state
        for idx, agent in enumerate(self.agents):
            self.render_dummy[idx] = np.copy(agent.state.image)
            agent.state.image[0] = agent.state.image[2]
            agent.state.scalars[:6] = agent.state.scalars[12:]

        t_steps = action["Agent_1"].t_steps

        for step in range(1, t_steps):
            # Prepare the action for each agent
            for agent in self.agents:
                # If the agent is dead, don't move him
                if done_buffer[agent.id] != 1:
                    traj = action[agent.id]
                    yaw = traj.get_angle(step)
                    sub_action[agent.id] = (traj.s_coordinates[step], traj.d_coordinates[step],
                                            traj.s_velocity[step], traj.d_velocity[step],
                                            traj.s_acceleration[step], traj.d_acceleration[step],
                                            agent.state.direction_heading + yaw)
                    previous_state[agent.id] = (agent.state.position[0], agent.state.position[1],
                                                agent.state.xy_velocity[0], agent.state.xy_velocity[1],
                                                agent.state.xy_acceleration[0], agent.state.xy_acceleration[1],
                                                agent.state.rotation)

            # Execute the sub_actions
            obs, reward, done, info = self.env.step(
                sub_action)

            # Check for terminals and input data
            for agent in self.agents:
                # If agent is already dead, do nothing
                if done_buffer[agent.id] == 1:
                    pass
                # If agent died, set him back to his previous position
                elif done[agent.id] == 1:
                    reward_buffer[agent.id] = reward[agent.id]
                    self._set_back_agent(agent, previous_state[agent.id])
                    sub_action[agent.id] = previous_state[agent.id]
                    done_buffer[agent.id] = 1
                # If agent has already reached the end of the episode, do nothing
                elif done_buffer[agent.id] == 2:
                    pass
                # If agent reached the end of episode, remember it
                elif done[agent.id] == 2:
                    done_buffer[agent.id] = 2
                # If agent steps in the environment, let him collect the input data for the next step
                else:
                    if step == int(t_steps / 2):
                        agent.state.image[1] = self._make_image(agent)
                        agent.state.scalars[6:12] = self._make_scalars(agent)
                    if step == t_steps - 1:
                        agent.state.image[2] = self._make_image(agent)
                        agent.state.scalars[12:] = self._make_scalars(agent)

            # Interrupt step if all agents are dead
            if all(d == 1 for d in done_buffer.values()):
                break

        # Build the reward for the action taken
        for agent in self.agents:
            # If agent has not died, take the last reward
            if done_buffer[agent.id] != 1:
                reward_buffer[agent.id] = reward[agent.id]

        return obs, reward_buffer, done_buffer, info

    def reset(self):
        """Resets the state of the environment and returns initial observations"""
        # Run reset
        obs = self.env.reset()

        # Update state
        for idx, agent in enumerate(self.agents):
            _ = self._make_image(agent)
            image = self._make_image(agent)
            agent.state.image = np.stack([image, image, image])
            scalars = self._make_scalars(agent)
            agent.state.scalars = np.hstack([scalars, scalars, scalars])
            self.render_dummy[idx] = agent.state.image

        return obs

    def get_trajectory(self, action):
        """Transforms an action into a trajectory

        This calculation is adapted to the CARLA coordinate system.
        There might be weird behavior if the road in CARLA is not straight.
        """
        trajectory = {}
        for agent in self.env.agents:

            # Get state and action
            position = agent.state.position
            velocity = agent.state.xy_velocity
            acceleration = agent.state.xy_acceleration
            rotation = agent.state.direction_heading
            a = action[agent.id]

            # Calculate trajectory
            # s-axis == x-axis in carla
            # d-axis == -1.0 * y-axis in carla
            traj = self.generator.calculate_trajectory(position=position, velocity=velocity, acceleration=acceleration,
                                                             deltaV=a[0], deltaD=-1.0 * a[1])
            # Correct road twist
            theta = np.radians(rotation)
            traj = self.generator.transform_trajectory(traj, theta)
            trajectory[agent.id] = traj

        return trajectory

    def switch_scenario(self, scenario):
        """Updates the reset information and spawns obstacles"""
        # Update reset information and agent setup
        self.env.switch_scenario(scenario)
        # Reinitialize OpenDrive Rendering
        if self.reset_info[self.map_name][self.scenario]["num_agents"] != self.num_agents:
            self.renderer = {agent.id: OpenDriveRenderer(self.map, self.scenario, agent)
                             for agent in self.agents}

    def switch_world(self, carla_map, scenario=None):
        """Loads a carla world and updates the reset information"""
        # Switch world and respawn agents
        self.env.switch_world(carla_map, scenario)
        # Reinitialize OpenDrive Rendering
        self.renderer = {agent.id: OpenDriveRenderer(self.map, self.scenario, agent)
                         for agent in self.agents}

    def _set_back_agent(self, agent, state):
        """Resets the agent to the previous spot and enable physics"""
        transform = carla.Transform(
            carla.Location(state[0], state[1]),
            carla.Rotation(yaw=state[6])
        )
        agent.vehicle.set_transform(transform)
        agent.state.xy_velocity = (state[2], state[3])
        agent.state.xy_acceleration = (state[4], state[5])
        agent.togglePhysics()

    def _make_scalars(self, agent):
        """Returns a list of relevant scalar state values"""
        # Velocity
        agent_velocity = agent.state.xy_velocity
        agent_velocity = np.around(np.sqrt(
            agent_velocity[0] ** 2 + agent_velocity[1] ** 2),
            2)
        # Acceleration
        agent_acceleration = agent.state.xy_acceleration
        agent_acceleration = np.around(np.sqrt(
            agent_acceleration[0] ** 2 + agent_acceleration[1] ** 2),
            2)
        # Rotation
        abs_agent_rotation = np.abs(agent.state.rotation)
        if abs_agent_rotation <= 90.0:
            agent_rotation = -1.0 * agent.state.rotation
        else:
            agent_rotation = 180.0 - abs_agent_rotation
            agent_rotation = np.sign(agent.state.rotation) * agent_rotation

        scalars = np.array([agent.state.position[0],
                            agent.state.position[1],
                            agent_velocity,
                            agent_acceleration,
                            agent_rotation,
                            agent.state.v_desire])

        return scalars

    def _make_image(self, agent):
        """Returns an OpenDrive image"""
        vehicles = [(actor, actor.get_transform())
                    for actor in self.world.get_actors() if 'vehicle' in actor.type_id]
        return self.renderer[agent.id].get_image(vehicles)


class RewardWrapper(gym.RewardWrapper):

    def __init__(self, env, coop_factor=0.0):
        super(RewardWrapper, self).__init__(env)

        self.coop_factor = coop_factor
        self.reward_weights = {}

    def reward(self, reward):
        """Calculates the cooperative reward for an agent"""
        if self.coop_factor == 0.0:
            return reward
        coop_reward = {}
        for agent in self.agents:
            other_reward = 0.0
            weights = self.reward_weights[agent.id]
            prob_mass = np.sum(np.exp(weights.values()))
            for agent_id, weight in weights.items():
                if agent.id != agent_id:
                    prob = (np.exp(weight) / prob_mass)
                    other_reward += \
                        self.coop_factor * prob * reward[agent_id]
            coop_reward[agent.id] = reward[agent.id] + other_reward

        return coop_reward


class ObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)

    def observation(self, observation):
        """Returns current observations

        obs = [x, y, v, a, heading, v_desire]
        """
        # Collect states
        states = {}
        for agent in self.agents:
            states[agent.id] = agent.state

        # Extract hidden and observable information
        obs = dict()
        for agent in self.agents:

            obs[agent.id] = [None, None, []]

            # Add observable information of other agents
            agent_reward_weights = {}
            for agent_id, state in states.items():
                if agent.id != agent_id:

                    # Center/Transform other agent's vector according to the ego agent
                    other_vector = self.center_scalars(state, agent.state)

                    # Calculate time to collision (ttc)
                    time_to_collision = self.calculate_ttc(agent.state, state)
                    agent_reward_weights[agent_id] = -1.0 * time_to_collision

                    # Stack ttc and obs
                    obs[agent.id][2].append(
                        np.append([time_to_collision], other_vector))

            self.reward_weights[agent.id] = agent_reward_weights

            # Get ego agent's state vector
            agent_vector = np.copy(agent.state.scalars)
            # Center the ego agent's positions
            self._correct_position(agent_vector, agent.state)
            # Transform rotation to rad
            agent_vector[[4, 10, 16]] = np.radians(agent_vector[[4, 10, 16]])

            # Stack obs
            obs[agent.id][0] = agent.state.image
            obs[agent.id][1] = agent_vector
            if self.num_agents > 1:
                sorted_obs = np.sort(obs[agent.id][2], 0)[::-1]
                obs[agent.id][2] = sorted_obs[:, 1:]

            # Normlize obs
            self.normalize_obs(obs[agent.id])

        return obs

    def center_scalars(self, other_state, agent_state):
        """Centers/Transforms observations concerning other agents according to the ego agent"""
        # Get other agent's state vector
        other_vector = np.copy(other_state.scalars)
        # Corrected Position
        self._correct_position(other_vector, agent_state)
        # Corrected Rotation
        if agent_state.driving_direction == other_state.driving_direction:
            other_vector[[4, 10, 16]] = np.radians(other_vector[[4, 10, 16]])
        else:
            other_vector[[4, 10, 16]] = -1.0 * np.sign(other_vector[[4, 10, 16]]) * \
                np.radians(180.0 - np.abs(other_vector[[4, 10, 16]]))
        # Correct Target Speed
        other_vector[[5, 11, 17]] = 12

        return other_vector

    def calculate_ttc(self, agent_state, other_state):
        """Calculates time to collision between two agents"""
        # Compare directions
        same_direction = agent_state.driving_direction == other_state.driving_direction
        # Get distance
        distance = np.sqrt(
            (agent_state.position[0] - other_state.position[0]) ** 2 +
            (agent_state.position[1] - other_state.position[1]) ** 2)
        # Calculate distance to soft terminal x-range condition
        x_range_diff_agent = \
            np.abs(agent_state.position[0] - other_state.x_range)
        x_range_diff_other = \
            np.abs(other_state.position[0] - other_state.x_range)
        # Calculate approx. 1D time to collision
        # Behind ego and opposite direction
        if not same_direction and x_range_diff_other <= x_range_diff_agent:
            time_to_collision = 80.0
        # Behind ego and same direction
        elif same_direction and x_range_diff_other > x_range_diff_agent:
            velocity_term = max(other_state.scalars[2] - agent_state.scalars[2],
                                0.0)
            time_to_collision = distance / (velocity_term + 1.0)
        # In front of ego and opposite direction
        elif not same_direction and x_range_diff_other > x_range_diff_agent:
            velocity_term = agent_state.scalars[2] + \
                other_state.scalars[2]
            time_to_collision = distance / (velocity_term + 1.0)
        # In front of ego and same direction
        elif same_direction and x_range_diff_other <= x_range_diff_agent:
            velocity_term = max(agent_state.scalars[2] - other_state.scalars[2],
                                0.0)
            time_to_collision = distance / (velocity_term + 1.0)

        return time_to_collision

    def normalize_obs(self, obs):
        """Normalize observations"""
        # Ego Agent
        obs[1][[0, 6, 12]] = self._norm_x(obs[1][[0, 6, 12]])
        obs[1][[1, 7, 13]] = self._norm_y(obs[1][[1, 7, 13]])
        obs[1][[2, 8, 14]] = self._norm_vel(obs[1][[2, 8, 14]])
        obs[1][[3, 9, 15]] = self._norm_acc(obs[1][[3, 9, 15]])
        obs[1][[4, 10, 16]] = self._norm_rot(obs[1][[4, 10, 16]])
        obs[1][[5, 11, 17]] = self._norm_vel(obs[1][[5, 11, 17]])
        # Other Agents
        if self.num_agents > 1:
            obs[2][:, [0, 6, 12]] = self._norm_x(obs[2][:, [0, 6, 12]])
            obs[2][:, [1, 7, 13]] = self._norm_y(obs[2][:, [1, 7, 13]])
            obs[2][:, [2, 8, 14]] = self._norm_vel(obs[2][:, [2, 8, 14]])
            obs[2][:, [3, 9, 15]] = self._norm_acc(obs[2][:, [3, 9, 15]])
            obs[2][:, [4, 10, 16]] = self._norm_rot(obs[2][:, [4, 10, 16]])
            obs[2][:, [5, 11, 17]] = self._norm_vel(obs[2][:, [5, 11, 17]])

    def _correct_position(self, other_vector, agent_state):
        """Centers position on the ego agent"""
        other_vector[[0, 6, 12]] -= agent_state.scalars[12]
        other_vector[[1, 7, 13]] -= agent_state.scalars[13]
        if np.abs(agent_state.rotation) <= 90.0:
            other_vector[[1, 7, 13]] *= -1.0
        else:
            other_vector[[0, 6, 12]] *= -1.0

    def _norm_x(self, x, min_=-80.0, max_=80.0):
        """Normalize x-position"""
        return self._min_max_scaler(x, min_, max_)

    def _norm_y(self, y, min_=-6.4, max_=6.4):
        """Normalize y-position"""
        return self._min_max_scaler(y, min_, max_)

    def _norm_vel(self, v, min_=0.0, max_=20.0):
        """Normalize velocity"""
        return self._min_max_scaler(v, min_, max_, 0.0, 1.0)

    def _norm_acc(self, a, min_=-9.81, max_=9.81):
        """Normalize acceleration"""
        return self._min_max_scaler(a, min_, max_)

    def _norm_rot(self, r, min_=-180.0, max_=180.0):
        """Normalize rotation"""
        return self._min_max_scaler(r, np.radians(min_), np.radians(max_))

    def _min_max_scaler(self, x, min_, max_, a=-1.0, b=1.0):
        """Min-Max-Scaler"""
        return a + ((x - min_) * (b - a)) / (max_ - min_)


def make_env(config):
    """Creates the environment"""

    env = BaseEnv(config)

    # Multi-Agent Environment
    env = MultiAgentWrapper(env)

    # Static and Dynamic obstacles
    env = ObstaclesWrapper(env)

    # Trajectory Actions
    env = TrajectoryWrapper(env, config.deltaT, config.dt, config.fraction)

    # Rewards
    env = RewardWrapper(env, config.coop_factor)

    # Observations
    env = ObservationWrapper(env)

    return env
