"""CARLA Environment Wrapper

This script provides gym wrapper classes to handle possible variations of the base env.

Classes:
    * MultiAgentWrapper - wrapper class for multi-agent setups
    * SkipFrameWrapper - wrapper class to handle frame skipping
    * SpaceWrapper - wrapper class for gym observation and action spaces
    * ObstaclesWrapper - wrapper class to spawn obstacles
"""
import numpy as np
import carla
import gym
from gym.spaces import Box, Dict


class MultiAgentWrapper(gym.Wrapper):

    def __init__(self, env):
        super(MultiAgentWrapper, self).__init__(env)

    def step(self, action):
        """Runs one timestep of the simulator's dynamics

        Accepts a dictionary which stores an action (list) for each agent and
        returns dictionaries accordingly (observations, rewards, terminals and infos)

        Parameters:
        ----------
        action: dict
            actions provided by policies

        Returns:
        ----------
        obs: dict
            observations of the agents current environment
        reward: dict
            rewards returned after executing the action
        done: dict
            whether the episode of the agents are done
        info: dict
            contains auxiliary diagnostic information
        """
        # Initialize dictionaries
        obs = dict()
        reward = dict()
        done = dict()
        info = dict()

        # Set step
        for agent in self.agents:
            agent.step(action[agent.id])

        # Run step and update state
        self.frame = self.world.tick()
        for agent in self.agents:
            agent.update_state(self.frame,
                               self.start_frame,
                               self.timeout)
            # Retrieve observations, rewards, terminals and infos
            obs[agent.id] = self.get_obs(agent)
            reward[agent.id] = self.get_reward(agent)
            done[agent.id] = self.is_done(agent)
            info[agent.id] = self.get_info(agent)

        if self.render_enabled:
            self.render(self.frame)

        return obs, reward, done, info

    def reset(self):
        """Resets the state of the environment and returns initial observations

        Returns:
        ----------
        obs: dict
            observations of the agents current environment
        """
        # Initialize obs dictionary
        obs = dict()

        # Set reset
        for agent in self.agents:
            reset = self.get_reset(agent)
            agent.reset(reset)

        # Run reset and update state
        self.start_frame = self.world.tick()
        for agent in self.agents:
            agent.update_state(self.start_frame,
                               self.start_frame,
                               self.timeout)
            # Retrieve observations
            obs[agent.id] = self.get_obs(agent)

        if self.render_enabled:
            self.render(self.start_frame)

        return obs


class SkipFrameWrapper(gym.Wrapper):

    def __init__(self, env, frame_skip):
        super(SkipFrameWrapper, self).__init__(env)
        self.frame_skip = frame_skip
        self.multi_agent = True if self.num_agents > 1 else False

    def step(self, action):
        """Repeats actions and accumulates the rewards"""
        # Initialize reward counter
        if self.multi_agent:
            total_reward = dict.fromkeys(action.keys(), 0.0)
        else:
            total_reward = 0.0

        # Run steps
        for _ in range(self.frame_skip + 1):
            obs, reward, done, info = self.env.step(action)

            if self.multi_agent:
                total_reward = {k: v + reward[k]
                                for k, v in total_reward.items()}
            else:
                total_reward += reward

            terminal = any(d != 0 for d in done.values()
                           ) if self.multi_agent else (done != 0)
            if terminal:
                break

        return obs, total_reward, done, info


class SpaceWrapper(gym.Wrapper):

    def __init__(self, env):
        super(SpaceWrapper, self).__init__(env)
        self.action_space = None
        self.observation_space = None

        self._initialize_spaces()

    def _initialize_spaces(self):
        """Creates action and observation spaces"""
        # Create action space
        if self.agent_type == "continuous":
            low = np.array([-1.0, -1.0])
            high = np.array([1.0, 1.0])
        elif self.agent_type == "steering":
            low = -1.0
            high = 1.0
        self.action_space = Box(low, high, dtype=np.float32)

        # Create observation space
        width = self.camera_settings["width"]
        height = self.camera_settings["height"]
        if (self.camera_settings["type_id"] == 'segmentation' and
                self.camera_settings["spec"] == "raw"):
            low = 0
            high = 12
            shape = (height, width, 1)
        else:
            low = 1 if self.camera_settings["spec"] == "log_depth" else 0
            high = 255
            shape = (height, width, 3)
        self.observation_space = Box(low=low, high=high,
                                     shape=shape,
                                     dtype=np.uint8)


class ObstaclesWrapper(gym.Wrapper):

    def __init__(self, env):
        super(ObstaclesWrapper, self).__init__(env)

        self._reset_variables()

        self._start()

    def _start(self):
        """Starts the obstacles spawning"""
        # Update obstacle triggers
        self.static_enabled = self.reset_info[self.map_name][self.scenario]["Obstacles"]["static_enabled"]
        self.dynamic_enabled = self.reset_info[self.map_name][self.scenario]["Obstacles"]["dynamic_enabled"]
        # Spawn obstacles
        if self.static_enabled:
            self._initialize_static()
        if self.dynamic_enabled:
            self._initialize_dynamic()

    def _initialize_static(self):
        """Spawns static obstacles"""
        self.static_info = self.reset_info[self.map_name][self.scenario]["Obstacles"]["static"]

        # Static obstacles
        bp = self.world.get_blueprint_library().find('vehicle.carlamotors.carlacola')
        bp.set_attribute('role_name', "static_obstacle")
        for spot in self.static_info:
            spawn_point = carla.Location(x=self.save_spawn[0],
                                         y=self.save_spawn[1])
            transform = self.map.get_waypoint(spawn_point).transform
            transform.location.z = 0.1
            vehicle = self.world.spawn_actor(bp, transform)
            location = carla.Location(x=spot[0], y=spot[1])
            vehicle.set_location(location)
            self.static_obstacles.append(vehicle)
            self.save_spawn[0] += 7.0

    def _initialize_dynamic(self):
        """Spawns dynamic obstacles"""
        self.dynamic_info = self.reset_info[self.map_name][self.scenario]["Obstacles"]["dynamic"]

        # Dynamic obstacles
        bp = self.world.get_blueprint_library().find('vehicle.audi.tt')
        bp.set_attribute('role_name', "dynamic_obstacle")
        for spot in self.dynamic_info:
            spawn_point = carla.Location(x=self.save_spawn[0],
                                         y=self.save_spawn[1])
            transform = self.map.get_waypoint(spawn_point).transform
            transform.location.z = 0.1
            vehicle = self.world.spawn_actor(bp, transform)
            location = carla.Location(x=spot[0], y=spot[1])
            vehicle.set_location(location)
            self.dynamic_obstacles.append((vehicle, spot[2]))
            self.save_spawn[0] += 7.0

    def step(self, action):
        """Sets the actions for dynamic obstacles and runs the base step"""
        for vehicle, velocity in self.dynamic_obstacles:
            waypoint = self.map.get_waypoint(vehicle.get_location())
            next_waypoint = waypoint.next(self.delta_sec * velocity)[0]
            vehicle.set_transform(next_waypoint.transform)

        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

    def reset(self):
        """Resets the obstacles and runs the base reset"""

        if self.static_enabled:
            self._reset_obstacles(self.static_obstacles, self.static_info)
        if self.dynamic_enabled:
            dynamic_obstacles = [o[0] for o in self.dynamic_obstacles]
            self._reset_obstacles(dynamic_obstacles, self.dynamic_info)

        obs = self.env.reset()

        return obs

    def switch_scenario(self, scenario):
        """Updates the reset information and spawns obstacles"""
        # Destroy obstacles and reset variables
        self._destroy_obstacles()
        # Load new scenario
        self.env.switch_scenario(scenario)
        # Spawn obstacles
        self._start()

    def switch_world(self, carla_map, scenario=None):
        """Loads a carla world and updates the reset information"""
        # Destroy obstacles and reset variables
        self._destroy_obstacles()
        # Load new world and update reset information
        self.env.switch_world(carla_map, scenario)
        # Spawn obstacles
        self._start()

    def _reset_obstacles(self, obstacles, info):
        """Resets all obstacles"""
        for vehicle, spot in zip(obstacles, info):
            spawn_point = carla.Location(x=spot[0], y=spot[1])
            transform = self.map.get_waypoint(spawn_point).transform
            transform.location = spawn_point
            vehicle.set_transform(transform)
            vehicle.set_velocity(carla.Vector3D())
            vehicle.set_angular_velocity(carla.Vector3D())

    def _destroy_obstacles(self):
        """Destroys all obstacles"""
        obstacles = self.static_obstacles + [o[0]
                                             for o in self.dynamic_obstacles]
        for vehicle in obstacles:
            vehicle.destroy()
        self._reset_variables()

    def _reset_variables(self):
        """ """
        self.static_info = None
        self.static_enabled = False
        self.static_obstacles = []
        self.dynamic_info = None
        self.dynamic_enabled = False
        self.dynamic_obstacles = []
        self.save_spawn = [-100.0, 51.75]

    def close(self):
        """ """
        for vehicle in self.static_obstacles:
            vehicle.destroy()
        for vehicle, _ in self.dynamic_obstacles:
            vehicle.destroy()
        self.env.close()


class MultiAgentRewardWrapper(gym.RewardWrapper):

    def __init__(self, env, coop_factor=0.0):
        super(MultiAgentRewardWrapper, self).__init__(env)

        self.coop_factor = coop_factor

    def reward(self, reward):
        """Calculates a global reward for an agent"""
        reward_sum = np.sum(reward.values())
        for agent in self.agents:
            reward[agent.id] += (reward_sum -
                                 reward[agent.id]) * self.coop_factor
        return reward
