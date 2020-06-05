"""CARLA Base Environment

This script provides a basic single- and multi-agent environment for
Reinforcement Learning with the Carla Simulator.

Classes:
    * BaseEnv - base environment class
Functions:
    * make_env - creates the environment
"""
import numpy as np
import carla
import importlib
import gym
from gym.spaces import Box, Dict
from carla_rllib.utils.spectators import ActorSpectator
from carla_rllib.utils.rendering import OPENDRIVE_CMAP, CARLA_CMAP
from carla_rllib.utils.reward_functions import example_reward
from carla_rllib.environments.carla_envs.env_wrapper import MultiAgentWrapper, SkipFrameWrapper, SpaceWrapper, ObstaclesWrapper, MultiAgentRewardWrapper


class BaseEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels']}

    def __init__(self, config):

        # Read config
        self._read_config(config)

        # Initialize client and get/load map
        self._initialize_client()

        # Enable/Disable synchronous mode
        self._initialize_world_settings()

        # Start the carla environment
        self._start()

    def _start(self):
        """Starts the carla environment"""
        # Create agent(s)
        self._initialize_agents()

        # Spawn agents
        self._spawn_agents()

        # Start rendering
        if self.render_enabled:
            self._start_rendering()

    def _read_config(self, config):
        """Extracts the configuration variables"""
        # CARLA Server
        self.host = config.host
        self.port = config.port
        self.map_name = config.map
        self.sync_mode = config.sync_mode
        self.delta_sec = config.delta_sec
        self.no_rendering_mode = config.no_rendering_mode

        # Environment
        self.reset_info = config.reset_info
        self.scenario = config.scenario if config.scenario else np.random.choice(
            self.reset_info[self.map_name].keys())
        self.num_agents = self.reset_info[self.map_name][self.scenario]["num_agents"]
        self.render_enabled = config.render

        # Agent
        self.agent_type = config.agent_type
        self.camera_settings = config.camera_settings

        # Declare remaining variables
        try:
            self.eval = config.eval
        except:
            self.eval = False
        self.frame = None
        self.timeout = 5.0
        self.render_dummy = [None for _ in range(self.num_agents)]

        assert (self.render_enabled and self.no_rendering_mode) == False, "No rendering mode and environment rendering are both enabled. Disable one of them!"

    def _initialize_client(self):
        """Connects to the simulator and retrieves world and map"""
        try:
            self.client = carla.Client(self.host, self.port)
            if (self.map_name and self.client.get_world().get_map().name != self.map_name):
                self._load_map(self.map_name)
            else:
                self.world = self.client.get_world()
                self.map = self.world.get_map()
                self.client.set_timeout(self.timeout)
            print("Connected to Carla Server")
        except:
            raise ConnectionError("Cannot connect to Carla Server!")

    def _load_map(self, map_name):
        """Loads a carla map"""
        self.client.set_timeout(100.0)
        print('Load map: %r.' % map_name)
        self.world = self.client.load_world(map_name)
        self.map = self.world.get_map()
        self.client.set_timeout(self.timeout)

    def _initialize_world_settings(self):
        """Applies the simulator settings"""
        self.settings = self.world.get_settings()
        _ = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=self.no_rendering_mode,
            synchronous_mode=self.sync_mode,
            fixed_delta_seconds=self.delta_sec))
        if self.sync_mode:
            print("Synchronous Mode enabled")
        else:
            print("Synchronous Mode disabled")

        if self.no_rendering_mode:
            print("No Rendering Mode enabled (==camera sensors are disabled)")
        else:
            print("No Rendering Mode disabled")

    def _initialize_agents(self):
        """Creates the agents"""
        self.agents = []
        spawn_points = self.map.get_spawn_points()[
            :self.num_agents]
        for idx, n in enumerate(range(self.num_agents), 1):
            module = importlib.import_module(
                "carla_rllib.carla_wrapper.wrappers")
            agent_type = self.agent_type.capitalize() + "Wrapper"
            agent = getattr(module, agent_type)(idx,
                                                self.world,
                                                spawn_points[n],
                                                self.camera_settings)
            self.agents.append(agent)

    def _spawn_agents(self):
        """Spawns the agents"""
        # Hacky workaround to solve waiting time when spawned:
        # Unreal Engine simulates starting the car and shifting gears,
        # so you are not able to apply controls for ~2s when an agent is spawned
        for agent in self.agents:
            agent.vehicle.apply_control(
                carla.VehicleControl(manual_gear_shift=True, gear=1))
        self.start_frame = self.world.tick()
        for agent in self.agents:
            agent.vehicle.apply_control(
                carla.VehicleControl(manual_gear_shift=False))
        print("Agent(s) spawned")

        if self.num_agents == 1:
            self.agent = self.agents[0]

    def _start_rendering(self):
        """Turns on rendering mode"""
        class config:
            pass
        config.width = 1280
        config.height = 720
        config.fov = 100.0
        config.gamma = 2.2
        config.location = [-8.0, 0.0, 6.0]
        config.rotation = [0.0, -30.0, 0.0]
        self.spectator = ActorSpectator(self.world, config,
                                        integrated=True,
                                        recording=False,
                                        record_path="")

    def step(self, action):
        """Runs one timestep of the simulator's dynamics

        Accepts an action (list) and returns a tuple (observations, reward, terminal and info)

        Parameters:
        ----------
        action: list
            action provided by a policy

        Returns:
        ----------
        obs: object
            observation of the agent's current environment
        reward: float
            reward returned after executing the action
        done: bool
            whether the episode of the agent is done
        info: dict
            contains auxiliary diagnostic information
        """
        # Set step
        self.agent.step(action)

        # Run step and update state
        self.frame = self.world.tick()
        self.agent.update_state(self.frame,
                                self.start_frame,
                                self.timeout)

        # Retrieve observations, rewards, terminals and infos
        obs = self.get_obs(self.agent)
        reward = self.get_reward(self.agent)
        done = self.is_done(self.agent)
        info = self.get_info(self.agent)

        if self.render_enabled:
            self.render(self.frame)

        return obs, reward, done, info

    def reset(self):
        """Resets the state of the environment and returns initial observations

        Returns:
        ----------
        obs: object
            observation of the agent's current environment
        """
        # Set reset
        reset = self.get_reset(self.agent)
        self.agent.reset(reset)

        # Run reset and update state
        self.start_frame = self.world.tick()
        self.agent.update_state(self.start_frame,
                                self.start_frame,
                                self.timeout)

        # Retrieve observations
        obs = self.get_obs(self.agent)

        # Spectator rendering
        if self.render_enabled:
            self.render(self.start_frame)

        return obs

    def render(self, frame, mode='human'):
        """Renders the spectator"""
        e = self.spectator.parse_events()
        if e == 1:
            return exit()
        if e == 2:
            self.render_enabled = False
            self.spectator.destroy()
            return

        # Prepare neural net input(s) to be rendered in the spectator window
        # Example: convert segmentation map (single channel) to RGB image using CARLA's colormap
        # This is a prototypical implementation. Adjust the render()-method in the spectator class if necessary
        image = self.render_dummy[self.spectator.index] if self.render_dummy[self.spectator.index] is not None else self.agents[
            self.spectator.index].state.image
        cam_enabled = self.camera_settings["enabled"]
        cam_type = self.camera_settings["type_id"]

        if (cam_enabled and (cam_type == "rgb" or cam_type == "depth")):
            self.spectator.render(frame, image)

        elif (cam_enabled and cam_type == "segmentation"):
            cmap = CARLA_CMAP
            rgb = self._convert_to_rgb(image[:, :, 0], cmap)
            self.spectator.render(frame, rgb)

        elif (not cam_enabled and isinstance(image, np.ndarray)):
            cat_img = 99.0 * np.ones((160, 3))
            for idx in range(image.shape[0]):
                sep = 99.0 * np.ones((160, 3))
                cat_img = np.concatenate([cat_img, image[idx], sep], axis=1)
            cmap = OPENDRIVE_CMAP
            rgb = self._convert_to_rgb(cat_img, cmap)
            self.spectator.render(frame, rgb)

        else:
            self.spectator.render(frame, None)

    def _convert_to_rgb(self, image, cmap):
        """Converts segmentation images to rgb"""
        rgb = np.zeros((image.shape[0], image.shape[1], 3))
        for key, value in cmap.items():
            rgb[np.where(image == key)] = value
        return rgb

    def get_obs(self, agent):
        """Returns current observations

        ---Note---
        Use the agent's state information
        """
        obs = agent.state.image
        return obs

    def get_reward(self, agent):
        """Returns the current reward"""
        if self.agent_type == "continuous":
            reward = example_reward(agent.state.distance_to_center_line,
                                    agent.state.delta_heading,
                                    agent.state.velocity,
                                    agent.state.v_desire,
                                    agent.state.terminal,
                                    self.eval)
        elif self.agent_type == "discrete":
            velocity = agent.state.xy_velocity
            current_speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            reward = example_reward(agent.state.distance_to_center_line,
                                    agent.state.delta_heading,
                                    current_speed,
                                    agent.state.v_desire,
                                    agent.state.terminal,
                                    self.eval)
        return reward

    def is_done(self, agent):
        """Returns the current terminal condition"""
        done = agent.state.terminal
        return done

    def get_info(self, agent):
        """Returns additional information"""
        info_dict = dict(Info="Store whatever you want")
        return info_dict

    def get_reset(self, agent):
        """Returns the reset information for an agent

        ---Note---
        Specify the reset information in the reset_info.json configuration file
        Adjust wrapper reset function if necessary
        """
        agent_reset = self.reset_info[self.map_name][self.scenario][agent.id]
        return agent_reset

    def switch_scenario(self, scenario):
        """Updates the reset information (and respawns agents)"""
        self.scenario = scenario
        if self.reset_info[self.map_name][self.scenario]["num_agents"] != self.num_agents:
            # Destroy agents
            self._destroy_agents()
            self.num_agents = self.reset_info[self.map_name][self.scenario]["num_agents"]
            self.render_dummy = [None for _ in range(self.num_agents)]
            # Start the carla environment
            self._start()
        print("Reset information updated: {}".format(scenario))

    def switch_world(self, carla_map, scenario=None):
        """Loads a carla world and updates the reset information"""
        if carla_map != self.map_name:
            # Destroy agents
            self._destroy_agents()
            # Load new carla world
            self._load_map(carla_map)
            self.map_name = carla_map
            # Update reset information
            self.scenario = scenario if scenario else np.random.choice(
                self.reset_info[self.map_name].keys())
            self.num_agents = self.reset_info[self.map_name][self.scenario]["num_agents"]
            self.render_dummy = [None for _ in range(self.num_agents)]
            # Enable/Disable synchronous mode
            self._initialize_world_settings()
            # Start the carla environment
            self._start()
            print("The world has been switched and " +
                  "the reset information has been updated!")
        elif scenario is not None and self.scenario != scenario:
            self.switch_scenario(scenario)

    def close(self):
        """Destroys the agent(s) and resets world settings"""
        self._destroy_agents()
        if self.render_enabled:
            self.spectator.destroy()
        self.world.apply_settings(self.settings)

    def _destroy_agents(self):
        """ """
        for agent in self.agents:
            agent.destroy()


def make_env(config):
    """Creates the environment"""
    print("-----Starting Environment-----")

    env = BaseEnv(config)

    # Multi-Agent Environment
    if config.num_agents > 1:
        env = MultiAgentWrapper(env)

        if config.coop_factor != 0.0:
            env = MultiAgentRewardWrapper(config.coop_factor)

    # Static and Dynamic obstacles
    if (env.reset_info[env.map_name][env.scenario]["Obstacles"]["static_enabled"] or
            env.reset_info[env.map_name][env.scenario]["Obstacles"]["dynamic_enabled"]):
        env = ObstaclesWrapper(env)

    # Frame skipping
    if (config.frame_skip > 0 and config.agent_type != "discrete"):
        env = SkipFrameWrapper(env, config.frame_skip)
        print("Frame skipping enabled")
    else:
        print("Frame skipping disabled")

    # Baselines support
    if (config.stable_baselines and
        config.num_agents == 1 and
            config.camera_settings["enabled"] == True):
        env = SpaceWrapper(env)
        print("Baselines support enabled")
    else:
        print("Baselines support disabled")

    print("-----Environment has started-----")

    return env
