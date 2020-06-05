"""Configuration

This script provides a basic configuration class to parse a json configuration file.
"""
import os
import json
import numpy as np


class BaseConfig(object):

    def __init__(self, json_file):

        with open(json_file, "r") as f:
            config_json = json.load(f)

        self.parse(config_json)

    def parse(self, config_json):
        """Parse json configuration file"""
        # CARLA server
        self.host = str(config_json["carla_server"]["host"])
        self.port = config_json["carla_server"]["port"]
        self.map = str(config_json["carla_server"]["map"])
        self.sync_mode = config_json["carla_server"]["sync_mode"]
        self.delta_sec = config_json["carla_server"]["delta_sec"]
        self.no_rendering_mode = config_json["carla_server"]["no_rendering_mode"]
        # Environment
        with open(os.path.expanduser(config_json["environment"]["reset_info"]), "r") as f:
            self.reset_info = json.load(f)
        self.scenario = config_json["environment"]["scenario"]
        self.scenario = self.scenario if self.scenario else np.random.choice(
            self.reset_info[self.map].keys())
        self.num_agents = self.reset_info[self.map][self.scenario]["num_agents"]
        self.coop_factor = config_json["environment"]["coop_factor"]
        self.frame_skip = config_json["environment"]["frame_skip"]
        self.render = config_json["environment"]["render"]
        self.stable_baselines = config_json["environment"]["stable_baselines"]
        # Agent
        self.agent_type = str(config_json["agent"]["agent_type"])
        self.camera_settings = config_json["agent"]["camera_settings"]

    def __repr__(self):
        return ("CARLA Server:\n" +
                "\tHost: %s\n" +
                "\tPort: %s\n" +
                "\tMap: %s\n" +
                "\tSync Mode: %s\n" +
                "\tDelta Seconds (Sync mode only): %s\n" +
                "\tNo rendering mode: %s\n" +
                "Environment:\n" +
                "\tScenario: %s\n" +
                "\tNumber of Agents: %s\n" +
                "\tCooperation Factor: %s\n" +
                "\tFrame skipping: %s\n" +
                "\tRendering: %s\n" +
                "\tBaselines support (Single agent only): %s\n" +
                "Agent:\n" +
                "\tAgent type: %s\n" +
                "\tCamera settings: %s") % (self.host,
                                            self.port,
                                            self.map,
                                            self.sync_mode,
                                            self.delta_sec,
                                            self.no_rendering_mode,
                                            self.scenario,
                                            self.num_agents,
                                            self.coop_factor,
                                            self.frame_skip,
                                            self.render,
                                            self.stable_baselines,
                                            self.agent_type,
                                            self.camera_settings)
