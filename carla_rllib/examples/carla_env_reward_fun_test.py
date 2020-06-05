import os
import argparse
import numpy as np
import carla_rllib
from carla_rllib.environments.carla_envs.base_env import make_env
from carla_rllib.environments.carla_envs.config import BaseConfig
from carla_rllib.utils.clean_up import clear_carla

import pygame
from pygame.locals import K_w
from pygame.locals import K_a
from pygame.locals import K_s
from pygame.locals import K_d


def run_test(config):
    """Reward test

    Mandatory configuration settings:
        - 'continuous' agent
    """
    env = None
    try:
        # Create Environment
        env = make_env(config)
        obs = env.reset()

        a = 0
        y = 0
        steer_cache = 0
        clock = pygame.time.Clock()
        print("-----Carla Environment is running-----")
        while True:
            milliseconds = clock.get_time()
            keys = pygame.key.get_pressed()
            steer_increment = 9e-4 * milliseconds
            if keys[K_w]:
                a = 1.0
            elif keys[K_s]:
                a = -1.0
            else:
                a = 0.0

            if keys[K_a]:
                steer_cache -= steer_increment
            elif keys[K_d]:
                steer_cache += steer_increment
            else:
                steer_cache = 0.0

            s = round(min(0.7, max(-0.7, steer_cache)))

            action = [s, a]
            obs, reward, done, info = env.step(action)
            clock.tick(5)

            y += 1
            if y % 5 == 0:
                print("reward", reward)
                y = 0

            if done:
                env.reset()

    finally:
        if env:
            env.close()
        else:
            clear_carla(config.host, config.port)
        print("-----Carla Environment is closed-----")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='CARLA RLLIB ENV')
    package_path, _ = os.path.split(os.path.abspath(carla_rllib.__file__))
    argparser.add_argument(
        '-c', '--config',
        metavar='CONFIG',
        default=os.path.join(package_path +
                             "/config.json"),
        type=str,
        help='Path to configuration file (default: root of the package -> carla_rllib)')
    args = argparser.parse_args()
    config = BaseConfig(args.config)
    print("-----Configuration-----")
    print(config)

    run_test(config)
