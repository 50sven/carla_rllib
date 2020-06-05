import os
import argparse
import numpy as np
import carla_rllib
from carla_rllib.environments.carla_envs.base_env import make_env
from carla_rllib.environments.carla_envs.config import BaseConfig
from carla_rllib.utils.clean_up import clear_carla


def run_test(config):
    """Base env test

    Mandatory configuration settings:
        - 'continuous' agent
    """
    env = None
    try:
        # Create Environment
        env = make_env(config)
        obs = env.reset()

        t = 0
        print("-----Carla Environment is running-----")
        while True:

            # Calculate/Predict Actions
            t -= 0.3
            s = 0.4 * np.sin(t)
            a = 0.3

            if env.num_agents == 1:
                action = [s, a]  # Single agent (with continuous control)
            else:
                action = dict(Agent_1=[s, a],  # Two agents (with continuous control)
                              Agent_2=[s, a])

            # Make step in environment
            obs, reward, done, info = env.step(action)

            print(env.agents[0].state)
            print("Reward:", reward)

            # Reset if done
            terminal = any(d != 0 for d in done.values()
                           ) if env.num_agents > 1 else (done != 0)
            if terminal:
                obs = env.reset()
                t = 0
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
