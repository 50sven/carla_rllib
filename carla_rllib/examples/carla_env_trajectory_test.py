import os
import argparse
import numpy as np
import carla_rllib
from carla_rllib.environments.carla_envs.trajectory_env import make_env
from carla_rllib.environments.carla_envs.config import BaseConfig
from carla_rllib.utils.clean_up import clear_carla
from carla_rllib.utils.trajectory_planning import PolynomialGenerator


def run_test(config):
    """Trajectory env test

    Mandatory configuration settings:
        - 'discrete' agent
    """
    env = None
    try:
        # Add parameters for polynomial trajectory generation to config
        config.deltaT, config.dt, config.fraction = 1.0, 0.05, 0.25

        # Create Environment
        env = make_env(config)
        obs = env.reset()
        # Initialize maneuver planning
        generator = PolynomialGenerator(deltaT=1.0, dt=0.05, fraction=0.25)
        print("-----Carla Environment is running-----")
        while True:

            # Calculate/Predict Actions
            random_action = np.random.normal([0.0, 0.0], [1.0, 0.5])
            dv = np.clip(random_action[0], -3.5, 3.5)
            dl = np.clip(random_action[1], -1.75, 1.75)
            deltaV, deltaD = dv, dl

            action_dict = dict(Agent_1=[deltaV, deltaD],
                               Agent_2=[deltaV, deltaD * -1.0],
                               Agent_3=[deltaV, deltaD],
                               Agent_4=[deltaV, deltaD * -1.0])

            # Make step in environment
            obs, reward, done, info = env.step(action_dict)
            print(reward)
            print("-" * 20)

            # Reset if done
            terminal = any(d != 0 for d in done.values())

            if terminal:
                obs = env.reset()

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
