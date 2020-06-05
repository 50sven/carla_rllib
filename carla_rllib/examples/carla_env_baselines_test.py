import os
import argparse
import numpy as np
import carla_rllib
from carla_rllib.environments.carla_envs.base_env import make_env
from carla_rllib.environments.carla_envs.config import BaseConfig
from carla_rllib.utils.clean_up import clear_carla

from stable_baselines import DDPG
from stable_baselines.ddpg.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise


def run_test(config):
    """Stable baselines test

    Mandatory configuration settings:
        - 'continuous' agent
        - camera_settings enabled
        - stable_baselines enabled
    """
    env = None
    try:
        # Create Environment
        env = make_env(config)
        env = DummyVecEnv([lambda: env])

        # Initialize DDPG and start learning
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        model = DDPG(CnnPolicy, env, verbose=1, param_noise=param_noise,
                     action_noise=action_noise, random_exploration=0.8)
        model.learn(total_timesteps=10000)

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
