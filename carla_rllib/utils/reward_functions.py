"""Examples of Reward Functions

This script provides examples of reward functions for reinforcement learning.
"""
import numpy as np


def example_reward(distance_to_center_line, delta_heading, current_speed, target_speed, terminal, eval_):
    """Reward example for agents with continuous or discrete control (ContinuousWrapper/DiscreteWrapper)"""
    # Terminal reward
    if terminal == 1:
        return [-50.0, 0.0, 0.0, 0.0] if eval_ else -50.0

    # Distance to center line reward
    distance_reward = -distance_to_center_line / 1.75

    # Speed reward
    speed_reward = -0.5 * np.abs(target_speed - current_speed)

    # Heading reward
    heading_reward = -min(0.02 * delta_heading, 0.5)

    # Combination
    reward = 2.0 + distance_reward + speed_reward + heading_reward

    return [reward, distance_reward, speed_reward, heading_reward] if eval_ else reward
