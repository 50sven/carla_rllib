"""State Information

This script provides a basic state class to store various information about an agent.
"""
import numpy as np


class BaseState(object):

    def __init__(self):

        self.reset()

    def reset(self):
        """Initializes/resets the state"""
        self.image = None
        self.elapsed_ticks = 0
        self.position = (0.0, 0.0)
        self.rotation = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.distance_to_center_line = 0.0
        self.direction_heading = 0.0
        self.delta_heading = 0.0
        self.speed_limit = 0.0
        self.driving_direction = 0
        self.lane_id = 0
        self.lane_width = 0.0
        self.lane_type = "NONE"
        self.lane_change = "NONE"
        self.lane_invasion = 0
        self.opposite_lane = False
        self.junction = False
        self.collision = False
        self.collided_obstacle = None
        self.terminal = 0
        # 0 = not terminal,
        # 1 = terminal due to hard condition (e.g. collision),
        # 2 = terminal due to soft condition (e.g. early stopping)

        # These variables are only used in the trajectory env
        self.scalars = None
        self.v_desire = 0.0
        self.xy_velocity = (0.0, 0.0)
        self.xy_acceleration = (0.0, 0.0)
        self.desired_lane = None
        self.x_range = 0.0

    def __repr__(self):
        return ("Image: %s\n" +
                "Elapsed ticks: %s\n" +
                "Position: %s\n" +
                "Rotation: %.2f\n" +
                "Velocity: %.2f\n" +
                "Acceleration: %.2f\n" +
                "Distance to center line: %.2f\n" +
                "Direction heading: %.2f\n" +
                "Delta heading: %.2f\n" +
                "Speed limit: %.2f\n" +
                "Driving direction: %s\n" +
                "Lane id: %s\n" +
                "Lane width %s\n" +
                "Lane type: %s\n" +
                "Lane change: %s\n" +
                "Opposite lane: %s\n" +
                "Lane Invasion: %s\n" +
                "Junction: %s\n" +
                "Collision: %s\n" +
                "Collided obstacle: %s\n" +
                "Terminal: %s") % (type(self.image),
                                   self.elapsed_ticks,
                                   self.position,
                                   self.rotation,
                                   self.velocity,
                                   self.acceleration,
                                   self.distance_to_center_line,
                                   self.direction_heading,
                                   self.delta_heading,
                                   self.speed_limit,
                                   self.driving_direction,
                                   self.lane_id,
                                   self.lane_width,
                                   self.lane_type,
                                   self.lane_change,
                                   self.opposite_lane,
                                   self.lane_invasion,
                                   self.junction,
                                   self.collision,
                                   self.collided_obstacle,
                                   self.terminal)
