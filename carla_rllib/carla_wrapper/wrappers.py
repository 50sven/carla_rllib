"""CARLA Wrapper

This script provides CARLA wrappers to control actors in the CARLA simulator.

Classes:
    * BaseWrapper - wrapper base class
    * ContinuousWrapper - vehicle actor with steer and throttle control
    * SteeringWrapper - vehicle actor with steer control and constant velocity
    * DiscreteWrapper - vehicle actor with discrete action control (teleportation, for e.g. trajectories)
"""
import pygame
import numpy as np
import carla
from carla import ColorConverter as cc
from carla_rllib.carla_wrapper.sensors import CameraSensor, CollisionSensor, LaneInvasionSensor
from carla_rllib.carla_wrapper.states import BaseState


VEHICLE_MODELS = ['vehicle.audi.a2',
                  'vehicle.audi.tt',
                  'vehicle.carlamotors.carlacola',
                  'vehicle.citroen.c3',
                  'vehicle.dodge_charger.police',
                  'vehicle.jeep.wrangler_rubicon',
                  'vehicle.yamaha.yzf',
                  'vehicle.nissan.patrol',
                  'vehicle.gazelle.omafiets',
                  'vehicle.ford.mustang',
                  'vehicle.bmw.isetta',
                  'vehicle.audi.etron',
                  'vehicle.bmw.grandtourer',
                  'vehicle.mercedes-benz.coupe',
                  'vehicle.toyota.prius',
                  'vehicle.diamondback.century',
                  'vehicle.tesla.model3',
                  'vehicle.seat.leon',
                  'vehicle.lincoln.mkz2017',
                  'vehicle.kawasaki.ninja',
                  'vehicle.volkswagen.t2',
                  'vehicle.nissan.micra',
                  'vehicle.chevrolet.impala',
                  'vehicle.mini.cooperst']


class BaseWrapper(object):

    def __init__(self, idx, world, spawn_point, camera_settings):

        self.id = "Agent_" + str(idx)
        self.world = world
        self.map = self.world.get_map()
        self.camera_settings = camera_settings
        self.vehicle = None
        self.sensors = []
        self.state = BaseState()
        self.simulate_physics = True

        model = VEHICLE_MODELS[1]
        self._initialize_vehicle(spawn_point, model)
        self._initialize_sensors()

        print(self.id + " was spawned in " + str(self.map.name))

    def _initialize_vehicle(self, spawn_point, actor_model=None):
        """Spawns vehicle"""
        # Get (random) blueprint
        if actor_model:
            blueprint = self.world.get_blueprint_library().find(actor_model)
        else:
            blueprint = np.random.choice(
                self.world.get_blueprint_library().filter("vehicle.*"))
        blueprint.set_attribute('role_name', self.id)
        if blueprint.has_attribute('color'):
            color = np.random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

    def _initialize_sensors(self):
        """Initializes sensors"""
        self.sensors.append(CollisionSensor(self.vehicle))
        self.sensors.append(LaneInvasionSensor(self.vehicle))
        if self.camera_settings["enabled"]:
            self.sensors.append(CameraSensor(parent_actor=self.vehicle,
                                             type_id=self.camera_settings["type_id"],
                                             spec=self.camera_settings["spec"],
                                             width=self.camera_settings["width"],
                                             height=self.camera_settings["height"],
                                             fov=self.camera_settings["fov"],
                                             location=self.camera_settings["location"],
                                             rotation=self.camera_settings["rotation"]))

    def step(self, action):
        """Sets the action of an actor

        Parameters:
        ----------
        action: array-like
            action to control the vehicle actor
        """
        raise NotImplementedError

    def reset(self, reset):
        """Resets state, physic controls and sensors

        Parameters:
        ----------
        reset: array-like object
            Position coordinates x and y (can be extended)
        """
        # Position
        if isinstance(reset["X"], list):
            idx = np.random.choice(len(reset["X"]))
            pos_x, pos_y = reset["X"][idx], reset["Y"][idx]
        else:
            pos_x, pos_y = reset["X"], reset["Y"]
        location = carla.Location(x=pos_x, y=pos_y)
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        self.vehicle.set_transform(waypoint.transform)

        # Physic controls
        self.vehicle.set_velocity(carla.Vector3D())
        self.vehicle.set_angular_velocity(carla.Vector3D())
        control = self.vehicle.get_control()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        control.manual_gear_shift = False
        control.reverse = False
        self.vehicle.apply_control(control)

        # Sensors and state
        self.sensors[0].reset()
        self.sensors[1].reset()
        self.state.reset()
        self.state.driving_direction = np.sign(waypoint.lane_id)

        # Enable simulation physics if disabled
        if not self.simulate_physics:
            self.togglePhysics()

    def update_state(self, frame, start_frame, timeout):
        """Updates the agent's current state"""

        # Retrieve sensor data
        self._get_sensor_data(frame, timeout)

        # Calculate non-sensor data
        self._get_non_sensor_data(frame, start_frame)

        # Check terminal conditions
        self.state.terminal = self._is_terminal()

        # Disable simulation physics if terminal
        if self.state.terminal:
            self.togglePhysics()

    def _get_sensor_data(self, frame, timeout):
        """Retrieves sensor data"""
        data = [s.retrieve_data(frame, timeout)
                for s in self.sensors]
        self.state.collision = data[0][0]
        self.state.collided_obstacle = data[0][1]
        self.state.lane_invasion = data[1]
        if self.camera_settings["enabled"]:
            self.state.image = data[2]

    def _get_non_sensor_data(self, frame, start_frame):
        """Calculates non-sensor data"""

        # Position
        transformation = self.vehicle.get_transform()
        location = transformation.location
        self.state.position = (np.around(location.x, 2),
                               np.around(location.y, 2))

        # Rotation
        rotation = transformation.rotation
        self.state.rotation = np.around(rotation.yaw, 2)

        # Velocity
        velocity = self.vehicle.get_velocity()
        self.state.velocity = np.around(np.sqrt(velocity.x**2 +
                                                velocity.y**2), 2)

        # Acceleration
        acceleration = self.vehicle.get_acceleration()
        self.state.acceleration = np.around(np.sqrt(acceleration.x**2 +
                                                    acceleration.y**2), 2)

        # Heading wrt lane direction
        nearest_wp = self.map.get_waypoint(location,
                                           project_to_road=True)
        wp_heading = nearest_wp.transform.rotation.yaw
        delta_heading = np.abs(rotation.yaw - wp_heading)
        if delta_heading <= 180:
            delta_heading = delta_heading
        elif delta_heading > 180 and delta_heading <= 360:
            delta_heading = 360 - delta_heading
        else:
            delta_heading = delta_heading - 360

        # Lane id
        self.state.lane_id = nearest_wp.lane_id

        # Opposite lane check,
        # Delta heading,
        # Direction heading and
        # Distance to center line of the closest right lane
        if (self.state.driving_direction != np.sign(self.state.lane_id) or
                delta_heading >= 90):
            self.state.opposite_lane = True
            if self.state.driving_direction == np.sign(self.state.lane_id):
                self.state.delta_heading = delta_heading
            else:
                self.state.delta_heading = 180 - delta_heading
            self.state.direction_heading = wp_heading - \
                180 * np.sign(wp_heading)
            try:
                wp_right = nearest_wp.get_left_lane()
                self.state.distance_to_center_line = np.sqrt(
                    (location.x - wp_right.transform.location.x) ** 2 +
                    (location.y - wp_right.transform.location.y) ** 2
                )
            except:
                distance = np.sqrt(
                    (location.x - nearest_wp.transform.location.x) ** 2 +
                    (location.y - nearest_wp.transform.location.y) ** 2
                )
                self.state.distance_to_center_line = nearest_wp.lane_width - distance

        else:
            self.state.opposite_lane = False
            self.state.delta_heading = delta_heading
            self.state.direction_heading = wp_heading
            self.state.distance_to_center_line = np.sqrt(
                (location.x - nearest_wp.transform.location.x) ** 2 +
                (location.y - nearest_wp.transform.location.y) ** 2
            )

        # Lane type
        self.state.lane_type = nearest_wp.lane_type.name

        # Lane change
        self.state.lane_change = nearest_wp.lane_change.name

        # Lane width
        self.state.lane_width = nearest_wp.lane_width

        # Junction check
        self.junction = nearest_wp.is_junction

        # Elapsed ticks
        self.state.elapsed_ticks = frame - start_frame

        # Speed limit
        speed_limit = self.vehicle.get_speed_limit()
        if speed_limit:
            self.state.speed_limit = speed_limit / 3.6
        else:
            self.state.speed_limit = 50.0 / 3.6

    def _is_terminal(self):
        """Reviews terminal conditions"""
        if (self.state.collision):
            return 1
        elif (self.state.elapsed_ticks >= 1000):
            return 2
        else:
            return 0

    def togglePhysics(self):
        """Enables/Disables physic simulation"""
        self.simulate_physics = not self.simulate_physics
        self.vehicle.set_simulate_physics(self.simulate_physics)

    def destroy(self):
        """Destroys vehicle and sensors"""
        actors = [s.sensor for s in self.sensors] + [self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()


class ContinuousWrapper(BaseWrapper):

    def __init__(self, idx, world, spawn_point, camera_settings):
        super(ContinuousWrapper, self).__init__(
            idx, world, spawn_point, camera_settings)

    def step(self, action):
        """Sets steer and throttle/brake control

        action = [steer, acceleration]
        """
        control = self.vehicle.get_control()
        control.manual_gear_shift = False
        control.reverse = False
        control.hand_brake = False
        control.steer = float(action[0])

        if action[1] >= 0:
            control.brake = 0
            control.throttle = float(action[1])
        else:
            control.throttle = 0
            control.brake = -float(action[1])
        self.vehicle.apply_control(control)


class SteeringWrapper(BaseWrapper):

    def __init__(self, idx, world, spawn_point, camera_settings):
        super(SteeringWrapper, self).__init__(
            idx, world, spawn_point, camera_settings)

    def step(self, action):
        """Sets steer control

        action = [steer]
        """
        # Steering
        control = self.vehicle.get_control()
        control.steer = float(action[0])

        # Constant Velocity
        CONST_VELOCITY = 20  # km/h
        t = self.vehicle.get_transform()
        yaw = t.rotation.yaw

        vx = 1.0
        vy = 0.0

        vx_ = vx * np.cos(yaw * np.pi / 180.0) - vy * \
            np.sin(yaw * np.pi / 180.0)
        vy_ = vy * np.cos(yaw * np.pi / 180.0) + vx * \
            np.sin(yaw * np.pi / 180.0)

        vx = vx_ * (CONST_VELOCITY / 3.6)
        vy = vy_ * (CONST_VELOCITY / 3.6)

        velocity = carla.Vector3D(x=vx, y=vy)

        self.vehicle.apply_control(control)
        self.vehicle.set_velocity(velocity)


class DiscreteWrapper(BaseWrapper):

    def __init__(self, idx, world, spawn_point, camera_settings):
        super(DiscreteWrapper, self).__init__(
            idx, world, spawn_point, camera_settings)

    def step(self, action):
        """Sets discrete transformation/teleportation

        action = [x, y, vx, vy, ax, ay, rotation]
        """
        transform = carla.Transform(
            carla.Location(action[0], action[1], 0.05),
            carla.Rotation(yaw=action[6])
        )
        self.vehicle.set_transform(transform)
        self.state.xy_velocity = (action[2], action[3])
        self.state.xy_acceleration = (action[4], action[5])

    def reset(self, reset):
        """Resets state, physic controls and sensors

        Parameters:
        ----------
        reset: array-like object
            Position coordinates x and y (can be extended)
        """
        # Read reset information
        if isinstance(reset["X"], list):
            idx = np.random.choice(len(reset["X"]))
            pos_x, pos_y = reset["X"][idx], reset["Y"][idx]
        else:
            pos_x, pos_y = reset["X"], reset["Y"]
        low_x, high_x = reset["uniformIntervalX"][0], reset["uniformIntervalX"][1]
        low_y, high_y = reset["uniformIntervalY"][0], reset["uniformIntervalY"][1]
        velocity_x = reset["velocityX"]
        sigma_velocity_x = reset["sigmaVelocityX"]
        desired_velocity = reset["desiredVelocity"]
        sigma_desired_velocity = reset["sigmaDesiredVelocity"]
        x_range = reset["rangeX"]

        # Position
        location = carla.Location(x=pos_x, y=pos_y)
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        transform = waypoint.transform
        transform.location.x += np.around(np.random.uniform(low_x, high_x), 2)
        transform.location.y += np.around(np.random.uniform(low_y, high_y), 2)
        self.vehicle.set_transform(transform)

        # Physic controls
        self.vehicle.set_velocity(carla.Vector3D())
        self.vehicle.set_angular_velocity(carla.Vector3D())
        control = self.vehicle.get_control()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        control.manual_gear_shift = False
        control.reverse = False
        self.vehicle.apply_control(control)

        # Sensors and State
        self.sensors[0].reset()
        self.sensors[1].reset()
        self.state.reset()
        self.state.driving_direction = np.sign(waypoint.lane_id)
        self.state.v_desire = np.around(
            np.clip(np.random.normal(desired_velocity,
                                     sigma_desired_velocity), 8.0, 16.0),
            2)
        self.state.xy_velocity = (np.around(
            np.clip(np.random.normal(velocity_x, sigma_velocity_x), 8.0, 16.0),
            2), 0)
        self.state.x_range = x_range

        # Enable simulation physics if disabled
        if not self.simulate_physics:
            self.togglePhysics()

    def _is_terminal(self):
        """Reviews terminal conditions"""
        diff_x_last_x = np.abs(self.state.position[0] - self.state.x_range)
        if (self.state.collision or
                self.state.delta_heading >= 25):
            return 1
        elif (self.state.elapsed_ticks >= 500 or
                diff_x_last_x < 10.0):
            return 2
        else:
            return 0
