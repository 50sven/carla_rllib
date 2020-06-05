"""CARLA Sensor Suite

This script provides various sensor classes to get data from the CARLA simulator.

Classes:
    * CameraSensor - sensor for rgb, segmentation and depth images
    * LidarSensor - sensor for lidar data
    * GnssSensor - sensor for longitudinal and latitudinal coordinates
    * LaneInvasionSensor - sensor to capture lane invasions (currently very limited)
    * CollisionSensor - sensor to capture collisions
Functions:
    * get_actor_name - extracts the name of an actor
"""

import queue
import pygame
import weakref
import numpy as np
import carla
from carla import ColorConverter as cc


def get_actor_name(actor, truncate=250):
    """Returns the name of an actor"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class CameraSensor(object):

    def __init__(self, parent_actor, type_id, spec=None,
                 width=128, height=72, fov=90.0,
                 location=[1.3, 0.0, 1.4], rotation=[0.0, -25.0, 0.0]):
        self.sensor = None
        self.queue = queue.Queue()
        self.world = parent_actor.get_world()

        self.spec = self._get_spec(spec)
        blueprint = self._get_blueprint(type_id, width, height, fov)

        self.sensor = self.world.spawn_actor(blueprint, carla.Transform(
            carla.Location(x=location[0], y=location[1], z=location[2]),
            carla.Rotation(yaw=rotation[0], pitch=rotation[1], roll=rotation[2])),
            attach_to=parent_actor,
            attachment_type=carla.AttachmentType.Rigid)
        self.sensor.listen(self.queue.put)

    def _get_spec(self, spec):
        """Returns the specification for converting raw image data"""
        if spec is None or spec == 'raw':
            return cc.Raw
        elif spec == 'city_scapes':
            return cc.CityScapesPalette
        elif spec == 'depth':
            return cc.Depth
        elif spec == 'log_depth':
            return cc.LogarithmicDepth
        else:
            raise ValueError(
                "Image conversion specification \'{}\' not available. Use \'raw\', \'city_scapes\', \'depth\' or \'log_depth\'".format(spec))

    def _get_blueprint(self, type_id, width, height, fov):
        """Returns the camera blueprint"""
        if type_id == 'rgb':
            type_id = 'sensor.camera.rgb'
        elif type_id == 'depth':
            type_id = 'sensor.camera.depth'
        elif type_id == 'segmentation':
            type_id = 'sensor.camera.semantic_segmentation'
        else:
            raise ValueError(
                "Camera type \'{}\' not available. Use \'rgb\', \'depth\' or \'segmentation\'".format(spec))

        bp = self.world.get_blueprint_library().find(type_id)
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        bp.set_attribute('fov', str(fov))

        return bp

    def retrieve_data(self, frame, timeout):
        """Returns the image data"""
        while True:
            image = self.queue.get(timeout=timeout)
            if image.frame == frame:
                image = self._preprocess_data(image)
                return image

    def _preprocess_data(self, image):
        """Process and returns the image data"""
        image.convert(self.spec)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        if 'segmentation' in self.sensor.type_id and self.spec == cc.Raw:
            array = array[:, :, 0, np.newaxis]
        return array


class LidarSensor(object):
    def __init__(self, parent_actor, width=128, height=72,
                 location=[-5.5, 0.0, 2.5], rotation=[0.0, 8.0, 0.0]):
        self.sensor = None
        self.world = parent_actor.get_world()
        self.width = width
        self.height = height
        self.queue = queue.Queue()

        # Initialize Sensor and start to listen
        blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        blueprint.set_attribute('range', '5000')
        self.sensor = self.world.spawn_actor(blueprint, carla.Transform(
            carla.Location(x=location[0], y=location[1], z=location[2]),
            carla.Rotation(yaw=rotation[0], pitch=rotation[1], roll=rotation[2])),
            attach_to=parent_actor,
            attachment_type=carla.AttachmentType.SpringArm)
        self.sensor.listen(self.queue.put)

    def retrieve_data(self, frame, timeout):
        """Returns the lidar data"""
        while True:
            image = self.queue.get(timeout=timeout)
            if image.frame == frame:
                image = self._preprocess_data(image)
                return image

    def _preprocess_data(self, image):
        """Process and returns the lidar data"""
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self.width, self.height) / 100.0
        lidar_data += (0.5 * self.width, 0.5 * self.height)
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (self.width, self.height, 3)
        lidar_img = np.zeros((lidar_img_size), dtype=int)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        return lidar_img


class GnssSensor(object):
    def __init__(self, parent_actor, location=[1.0, 0.0, 1.0]):
        self.sensor = None
        self.queue = queue.Queue()

        # Initialize Sensor and start to listen
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=location[0], y=location[1], z=location[2])),
            attach_to=parent_actor)
        self.sensor.listen(self.queue.put)

    def retrieve_data(self, frame, timeout):
        """Returns the gnss data"""
        while True:
            data = self.queue.get(timeout=timeout)
            if data.frame == frame:
                data = self._preprocess_data(data)
                return data

    def _preprocess_data(self, data):
        """Returns longitudinal and latitudinal coordinates"""
        return (data.latitude, data.longitude)


class LaneInvasionSensor(object):

    """
    https://github.com/carla-simulator/carla
    """
    def __init__(self, parent_actor):

        # This sensor is a work in progress, currently very limited.

        self.sensor = None
        self.history = []
        self.last_event_frame = 0
        self.crossed_lane_counter = 0

        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(),
                                        attach_to=parent_actor)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    def retrieve_data(self, frame, timeout):
        """Returns the number of lane invasions"""
        return self.crossed_lane_counter

    def reset(self):
        """Resets the lane invasion counter"""
        self.crossed_lane_counter = 0
        self.last_event_frame = 0

    @staticmethod
    def _on_invasion(weak_self, event):
        """Records lane invasions"""
        self = weak_self()
        if not self:
            return
        if event.crossed_lane_markings:
            # Current workaround to count and store crossed lane markings:
            # Prevent crossed lane markings from being counted several times
            # by ignoring all subsequent events within a period of 20 frames.
            if (event.frame - self.last_event_frame > 20):
                self.last_event_frame = event.frame
                self.crossed_lane_counter += 1
                self.history.append(event.crossed_lane_markings[0].type.name)


class CollisionSensor(object):

    """
    https://github.com/carla-simulator/carla
    """

    def __init__(self, parent_actor):
        self.sensor = None
        self.collision = None
        self.other_actor = None

        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(),
                                        attach_to=parent_actor)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def retrieve_data(self, frame, timeout):
        """Returns the collision record"""
        return (self.collision, self.other_actor)

    def reset(self):
        """Resets the collision record"""
        self.collision = False
        self.other_actor = None

    @staticmethod
    def _on_collision(weak_self, event):
        """Records (non-)collision events"""
        self = weak_self()
        if not self:
            return
        self.other_actor = get_actor_name(event.other_actor)
        self.collision = False if self.other_actor == "Sidewalk" else True
