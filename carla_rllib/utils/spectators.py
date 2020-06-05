"""Spectator

This script allows the user to operate a spectator camera.
It enables to switch between all agents in the environment.

"""

import time
import os
import queue
import numpy as np
import pygame
import carla
from carla import ColorConverter as cc
from pygame.locals import K_ESCAPE
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_q


class ActorSpectator(object):
    def __init__(self, world, args, integrated=False, recording=False, record_path="~/no_backup/Images/"):
        self.world = world
        self.sensor = None
        self.integrated = integrated
        self.queue = queue.Queue()
        self.width = args.width
        self.height = args.height
        self.fov = args.fov
        self.gamma = args.gamma
        self.location = args.location
        self.rotation = args.rotation
        self.surface = None
        self.hud = None
        self.index = 0
        self.recording = recording
        self.record_path = record_path
        self.file_num = 0

        self._initialize_world()
        self._initialize_pygame()
        self._initialize_blueprint()
        self._set_camera(self.index)

    def _initialize_world(self):
        """ """
        # Wait for tick
        if not self.integrated:
            self.world.wait_for_tick(5.0)
        # Enable rendering if not yet done
        self.settings = self.world.get_settings()
        if self.settings.no_rendering_mode:
            _ = self.world.apply_settings(carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=self.settings.synchronous_mode,
                fixed_delta_seconds=self.settings.fixed_delta_seconds))
        # Get all agents
        self.actors = [actor
                       for actor in self.world.get_actors().filter("vehicle.*")
                       if "Agent" in actor.attributes["role_name"]]
        self.actors = sorted(self.actors, key=lambda x: x.id)

    def _initialize_pygame(self):
        """Initializes the pygame window"""
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

    def _initialize_blueprint(self):
        """Initializes the camera blueprint"""
        self.bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.bp.set_attribute('image_size_x', str(self.width))
        self.bp.set_attribute('image_size_y', str(self.height))
        self.bp.set_attribute('fov', str(self.fov))
        if self.bp.has_attribute('gamma'):
            self.bp.set_attribute('gamma', str(self.gamma))

    def _set_camera(self, index):
        """Sets the camera sensor"""
        index = index % len(self.actors)

        if self.sensor is not None:
            self.sensor.destroy()
            self.surface = None

        self.sensor = self.world.spawn_actor(self.bp, carla.Transform(
            carla.Location(x=self.location[0],
                           y=self.location[1],
                           z=self.location[2]),
            carla.Rotation(yaw=self.rotation[0],
                           pitch=self.rotation[1],
                           roll=self.rotation[2])),
            attach_to=self.actors[index],
            attachment_type=carla.AttachmentType.Rigid)

        self.sensor.listen(self.queue.put)
        self.index = index

    def render(self, frame, image=None):
        """Renders a spectator window and allows to display net inputs

        Parameters:
        ----------
        frame: int
            current frame to retrieve correct image
        image: numpy.ndarray
            net_input to be displayed
        """
        # Render spectator window
        array = self._retrieve_data(frame)
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.surface is not None:
            self.display.blit(self.surface, (0, 0))

        # Render net input(s)
        if image is not None:
            image = image[:, :, 0] if image.shape[-1] == 1 else image
            self.surface_input = pygame.surfarray.make_surface(
                image.swapaxes(0, 1))
            self.display.blit(self.surface_input, (20, 444))

        pygame.display.flip()

        # Save pygame display if you want
        if self.recording:
            self.file_num += 1
            filename = os.path.expanduser(
                self.record_path + "image_%04d.png" % self.file_num)
            pygame.image.save(self.display, filename)

    def parse_events(self):
        """Parse the keyboard inputs"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 1
            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    return 1
                if event.key == K_q:
                    return 2
        if any(x != 0 for x in pygame.key.get_pressed()):
            self._parse_keys(pygame.key.get_pressed())

    def _parse_keys(self, keys):
        """Controls the camera focus"""
        prev_index = self.index
        if keys[K_RIGHT]:
            self.index += 1
        if keys[K_LEFT]:
            self.index -= 1

        if prev_index != self.index:
            self._set_camera(self.index)

        time.sleep(0.3)

    def _retrieve_data(self, frame):
        """Returns the image data"""
        while True:
            try:
                image = self.queue.get(timeout=0.5)
                if image.frame == frame:
                    self.image = self._preprocess_data(image)
                    return self.image
            except:
                return np.zeros((1024, 720, 3))

    def _preprocess_data(self, image):
        """Process and returns the image data"""
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def destroy(self):
        """Destroys the camera sensor and quits pygame"""
        self.sensor.destroy()
        pygame.quit()
        _ = self.world.apply_settings(self.settings)


if __name__ == "__main__":

    import argparse

    argparser = argparse.ArgumentParser(
        description='CARLA Spectator')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--fov',
        metavar='FOV',
        default=100.0,
        type=float,
        help='Field of camera view (default: 100.0)')
    argparser.add_argument(
        '--gamma',
        metavar='GAMMA',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--location',
        metavar='LOCATION',
        nargs='+',
        default=[-8.0, 0.0, 6.0],
        type=float,
        help='Position of the camera (x, y, z) (default: -8.0 0.0 6.0)')
    argparser.add_argument(
        '--rotation',
        metavar='ROTATION',
        nargs='+',
        default=[0.0, -30.0, 0.0],
        type=float,
        help='Rotation of the camera (yaw, pitch, roll) (default: 0.0 -30.0 0.0)')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        world = client.get_world()

        spectator = ActorSpectator(world, args)

        while True:
            snapshot = world.wait_for_tick(10.0)
            if spectator.parse_events():
                break
            spectator.render(snapshot.frame)

    finally:
        spectator.destroy()
