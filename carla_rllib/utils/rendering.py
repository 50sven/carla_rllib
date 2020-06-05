"""OpenDrive Rendering

This rendering is specific to a particular application and
does not represent a universal blueprint for different tasks!
However, it can be adapted to other applications.

This script provides a rendering class that uses the underlying OpenDrive Map to render images.
In addition, it provides the CARLA color map for segmentation images and a custom color map for
OpenDrive images.

Map images are pre-built with the no_rendering.py example from the CARLA distribution and loaded in __init__.
"""

import os
import cv2
import pygame
import carla
import carla_rllib
import numpy as np


# Segmentation Color Map for segmentation image rendering
CARLA_CMAP = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (190, 153, 153),
    3: (250, 170, 160),
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (107, 142, 35),
    10: (0, 0, 142),
    11: (102, 102, 156),
    12: (220, 220, 0)
}
# Segmentation Color Map for OpenDrive Map rendering
OPENDRIVE_CMAP = {
    99.0: (255, 255, 255),  # separator dummy
    0.0: (255, 255, 255),   # non-driveable / not allowed to drive
    0.0: (0, 0, 0),         # non-driveable / not allowed to drive
    -1.0: (46, 52, 54),     # static/dynamic obstacles
    -0.2: (204, 0, 0),      # 2. opposite lane
    -0.1: (239, 41, 41),    # 1. opposite lane
    0.1: (138, 226, 52),    # 1. right lane
    0.2: (115, 210, 22),    # 2. right lane
    1.0: (0, 0, 255)        # ego vehicle
}


COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)

COLOR_BLUE_0 = pygame.Color(0, 0, 255)

COLOR_VIOLETTE_0 = pygame.Color(153, 0, 153)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(46, 52, 54)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)


class OpenDriveRenderer(object):

    def __init__(self, carla_map, scenario, agent):

        self.agent = agent
        # Load map
        package_path, _ = os.path.split(os.path.abspath(carla_rllib.__file__))

        path = os.path.join(package_path + "/" + carla_map.name + ".tga")

        self.map_image = pygame.image.load(path)
        self.scale = 1.0
        self.pixels_per_meter = 10
        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        min_x = min(
            waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(
            waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin
        self.world_offset = (min_x, min_y)

        # Create surfaces
        surface_size = min(self.map_image.get_width(),
                           self.map_image.get_height())
        self.actors_surface = pygame.Surface(
            (self.map_image.get_width(), self.map_image.get_height()))
        self.actors_surface.set_colorkey(COLOR_BLACK)
        self.agent_surface = pygame.Surface((1700, 400))
        self.result_surface = pygame.Surface(
            (surface_size, surface_size))
        self.result_surface.set_colorkey(COLOR_BLACK)

    def world_to_pixel(self, location, offset=(0, 0)):
        """Returns the pixel position of a world coordinate"""
        x = self.scale * self.pixels_per_meter * \
            (location.x - self.world_offset[0])
        y = self.scale * self.pixels_per_meter * \
            (location.y - self.world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def get_image(self, vehicles):
        """Creates and preprocesses the OpenDrive image"""
        # Render actors
        self.result_surface.fill(COLOR_BLACK)
        self.actors_surface.fill(COLOR_BLACK)
        self.render_vehicles(
            self.actors_surface,
            vehicles)

        # Get ego-agent location
        agent_location_screen = self.world_to_pixel(
            self.agent.vehicle.get_location())
        translation_offset = (
            agent_location_screen[0] -
            self.agent_surface.get_width() /
            2,
            (agent_location_screen[1] -
                self.agent_surface.get_height() /
                2)
        )

        # Apply clipping rect
        clipping_rect = pygame.Rect(translation_offset[0],
                                    translation_offset[1],
                                    self.agent_surface.get_width(),
                                    self.agent_surface.get_height())
        self.actors_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

        self.result_surface.blit(
            self.map_image, (0, 0))
        self.result_surface.blit(
            self.actors_surface, (0, 0))

        self.agent_surface.fill(COLOR_BLACK)
        self.agent_surface.blit(self.result_surface, (-translation_offset[0],
                                                      -translation_offset[1]))
        orig_rect = self.agent_surface.get_rect()
        rot_image = pygame.transform.rotate(
            self.agent_surface, self.agent.state.rotation)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        image = rot_image.subsurface(rot_rect)

        # Transform surface to segmentation array
        image = pygame.surfarray.array3d(image)
        mask = {v: k for k, v in OPENDRIVE_CMAP.items() if k != 0}
        array = np.zeros((image.shape[0], image.shape[1]))
        for k, v in mask.items():
            binary_mask = (image[:, :, 0] == k[0]) & (
                image[:, :, 1] == k[1]) & (image[:, :, 2] == k[2])
            array[binary_mask] = v

        # Preprocess map
        w, h = array.shape[1], array.shape[0]
        center = (w / 2, h / 2)
        angle = 180.0
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated90 = cv2.warpAffine(array, M, (w, h))
        cropped = rotated90[50:1650, 200 - 64: 200 + 64]
        resized = cv2.resize(cropped, (128, 160),
                             interpolation=cv2.INTER_NEAREST)
        flipped = cv2.flip(resized, 1)

        return flipped

    def render_vehicles(self, surface, vehicles):
        """Renders the vehicle actors"""
        for v in vehicles:
            color = COLOR_SCARLET_RED_0
            if v[0].attributes['role_name'] == "static_obstacle":
                color = COLOR_SCARLET_RED_0
            if v[0].attributes['role_name'] == self.agent.id:
                color = COLOR_CHAMELEON_0
            # Compute bounding box points
            bb = v[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y),
                       carla.Location(x=bb.x, y=-bb.y),
                       carla.Location(x=bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=-bb.y)
                       ]
            v[1].transform(corners)
            corners = [self.world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)
