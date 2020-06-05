"""CARLA cleaning

This script provides a function to delete all actors in the simulator.
"""
import carla


def clear_carla(host, port):
    """ """
    client = carla.Client(host, port)
    world = client.get_world()

    sensors = world.get_actors().filter("sensor.*")
    vehicles = world.get_actors().filter("vehicle.*")
    walkers = world.get_actors().filter("walker.*")

    for sensor in sensors:
        sensor.destroy()

    for vehicle in vehicles:
        vehicle.destroy()

    for walker in walkers:
        walker.destroy()


if __name__ == "__main__":

    clear_carla("localhost", 6379)
