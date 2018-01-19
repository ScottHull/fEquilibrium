from box.Box import box
from random import randint, uniform, choice
import math
import os
import numpy as np


# set the box width (x), length (y), height (z), and the spatial resolution (all in m)
width = 3.0
length = 3.0
height = 4
space_resolution = 0.2

# use these to generate random coordinates in the specified range, if you wish
# x_coords_range = np.arange(0, round((length + space_resolution), len(str(space_resolution))), space_resolution) # generate range of x-coords
# y_coords_range = np.arange(0, round((width + space_resolution), len(str(space_resolution))), space_resolution) # generate range of y-coords
# z_coords_range = np.arange(0, round((height + space_resolution), len(str(space_resolution))), space_resolution) # generate range of z-coords

# instantiate the box
model = box(length=length,
            width=width,
            height=height,
            model_time=5,
            space_resolution=space_resolution,
            fO2_buffer='IW',
            visualize_system=True,
            object_history=True,
            visualize_neighbors=False,
            animate_neighbors=False)

# insert one more more matrix materials into the box, will populate all possible coordinate positions
# boundary layers do not change in any property, nor can sinking objects penetrate them
model.insert_matrix(matrix_material='Silicate Liquid',
                    composition={'SiO2': 50, 'FeO': 50, '182-Hf': 100},
                    z_range=[0,4],
                    initial_temperature=2000,
                    temperature_gradient=2,
                    initial_pressure=(1 * 10 ** 9),
                    pressure_gradient=2,
                    initial_fO2=(-1.2),
                    fO2_gradient=(-.01))

# insert boundary with conditions at a specified z range
model.insert_boundary(temperature=2000,
                      z_range=[0,0.2],
                      boundary_location='top')
model.insert_boundary(temperature=2200,
                      z_range=[3.8,4],
                      boundary_location='bottom')

# insert X number of objects into the box, specify their location, will overwrite matrix at that point
for i in list(range(1)):  # this just provides a loop in case you'd like to insert many diapirs w/ minimal code
    model.insert_object(object='Metal Liquid',
                        object_radius=0.01,
                        x_coord=1.6,
                        y_coord=1.6,
                        z_coord=0.6,
                        composition={'SiO2': 20, 'FeO': 80},
                        initial_temperature=2200)
    model.insert_object(object='Metal Liquid',
                        object_radius=0.08,
                        x_coord=1.6,
                        y_coord=1.6,
                        z_coord=0.4,
                        composition={'SiO2': 20, 'FeO': 80},
                        initial_temperature=2200)

# automatically update the box iteratively to model_time = 0
# a time step is automatically calculated if one is not provided with the deltaTime argument
# (currently just calculates 1 second)
model.update_system()
