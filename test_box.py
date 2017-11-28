from box.Box import box
from random import randint, uniform, choice
import math
import os
import numpy as np

width = 3.0
length = 3.0
height = 5.0
space_resolution = 0.5
# x_coords_range = np.arange(0, round((length + space_resolution), len(str(space_resolution))), space_resolution) # generate range of x-coords
# y_coords_range = np.arange(0, round((width + space_resolution), len(str(space_resolution))), space_resolution) # generate range of y-coords
# z_coords_range = np.arange(0, round((height + space_resolution), len(str(space_resolution))), space_resolution) # generate range of z-coords


# instantiate the box
model = box(length=length, width=width, height=height, model_time=10, space_resolution=space_resolution,
            visualize_system=True, object_history=True, visualize_neighbors=False, animate_neighbors=False)

# insert one more more matrix materials into the box, will populate all possible coordinate positions
model.insert_matrix(matrix_material='Silicate Liquid', composition={'SiO2': 50, 'FeO': 50, '182-Hf': 100},
                    z_range=[0,4], initial_temperature=2000)
# model.insert_matrix(matrix_material='Denser Silicate Liquid', composition={'SiO2': 50, 'FeO': 50, '182-Hf': 100},
#                     z_range=[20,30], initial_temperature=2000)

# insert boundary with conditions at a specified z range
model.insert_boundary(temperature=2200, z_range=[0,0.5], boundary_location='top')
model.insert_boundary(temperature=2200, z_range=[4.5,5], boundary_location='bottom')

# insert X number of objects into the box, specify their location, will overwrite matrix at that point
for i in list(range(1)):
    model.insert_object(object='Metal Liquid', object_radius=0.1, x_coord=2.5,
                        y_coord=2.5, z_coord=1, composition={'SiO2': 20, 'FeO': 80}, initial_temperature=2200)

# automatically update the box iteratively to model_time = 0
# a time step is automatically calculated if one is not provided with the deltaTime argument
model.update_system()
