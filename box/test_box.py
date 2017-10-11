from box.Box import box
from random import randint
import os


model = box(length=8, width=8, height=5, model_time=15, space_resolution=1, visualize_system=True, object_history=True)
model.insert_matrix(matrix_material='Silicate Liquid')
for i in list(range(1)):
    model.insert_object(object='Metal Liquid', object_radius=0.1, x_coord=4,
                        y_coord=4, z_coord=0, initial_mass=1)
model.update_system()