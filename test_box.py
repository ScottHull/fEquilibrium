from box.Box import box
from random import randint
import os


# create the box
model = box(length=3, width=3, height=10, model_time=20, space_resolution=0.2, visualize_system=True, object_history=True)

# insert the matrix matrial into the box, will populate all possible coordinate positions
model.insert_matrix(matrix_material='Silicate Liquid', composition={'SiO2': 50, 'FeO': 50, '182-Hf': 100}, z_range=[0,4], initial_temperature=2000)
model.insert_matrix(matrix_material='Denser Silicate Liquid', composition={'SiO2': 50, 'FeO': 50, '182-Hf': 100}, z_range=[5,10], initial_temperature=2000)

# insert X number of objects into the box, specify their location, will overwrite matrix at that point
for i in list(range(1)):
    model.insert_object(object='Metal Liquid', object_radius=0.01, x_coord=2,
                        y_coord=2, z_coord=0, composition={'SiO2': 20, 'FeO': 80}, initial_temperature=2000)

# automatically iteratively update the box to model_time = 0
model.update_system()