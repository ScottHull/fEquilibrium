from box.Box import box
from random import randint
import os


# instantiate the box
model = box(length=3, width=3, height=30, model_time=50, space_resolution=0.2, visualize_system=True, object_history=True)

# insert one more more matrix materials into the box, will populate all possible coordinate positions
model.insert_matrix(matrix_material='Silicate Liquid', composition={'SiO2': 50, 'FeO': 50, '182-Hf': 100},
                    z_range=[0,15], initial_temperature=2000)
model.insert_matrix(matrix_material='Denser Silicate Liquid', composition={'SiO2': 50, 'FeO': 50, '182-Hf': 100},
                    z_range=[15,30], initial_temperature=2000)

# insert X number of objects into the box, specify their location, will overwrite matrix at that point
for i in list(range(1)):
    model.insert_object(object='Metal Liquid', object_radius=0.01, x_coord=2,
                        y_coord=2, z_coord=0, composition={'SiO2': 20, 'FeO': 80}, initial_temperature=2000)

# automatically update the box iteratively to model_time = 0
# a time step is automatically calculated if one is not provided with the deltaTime argument
model.update_system()
