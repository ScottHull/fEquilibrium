from box.Box import box
from random import randint


model = box(length=10, width=10, height=5, model_time=20, space_resolution=1, visualize_system=True, object_history=True)
model.insert_matrix(matrix_material='Silicate Liquid')
for i in list(range(1)):
    model.insert_object(object='Metal Liquid', object_radius=0.1, x_coord=2,
                        y_coord=2, z_coord=0, initial_mass=1)
model.update_system()