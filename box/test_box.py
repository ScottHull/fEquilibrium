from box.Box import box
from random import randint


model = box(length=3, width=3, height=3, model_time=5, space_resolution=1, visualize_system=True)
model.insert_matrix(matrix_material='Silicate Liquid')
for i in list(range(1)):
    model.insert_object(object='Metal Liquid', object_radius=0.1, x_coord=randint(0, 3),
                        y_coord=randint(0, 3), z_coord=0, initial_mass=1)
print(model.update_system())