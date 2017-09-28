from box.Box import box
from random import randint



model = box(length=5, width=5, height=80, model_time=50, space_resolution=1, visualize_system=True)
model.insert_matrix(matrix_material='silicate liquid')
for i in list(range(11)):
    model.insert_object(object='fe_diapir', object_size=0.1, x_coord=randint(0, 5), y_coord=randint(0, 5), z_coord=randint(0, 80))
for i in list(range(51)):
    print('Working on loop: {}'.format(i))
    model.update_system(deltaTime=1)