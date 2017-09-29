from box.Box import box
from random import randint
import random


model = box(length=3, width=3, height=3, model_time=5, space_resolution=1, visualize_system=True)
model.insert_matrix(matrix_material='silicate liquid')
for i in list(range(2)):
    model.insert_object(object='fe_diapir', object_size=0.1, x_coord=randint(0, 3),
                        y_coord=randint(0, 3), z_coord=randint(0, 3))
model.update_system()
# for i in list(range(11)):
#     print('Working on loop: {}'.format(i))
#     model.update_system(deltaTime=1)