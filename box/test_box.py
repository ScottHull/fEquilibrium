from box.Box import box



model = box(length=3, width=3, height=5, model_time=10, space_resolution=1, visualize_system=True)
model.insert_matrix(matrix_material='silicate liquid')
model.insert_object(object='fe diapir', object_size=0.1, x_coord=1, y_coord=1, z_coord=0)
model.insert_object(object='fe diapir', object_size=0.1, x_coord=1, y_coord=2, z_coord=1)
for i in list(range(11)):
    print('Working on loop: {}'.format(i))
    model.update_system(deltaTime=1)