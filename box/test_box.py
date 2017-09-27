from box.Box import box



model = box(length=100, width=100, height=100, model_time=10, space_resolution=1)
model.insert_matrix(matrix_material='silicate liquid')
model.insert_object(object='fe diapir', object_size=0.1, x_coord=50, y_coord=50, z_coord=1)
for i in list(range(100)):
    print('Working on loop: {}'.format(i))
    model.update_system(deltaTime=i)