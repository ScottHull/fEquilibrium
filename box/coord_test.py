import numpy as np

coords = []
x_coords_range = np.arange(1, 11, 1)
y_coords_range = np.arange(1, 11, 1)
z_coords_range = np.arange(1, 11, 1)
print(x_coords_range)
print(y_coords_range)
print(z_coords_range)
for i in x_coords_range:
    for j in y_coords_range:
        for q in z_coords_range:
            temp_coords = []
            temp_coords.append(i)
            temp_coords.append(j)
            temp_coords.append(q)
            coords.append(temp_coords)


print(coords)
print(len(coords))