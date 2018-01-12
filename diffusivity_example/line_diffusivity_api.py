from math import pi
from diffusivity_example.line_diffusivity import line


max_r = 0.5  # in m
radial_mesh_res = 0.01
diapir_r = 0.05


s = line(max_r=max_r, radial_mesh_res=radial_mesh_res, model_time=30, time_interval=1, partitioning_coefficient=20)
s.insert_diapir(radius=diapir_r, concentration=0, diffusivity=(10**-6))
s.insert_melt(concentration=100, diffusivity=(10**-6))
s.insert_boundary(concentration_diapir=.01, concentration_melt=100)
s.update_model()