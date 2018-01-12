from math import pi
from diffusivity_example.spherical_diffusivity import sphere


max_r = 3  # in m
max_theta = 2 * pi  # in radians
max_phi = 2 * pi  # in radians

radial_mesh_res = 0.01  # in m
angular_mesh_res = pi / 6

diapir_r = .01


s = sphere(max_r=max_r, max_theta=max_theta, max_phi=max_phi, radial_mesh_res=radial_mesh_res,
           angular_mesh_res=angular_mesh_res, diapir_r=diapir_r)