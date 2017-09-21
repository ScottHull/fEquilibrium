import pandas as pd
import numpy as np
from math import pi



class gravity:

    def __init__(self):
        pass


class move_particle:

    def __init__(self, body_type):
        self.body_type = body_type

    def friction_coefficient(self, density_body, density_matrix, diameter_body, matrix_viscosity):
        """
        :param density_body: density of the body, in kg/m^3
        :param density_matrix: density of the matrix, in kg/m^3
        :param diameter_body: diameter of the body, in m
        :param matrix_viscosity: viscosity of the matrix, in Pa*s
        :return: f, friction coefficient (dimensionless)
        """
        grav_constant = 9.81 # m/s^2
        f = (pi/6) * ((density_body - density_matrix) / density_matrix) * \
            (density_matrix / matrix_viscosity)**2 * (grav_constant * diameter_body*3)
        return f

    def stokes_settling(self, density_body, density_matrix, diameter_body, matrix_viscosity):
        """
        :param density_body: density of the body, in kg/m^3
        :param density_matrix: density of the matrix, in kg/m^3
        :param diameter_body: diameter of the body, in m
        :param matrix_viscosity: viscosity of the matrix, in Pa*s
        :return: v, settling velocity
        """
        grav_constant = 9.81 # m/s^2
        f = self.friction_coefficient(density_body=density_body, density_matrix=density_matrix,
                                      diameter_body=diameter_body, matrix_viscosity=matrix_viscosity)
        if f < 10:
            v = ((density_body - density_matrix) * g)


class droplet_size:

    """
    Predicts Fe-molten-alloy droplet size via the dimensionless Weber number.
    W = ((rho_m - rho_s)*d*v^2)/sigma.
    Settling velocity determined via Stoke's Law, when flow regime is lamellar or equation incorporates a drag
        coefficient when flow around falling droplet is turbulent.
    """
    def __init__(self, body_type):
        self.body_type = body_type

    def weber_size(self, density_body, density_matrix, diameter_body, surface_energy, settling_velocity):
        """
        :return: Weber number (dimensionless)
        """
        w = ((density_body - density_matrix) * diameter_body * settling_velocity) / surface_energy
        return w
