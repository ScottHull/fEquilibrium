import pandas as pd
import numpy as np
from math import pi, sqrt, exp
import time
import os
os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from meta.Console import console

# Some methods extracted from:
# Mechanisms of metal-silicate equilibration in the terrestrial magma ocean
#   D.C. Rubie a;, H.J. Melosh b, J.E. Reid a, C. Liebske a, K. Righter b



class gravity:
    """
    Calculates the gravity a body is subjected to as a function of depth.
    """

    def __init__(self):
        pass


class move_particle:

    def __init__(self, body_type, system_params):
        self.body_type = body_type
        self.system_params = system_params

    def viscosity(self, material, pressure, temperature):
        """
        A calculation of viscosity using the diffusion coefficient.  Diffusion is an act of Gibbs Free Energy minimization,
            where atoms diffuse down a concentration gradient to minimum energy configuration.  Diffusion is related to
            the viscosity of the material.
        :param material: the name of the material, as listed in 'physical_parameters.csv' in this file's working directory
        :return: viscosity, Pa*s = (N*s)/m^2=kg/(s*m)
        """
        material_properties = pd.read_csv("dynamics/physical_parameters.csv", index_col='Material')
        if pd.isnull(material_properties['Viscosity'][material]):
            gas_const = 8.312  # J/mol*k
            boltzmann_const = 1.3806 * 10**-23 # (m^2*kg)/(s^2*degK)
            D_not = material_properties['D_not'][material] # diffusion equation param, the diffusion coefficient
            H_star = material_properties['H*'][material] # diffusion equation param, the activation enthalpy
            V_star = material_properties['V*'][material] # diffusion equation param, the activation volume
            lambda_ = material_properties['Lambda_O'][material] # viscosity param, the jump distance for diffusing ions
            D = D_not * exp((-H_star + (pressure * V_star)) / (gas_const * temperature)) # the diffusion equation, D=D_not*exp(-(H*+PV*)/R*T)
            viscosity = (boltzmann_const * temperature) / (D * lambda_) # reworked from D_(Si,O)=(boltzmann*T)/(D*lambda)
            print("calculated viscosity: {} (pressure={}, temperature={})".format(viscosity, pressure, temperature))
            return viscosity
        else:
            viscosity = material_properties['Viscosity'][material]
            return viscosity

    def friction_coefficient(self, density_body, density_matrix, diameter_body, matrix_viscosity):
        """
        A dimensionless parameter to determine the mode of Stoke's settling.
        :param density_body: density of the body, in kg/m^3
        :param density_matrix: density of the matrix, in kg/m^3
        :param diameter_body: diameter of the body, in m
        :param matrix_viscosity: viscosity of the matrix, in Pa*s
        :return: f, friction coefficient (dimensionless)
        """
        grav_constant = 9.81 # m/s^2
        f = (pi/6) * ((density_body - density_matrix) / density_matrix) * \
            (density_matrix / matrix_viscosity)**2 * (grav_constant * diameter_body**3)
        return f

    # def stokes_settling(self, density_body, density_matrix, diameter_body, matrix_viscosity, drag_coeff=0.2):
    # TODO: return 3-component array such that an x, y, and z velocity are received
    def stokes_settling(self, object, object_radius, matrix_material, matrix_material_temp, matrix_material_pressure):
        """
        If friction coefficient, F, is <10 or >10, the modes of Stoke's settling are described as below as a result of
            differences in laminar versus turbulent flow.
        :param density_body: density of the body, in kg/m^3
        :param density_matrix: density of the matrix, in kg/m^3
        :param diameter_body: diameter of the body, in m
        :param matrix_viscosity: viscosity of the matrix, in Pa*s
        :param drag_coeff: asymptotic value of 0.2 at high frictional coefficients
        :return: v, settling velocity
        """

        material_properties = pd.read_csv("dynamics/physical_parameters.csv", index_col='Material')
        density_body = material_properties['Density'][object]
        density_matrix = material_properties['Density'][matrix_material]
        drag_coeff = material_properties['Drag Coefficient'][object]
        matrix_viscosity = self.viscosity(material=matrix_material, pressure=matrix_material_pressure,
                                        temperature=matrix_material_temp)
        diameter_body = object_radius * 2.0 # diameter=radius*2
        grav_constant = 9.81 # m/s^2
        f = self.friction_coefficient(density_body=density_body, density_matrix=density_matrix,
                                      diameter_body=diameter_body, matrix_viscosity=matrix_viscosity)
        if f < 10: # low frictional coefficient, when body is in laminar flow regime
            v = ((density_body - density_matrix) * grav_constant * diameter_body**2) / (18 * matrix_viscosity) # calculates the velocity of the body
            # console.pm_flush(
            #     "f: {}, velocity: {}, matrix_viscosity: {}, matrix_material: {}".format(f, v, matrix_viscosity,
            #                                                                             matrix_material))
            return v
        else:
            v = sqrt(((4 / (3 * drag_coeff)) * (((density_body - density_matrix) / density_matrix) * (grav_constant * diameter_body))))
            # console.pm_flush(
            #     "f: {}, velocity: {}, matrix_viscosity: {}, matrix_material: {}".format(f, v, matrix_viscosity,
            #                                                                             matrix_material))
            return v


# currently not in use
class droplet_size:

    """
    Predicts Fe-molten-alloy droplet size via the dimensionless Weber number.
    W = ((rho_m - rho_s)*d*v^2)/sigma.
    Settling velocity determined via Stoke's Law, when flow regime is lamellar or equation incorporates a drag
        coefficient when flow around falling droplet is turbulent.
    """
    def __init__(self, body_type):
        self.body_type = body_type

    def weber_number(self, density_body, density_matrix, diameter_body, surface_energy, settling_velocity):
        """
        A dimensionless number that is a ratio of stagnation pressure and internal pressure caused by surface tension.
        This number determines when a falling droplet becomes stable as it escapes fluid instabilities such as Rayleigh-Taylor
        or Kelvin-Helmholtz.  Usually stable when Weber number falls to ~10.

        :param density_body:
        :param density_matrix:
        :param diameter_body:
        :param surface_energy:
        :param settling_velocity:
        :return: w, Weber number (dimensionless)
        """
        w = ((density_body - density_matrix) * diameter_body * settling_velocity) / surface_energy
        return w
