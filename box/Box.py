import os
import matplotlib as mpl
mpl.use('Qt5Agg')
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
import numpy as np
import pandas as pd
from random import randint
import moviepy.editor as mpy
from dynamics.Movement import move_particle
from dynamics.Energy import energy, thermal_eq
from thermodynamics.Solution import solution
from meta.Console import console
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import sys
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.colorbar
from math import pi
from numbers import Number
import ast
from collections import Counter


# TODO: update some methods to class methods to avoid outside interference
class box:
    def __init__(self, length, width, height, space_resolution, model_time, fO2_buffer=None, visualize_system=False,
                 object_history=False, visualize_neighbors=False, animate_neighbors=False):
        """
        Instantiates the box.
        :param length: length of the system, in m
        :param width: width of the system, in m
        :param height: height of the system, in m
        :param space_resolution: spatial resolution of the system, in m
        :param model_time: initial time of the system, in 'years ago'
        :param visualize_system: optional parameter to turn on movie generation of the evolution of the system

        """
        console.pm_header("\n\n\nfEquilibrium\nScott D. Hull, 2017\n\n")
        console.pm_stat("Instantiating box. Please sit tight.")
        self.visualize_neighbors = visualize_neighbors  # option for generating frames of each point's nearest neighbor
        self.animate_neighbors = animate_neighbors  # option to stich together neighbor frames as an animation
        self.length = length  # x values of box
        self.width = width  # y values of box
        self.height = height  # z values of box
        self.fO2_buffer = fO2_buffer
        self.model_base = height  # sets the model base as the coords directly above the boundary layer
        self.model_top = 0  # sets the model top as the coords directly below the boundary layer
        self.boundary_vals = []  # stores limits of boundaries so that box integrity can be verified
        self.space_resolution = space_resolution  # the spatial resolution of the box
        self.model_time = float(model_time)  # the amount of time the model will run
        self.initial_time = float(model_time)  # the initial value of self.model_time
        # generates all possible coordinate points within the box
        self.coords = self.generate_coordinate_points(length=self.length, width=self.width, height=self.height,
                                                      space_resolution=self.space_resolution)
        self.visualize_system = visualize_system  # True/False, create animations of the box?
        self.object_history = object_history  # creates an output file that tracks objects with time
        # this is the central dataframe of the model
        # highly critical that this is accessible in memory for all processes
        self.space = pd.DataFrame({
            'coord_index': [str(i) for i in self.coords],
            'object_id': np.NAN,  # randomly generated object id tag to identify unique elements in box
            'object': np.NAN,  # name of the object, as defined in self.physical_parameters
            'x_coords': [float(i[0]) for i in self.coords],
            'y_coords': [float(i[1]) for i in self.coords],
            'z_coords': [float(i[2]) for i in self.coords],
            'nearest_neighbors': np.NAN,  # x, y, and z neighbors in each direction
            'object_radius': np.NAN,  # in m
            'density': np.NAN,  # in kg/m^3
            'temperature': np.NAN,  # in K
            'heat_generated': np.NAN,  # in K
            'pressure': np.NAN, # pressure in pascals, in order to work with ideal gas law
            'fO2_({})'.format(self.fO2_buffer): np.NAN, # log units below the IW buffer
            'object_velocity': [float(0) for i in self.coords],
            'rounded_object_velocity': np.NAN,
            'x_direct': np.NAN,  # in m
            'y_direct': np.NAN,  # in m
            'z_direct': np.NAN,  # in m
            'potential_energy': np.NAN,  # in J
            'kinetic_energy': np.NAN,  # in J
            'total_energy_released': np.NAN,  # in J
            'mass': np.NAN,  # in kg
            'volume': np.NAN,  # in m^3
            'drag_force': np.NAN,  # in N, the drag force exerted on sinking objects
            'buoyant_force': np.NAN,  # in N, the buoyant force exerted on particles (negative = downward buoyant force)
            'gravitational_force': np.NAN  # in N, the force pulling down on the objects due to gravity
        })
        self.num_coords = len(self.coords)
        self.solution = solution(box_length=self.num_coords)
        self.physical_parameters = pd.read_csv(
            os.path.dirname(os.path.abspath('.')) + "/fEquilibrium/dynamics/physical_parameters.csv",
            index_col='Material')
        # these are lists and directories for tracking animation frame ordering and storing animation frames
        self.movie_frames1 = []
        self.movie_frames2 = []
        self.movie_frames3 = []
        self.movie_frames4 = []
        if os.path.exists('object_dynamics'):
            shutil.rmtree('object_dynamics')
        if os.path.exists('thermal_equilibrium_heatmap'):
            shutil.rmtree('thermal_equilibrium_heatmap')
        if os.path.exists('nearest_neighbors'):
            shutil.rmtree('nearest_neighbors')
        if os.path.exists('temp_distrib_floor'):
            shutil.rmtree('temp_distrib_floor')
        os.mkdir('object_dynamics')
        os.mkdir('thermal_equilibrium_heatmap')
        os.mkdir('nearest_neighbors')
        os.mkdir('temp_distrib_floor')
        # opens the object history csv so object histories can be written after each time interval
        if self.object_history is True:
            if "object_history.csv" in os.listdir(os.getcwd()):
                os.remove("object_history.csv")
            self.object_output = open("object_history.csv", 'w')
            header = ['Model Time']
            for i in self.space.columns.tolist():
                header.append(str(i))
            formatted_header = ",".join(i for i in header)
            self.object_output.write("{}\n".format(formatted_header))
        # opens the object velocity csv so object velocities can be written after each time interval
        if 'object_velocities.csv' in os.listdir(os.getcwd()):
            os.remove('object_velocities.csv')
        self.velocity_output = open('object_velocities.csv', 'a')

    # returns a copy of the self.space dataframe
    def get_box(self):
        return self.space

    def classify_neighbors(self, animate_neighbors, visualize_neighbors):
        """
        classifies nearest neighbors, primarily for heat equilibrium
        assumption is that each point in the box is only in contact with its nearest neighbor
        this only executes once at the initial model time
        :param animate_neighbors:
        :param visualize_neighbors:
        :return: None
        """
        loop_total = len(self.space.index.tolist())
        console.pm_stat("Finding nearest neighbors for all points.  This may take several minutes...")
        # print("Finding nearest neighbors for all points.  This may take several minutes...")
        min_xcoords = 0.0
        max_xcoords = float(self.length)
        min_ycoords = 0.0
        max_ycoords = float(self.width)
        min_zcoords = 0.0
        max_zcoords = float(self.height)
        # iterates through each coordinate point in the box and identifies the nearest neighbors
        for row in self.space.itertuples():
            index = row.Index
            neighbors = thermal_eq.explicit_nearest_neighboor(system_data=self.space,
                                                              x_coord=self.space['x_coords'][index],
                                                              y_coord=self.space['y_coords'][index],
                                                              z_coord=self.space['z_coords'][index],
                                                              space_resolution=self.space_resolution,
                                                              minx=min_xcoords, maxx=max_xcoords,
                                                              miny=min_ycoords, maxy=max_ycoords,
                                                              minz=min_zcoords, maxz=max_zcoords,
                                                              animate_neighbors=animate_neighbors,
                                                              visualize_neighbors=visualize_neighbors)
            self.space['nearest_neighbors'][index] = str(neighbors)
            console.pm_flush(message="Found neighbors for {}/{} coordinate points.".format(index + 1, loop_total))
            if animate_neighbors is True:
                self.movie_frames3.append('snap_{}-{}-{}.png'.format(self.space['x_coords'][index],
                                                                     self.space['y_coords'][index],
                                                                     self.space['z_coords'][index]))
        print("")
        if animate_neighbors is True:
            # self.space.to_csv("space2_coords_check.csv")
            import moviepy.editor as mpy
            import os
            os.chdir(os.getcwd() + "/nearest_neighbors")
            animation = mpy.ImageSequenceClip(self.movie_frames3,
                                              fps=5,
                                              load_images=True)
            animation.write_gif('neighbors.gif', fps=5)
            os.chdir("..")
        return None

    def generate_coordinate_points(self, length, width, height, space_resolution):
        """
        Generates all possible coordinate points within the defined box
        :param length: length of the system, in m
        :param width: width of the system, in m
        :param height: height of the system, in m
        :param space_resolution: spatial resolution of the system, in m
        :return: coords, a list of all coordinate points available in the system
        """
        console.pm_stat("Generating coordinates...")
        # print("Generating coordinates...")
        coords = []
        x_coords_range = np.arange(0, round((length + space_resolution), len(str(space_resolution))),
                                   space_resolution)  # generate range of x-coords
        y_coords_range = np.arange(0, round((width + space_resolution), len(str(space_resolution))),
                                   space_resolution)  # generate range of y-coords
        z_coords_range = np.arange(0, round((height + space_resolution), len(str(space_resolution))),
                                   space_resolution)  # generate range of z-coords
        for i in x_coords_range:
            for j in y_coords_range:
                for q in z_coords_range:
                    temp_coords = []
                    temp_coords.append(round(i, len(str(space_resolution))))
                    temp_coords.append(round(j, len(str(space_resolution))))
                    temp_coords.append(round(q, len(str(space_resolution))))
                    coords.append(temp_coords)
        console.pm_stat("Coordinates generated!")
        return coords

    def round_coord_arbitrary(self, coordinate, system_data, coordinate_type):
        """
        Rounds a calculated coordinate to the nearest one defined by the spatial resolution
        :param coordinate:
        :param system_data:
        :param coordinate_type:
        :return: rounded_coordinate
        """
        rounded_coordinate = ''
        found_min = ''
        for i in system_data[coordinate_type]:
            attempted_min = abs(coordinate - i)
            if found_min == '':
                found_min = attempted_min
                rounded_coordinate = i
            else:
                if attempted_min < found_min:
                    found_min = attempted_min
                    rounded_coordinate = i
        return rounded_coordinate

    def check_coords(self, x_coord, y_coord, z_coord):
        console.pm_stat("Checking if coordinates are valid for object insertion...")
        x_min, x_max = self.space['x_coords'][0], self.space['x_coords'][len(self.coords) - 1]
        y_min, y_max = self.space['y_coords'][0], self.space['y_coords'][len(self.coords) - 1]
        z_min, z_max = self.space['z_coords'][0], self.space['z_coords'][len(self.coords) - 1]
        if x_coord >= x_min and x_coord <= x_max:
            if y_coord >= y_min and y_coord <= y_max:
                if z_coord >= z_min and z_coord <= z_max:
                    console.pm_stat("Coordinates validated for object insertion!")
                    return True
        else:
            console.pm_err("Coordinates invalid!")
            return False

    def generate_object_id(self, matrix):
        """
        Generates object ID codes so that specific objects and materials can be tracked
        Object/matrial types are unique and coded by the a letter specifying the general type followed by a
        unique number combination
        The general object/material types are coded by the first letter as follows:
            - 'A' = object
            - 'B' = matrix
            - 'C' = boundary
        :param matrix:
        :return: object_id
        """

        def random_gen(object_identifier):
            object_id = object_identifier + str(randint(0, len(self.coords) + len(self.coords)))
            return object_id

        if matrix is True:
            object_id = random_gen(object_identifier='B')  # matrix material objects begin with a B
            while object_id in self.space['object_id'].tolist():
                object_id = random_gen(object_identifier='B')  # matrix material objects begin with a B
            return object_id
        else:
            object_id = random_gen(object_identifier='A')  # non-matrix material objects begin with a A
            while object_id in self.space['object_id'].tolist():
                object_id = random_gen(object_identifier='A')  # matrix material objects begin with a A
            return object_id

    # def insert_at_coord(self, x_coord, y_coord, z_coord):
    #     """
    #     :param x_coord:
    #     :param y_coord:
    #     :param z_coord:
    #     :return: row, the row index value in the self.space dateframe of the coordinate in question
    #     """
    #     space_copy = self.space.copy(deep=True)
    #     space_copy.set_index(['x_coords', 'y_coords', 'z_coords'], inplace=True)
    #     row = space_copy.loc(x_coord, y_coord, z_coord)
    #
    #     return row

    def insert_object(self, object, x_coord, y_coord, z_coord, object_radius, composition, initial_temperature):
        """
        Allows the insertion of an object into the box
        The object should NOT be inserted into coordinates occupied by boundaries
        This function should be called AFTER matrix insertion--else it will be overwritten
        :param object:
        :param x_coord:
        :param y_coord:
        :param z_coord:
        :param object_radius:
        :param composition:
        :param initial_temperature:
        :return: None
        """
        console.pm_stat("Inserting object...")
        if object in self.physical_parameters.index:
            if self.check_coords(x_coord=x_coord, y_coord=y_coord,
                                 z_coord=z_coord) is True:  # checks to verify that coordinates exist in space
                for row in self.space.itertuples():
                    index = row.Index
                    if self.space['x_coords'][index] == x_coord:
                        if self.space['y_coords'][index] == y_coord:
                            if self.space['z_coords'][index] == z_coord:  # verifies that coordinates match to Dataframe
                                self.space['object'][
                                    index] = object  # the name of the object, as defined in dynamics/physical_parameters.csv
                                self.space['object_id'][index] = self.generate_object_id(
                                    matrix=False)  # generates object ID
                                self.space['object_radius'][index] = object_radius  # in m
                                self.space['volume'][index] = (4 / 3) * pi * (
                                    object_radius) ** 3  # assume volume of object is a perfect sphere
                                self.space['mass'][index] = self.physical_parameters['Density'][object] * \
                                                            self.space['volume'][index]  # mass = density * volume
                                self.space['temperature'][index] = initial_temperature
                                self.space['object_density'] = float(self.space['mass'][index]) / ((4 / 3) * pi *
                                                                                                   float(self.space[
                                                                                                             'object_radius'][
                                                                                                             index]) ** 3)  # assume object is a perfect sphere
                                self.solution.create_solution(box=self.space, composition=composition, row=index,
                                                              object=object)
                                console.pm_flush("Inserted object ({}) at coordinates: x:{} y:{}, z:{}".format(
                                    self.space['object'][index],
                                    self.space['x_coords'][index],
                                    self.space['y_coords'][index],
                                    self.space['z_coords'][index]))
                                break

            else:
                console.pm_err("Could not insert object!  Outside of defined coordinate points!")
                sys.exit(1)
        else:
            console.pm_err("Object not defined in {}!  Cannot insert object!".format(
                os.getcwd() + "/dynamics/physical_parameters.csv"))
            sys.exit(1)
        return None


    def insert_matrix(self, matrix_material, composition, initial_temperature, temperature_gradient=None,
                      initial_pressure=None, pressure_gradient=None, initial_fO2=None, fO2_gradient=None, z_range=[0, 0]):
        """
        This function allows for the insertion of a matrix material over a given z-range
        This function should be called FIRST when constructing the box
        :param matrix_material:
        :param composition:
        :param initial_temperature:
        :param z_range: The depths at which the matrix should be inserted into the box
        :return: None
        """
        if (pressure_gradient is not None and initial_pressure is None) or (fO2_gradient is not None and initial_fO2 is None):
            console.pm_err("A gradient was applied without an initial condition!  "
                           "Please make sure the optional initial parameter is filled before when applying a gradient.")
            sys.exit(1)
        fixed_grad_T = initial_temperature  # stores the next temperature to be added in case the gradient option is added
        fixed_grad_P = initial_pressure  # stores the next pressure to be added in case the gradient option is added
        fixed_grad_fO2 = initial_fO2  # stores the next fO2 to be added in case the gradient option is added
        z_coords_range = []
        if z_range[1] != 0:
            z_coords_range = list(np.arange(z_range[0], round((z_range[1] + self.space_resolution), len(str(self.space_resolution))),
                                   self.space_resolution))  # generate range of z-coords
        else:
            z_coords_range = list(
                np.arange(0, round((self.height + self.space_resolution), len(str(self.space_resolution))),
                          self.space_resolution))  # generate range of z-coords
        z_coords_range_rounded = []
        for i in z_coords_range:
            rounded_coord = round(i, len(str(self.space_resolution)))
            z_coords_range_rounded.append(rounded_coord)
        z_coords_range = z_coords_range_rounded
        t_range = {}
        p_range = {}
        fO2_range = {}
        if temperature_gradient is not None:
            for i in z_coords_range:
                t_range.update({i: fixed_grad_T})
                fixed_grad_T += temperature_gradient
        if pressure_gradient is not None:
            for i in z_coords_range:
                p_range.update({i: fixed_grad_P})
                fixed_grad_P += pressure_gradient
        if fO2_gradient is not None:
            for i in z_coords_range:
                fO2_range.update({i: fixed_grad_fO2})
                fixed_grad_fO2 += fO2_gradient
        console.pm_stat("Inserting matrix...")
        if matrix_material in self.physical_parameters.index:
            if z_range[1] == 0:
                # z range is a list of two numbers, the minimum depth at the index 0, and the maximum depth at index 1
                for row in self.space.itertuples():
                    index = row.Index
                    self.space['object_id'][index] = self.generate_object_id(matrix=True)  # generates the object id
                    self.space['object'][index] = matrix_material
                    if initial_pressure is not None:  # makes sure that a pressure is set
                        if pressure_gradient is not None:  # checks if the pressure gradient option is selected
                            self.space['pressure'][index] = p_range[round(self.space['z_coords'][
                                                                              index], len(str(
                                self.space_resolution)))]  # if a gradient is added, applies the pressure gradient
                        else:  # if no gradient selected, applies a homogeneous pressure
                            self.space['pressure'][index] = initial_pressure
                    if temperature_gradient is None:  # no gradient is selected, homogeneous matrix temperature
                        self.space['temperature'][index] = initial_temperature
                    else:  # if a gradient is added, applies the temperature gradient
                        self.space['temperature'][index] = t_range[round(self.space['z_coords'][
                                                                             index], len(str(self.space_resolution)))]
                    if initial_fO2 is not None:  # applies an fO2 if one is selected
                        if fO2_gradient is None:  # no gradient is selected, homogeneous matrix fO2
                            self.space['fO2_({})'.format(self.fO2_buffer)][index] = initial_fO2
                        else:  # if a gradient is added, applies the fO2 gradient
                            self.space['fO2_({})'.format(self.fO2_buffer)][index] = fO2_range[round(self.space['z_coords'][
                                                                                 index], len(str(self.space_resolution)))]
                    self.solution.create_solution(box=self.space, composition=composition, row=index,
                                                  object=matrix_material)
                    console.pm_flush(
                        "Inserted matrix ({}) at coordinates: x:{} y:{}, z:{}".format(self.space['object'][index],
                                                                                      self.space['x_coords'][index],
                                                                                      self.space['y_coords'][index],
                                                                                      self.space['z_coords'][
                                                                                          index]))
            else:
                for row in self.space.itertuples():
                    index = row.Index
                    if round(z_range[0], len(str(self.space_resolution))) <= self.space['z_coords'][index] <= round(
                            z_range[1], len(str(self.space_resolution))):  # inserts only between z gradients
                        self.space['object_id'][index] = self.generate_object_id(matrix=True)  # generates the object id
                        self.space['object'][index] = matrix_material
                        if initial_pressure is not None:  # makes sure that a pressure is set
                            if pressure_gradient is not None:  # checks if the pressure gradient option is selected
                                self.space['pressure'][index] = p_range[round(self.space['z_coords'][
                                                                                  index], len(str(
                                    self.space_resolution)))]  # if a gradient is added, applies the pressure gradient
                            else:  # if no gradient selected, applies a homogeneous pressure
                                self.space['pressure'][index] = initial_pressure
                        if temperature_gradient is None:  # no gradient is selected, homogeneous matrix temperature
                            self.space['temperature'][index] = initial_temperature
                        else:  # if a gradient is added, applies the temperature gradient
                            self.space['temperature'][index] = t_range[round(self.space['z_coords'][
                                                                                 index],
                                                                             len(str(self.space_resolution)))]
                        if initial_fO2 is not None:  # applies an fO2 if one is selected
                            if fO2_gradient is None:  # no gradient is selected, homogeneous matrix fO2
                                self.space['fO2_({})'.format(self.fO2_buffer)][index] = initial_fO2
                            else:  # if a gradient is added, applies the fO2 gradient
                                self.space['fO2_({})'.format(self.fO2_buffer)][index] = fO2_range[
                                    round(self.space['z_coords'][
                                              index], len(str(self.space_resolution)))]
                        self.solution.create_solution(box=self.space, composition=composition, row=index,
                                                      object=matrix_material)
                        console.pm_flush(
                            "Inserted matrix ({}) at coordinates: x:{} y:{}, z:{}".format(self.space['object'][index],
                                                                                          self.space['x_coords'][index],
                                                                                          self.space['y_coords'][index],
                                                                                          self.space['z_coords'][
                                                                                              index]))
            print("")
            console.pm_stat("Matrix material(s) ({}) inserted!".format(matrix_material))

        else:
            console.pm_err("Matrix material not defined in {}!  Cannot insert matrix material!".format(
                os.getcwd() + "/dynamics/physical_parameters.csv"))
            sys.exit(1)
        return None

    def insert_boundary(self, temperature, z_range, boundary_location='bottom', flux=True, pressure=None, fO2=None):
        """
        Insert a boundary layer for the purpose of regulating z-gradients in heat exchange.
        It is recommended that a boundary layer is inserted
        :param temperature:
        :param z_range:
        :param boundary_location: Either the boundary layer is on the 'top' or the 'bottom' of the model.
                The boundary location defaults to bottom if not explicitly stated.
        :param flux: allow heat flux from the boundary layers to permeate the rest of the model
        :return:
        """
        if z_range[1] != 0:
            self.boundary_vals.append(z_range[0])
            self.boundary_vals.append(z_range[1])
            if boundary_location == 'bottom':
                self.model_base = z_range[
                    0]  # base of model considered to be the top (highest z-coordinate) of boundary layer
            elif boundary_location == 'top':
                self.model_base = z_range[
                    1]  # top of model considered to be the bottom (lowest z-coordinate) of boundary layer
            for row in self.space.itertuples():
                index = row.Index
                if round(z_range[0], len(str(self.space_resolution))) <= self.space['z_coords'][index] <= round(
                        z_range[1], len(str(self.space_resolution))):
                    self.space['object_id'][index] = 'C'
                    self.space['object'][index] = "Boundary"
                    self.space['temperature'][index] = temperature
                    self.space['pressure'][index] = pressure
                    self.space['fO2'][index] = fO2
                    console.pm_flush("Inserted boundary at coordinates: x:{} y:{}, z:{}".format(
                        self.space['x_coords'][index],
                        self.space['y_coords'][index],
                        self.space['z_coords'][
                            index]))
            print("")
            console.pm_stat("Boundary layer inserted between z-range: {}m-{}m!".format(z_range[0], z_range[1]))

    def visualize_box(self):
        """
        Constructs animation frames that allows for the visualization of the box
        :return: None
        """
        # creates the 3D diapir movement animation frames
        if self.visualize_system != False:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlim(xmin=min(self.space['x_coords']), xmax=max(self.space['x_coords']))
            ax.set_ylim(ymin=min(self.space['y_coords']), ymax=max(self.space['y_coords']))
            ax.set_zlim(zmin=min(self.space['z_coords']), zmax=max(self.space['z_coords']))
            for row in self.space.itertuples():
                index = row.Index
                x = self.space['x_coords'][index]
                y = self.space['y_coords'][index]
                z = self.space['z_coords'][index]
                try:
                    if str(self.space['object_id'][index])[0] == 'A':
                        ax.scatter3D(x, y, z, color='b', s=self.space['object_radius'][index] * 100)
                except:
                    self.space.to_csv("alskdfjakhsdf.csv")
                    sys.exit(1)
            ax.set_title("System 3D Heatmap at Time {}".format(self.model_time))
            ax.set_xlabel("Box Length (x) (m)")
            ax.set_ylabel("Box Width (y) (m)")
            ax.set_zlabel("Box Height (z) (m)")
            ax.invert_zaxis()
            fig.savefig(os.getcwd() + '/object_dynamics/snap_{}.png'.format(self.model_time), format='png')
            fig.clf()
            self.movie_frames1.append('snap_{}.png'.format(self.model_time))
            console.pm_stat("System snapshot created: {}".format('snap_{}.png'.format(self.model_time)))

            # creates 3D heatmap animation frames
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlim(xmin=min(self.space['x_coords']), xmax=max(self.space['x_coords']))
            ax.set_ylim(ymin=min(self.space['y_coords']), ymax=max(self.space['y_coords']))
            ax.set_zlim(zmin=min(self.space['z_coords']), zmax=max(self.space['z_coords']))
            for row in self.space.itertuples():
                index = row.Index
                x = self.space['x_coords'][index]
                y = self.space['y_coords'][index]
                z = self.space['z_coords'][index]
                # velocity_x = self.space['x_direct'][row]
                try:
                    if str(self.space['object_id'][index][0]) == 'A':
                        # print("Plotted object at: x:{} y:{} z:{}.".format(x, y, z))
                        ax.scatter3D(x, y, z, color='b', s=self.space['object_radius'][index] * 100)
                except:
                    self.space.to_csv("alskdfjakhsdf.csv")
                    sys.exit(1)
            # norm_colors = mpl.colors.Normalize(vmin=self.space['temperature'].min(), vmax=self.space['temperature'].max())
            norm_colors = mpl.colors.Normalize(vmin=1900, vmax=2200)
            colorsmap = matplotlib.cm.ScalarMappable(norm=norm_colors, cmap='jet')
            colorsmap.set_array(self.space['temperature'])
            ax.scatter(self.space['x_coords'], self.space['y_coords'], self.space['z_coords'], marker='s', s=140,
                       c=self.space['temperature'], cmap='jet', alpha=0.50)
            cb = fig.colorbar(colorsmap)
            ax.set_title("System 3D Heatmap at Time {}".format(self.model_time))
            ax.set_xlabel("Box Length (x) (m)")
            ax.set_ylabel("Box Width (y) (m)")
            ax.set_zlabel("Box Height (z) (m)")
            ax.invert_zaxis()
            fig.savefig(os.getcwd() + '/thermal_equilibrium_heatmap/snap_{}.png'.format(self.model_time), format='png')
            self.movie_frames2.append('snap_{}.png'.format(self.model_time))
            fig.clf()

            # creates the 3D model base trisurf animation frames
            x_coords = []
            y_coords = []
            temperature = []
            for row in self.space.itertuples():
                index = row.Index
                surface_zcoord = round((self.model_base - self.space_resolution), len(str(self.space_resolution)))
                if float(self.space['z_coords'][index]) == surface_zcoord:
                    x_coords.append(self.space['x_coords'][index])
                    y_coords.append(self.space['y_coords'][index])
                    temperature.append(self.space['temperature'][index])
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_trisurf(x_coords, y_coords, temperature)
            ax.set_xlabel("Box Length (x) (m)")
            ax.set_ylabel("Box Width (y) (m)")
            ax.set_zlabel("Temperature (degK)")
            ax.set_zlim(zmin=1990, zmax=2500)
            ax.set_title("Temperature Distribution at Time {} At Base of Model".format(self.model_time))
            fig.savefig(os.getcwd() + '/temp_distrib_floor/snap_{}.png'.format(self.model_time), format='png')
            self.movie_frames4.append('snap_{}.png'.format(self.model_time))
            fig.clf()
        return None

    @staticmethod
    def grab_row_index_by_coord(system_data, x_coord, y_coord, z_coord):
        """
        Returns the index of the row in the instance's Pandas dataframe by associating with x, y, and z coordinates stored
            in the dataframe.
        :param system_data:
        :param x_coord:
        :param y_coord:
        :param z_coord:
        :return: row, the index
        """
        for row in system_data.itertuples():
            index = row.Index
            if system_data['x_coords'][index] == x_coord:
                if system_data['y_coords'][index] == y_coord:
                    if system_data['z_coords'][index] == z_coord:
                        return index

    @staticmethod
    def swap_rows(system_data, update_space, from_row_index, to_row_index):
        stationary_columns = ['x_coords', 'y_coords', 'z_coords', 'coord_index', 'nearest_neighbors']
        for i in system_data:
            if i not in stationary_columns:
                cp_from = system_data[i][from_row_index]
                cp_to = system_data[i][to_row_index]
                system_data[i][to_row_index] = cp_from
                system_data[i][from_row_index] = cp_to

        return system_data


    def replace_fromobject(self, system_data, update_space, from_object_index, to_object_index, stationary_columns):
        found = False
        from_neighbors = ast.literal_eval(system_data['nearest_neighbors'][from_object_index])
        avg_temps_list = []
        avg_object_list = []
        for i in from_neighbors:
            for z in from_neighbors[i]:
                if "+" in z or "-" in z:
                    if len(from_neighbors[i][z]['index']) != 0:
                        temp = system_data['temperature'][from_neighbors[i][z]['index']].values.tolist()[0]
                        avg_obj = system_data['object'][from_neighbors[i][z]['index']].values.tolist()[0]
                        if avg_obj != 'Boundary':
                            avg_temps_list.append(temp)
                            avg_object_list.append(avg_obj)
        avg_temp = sum(avg_temps_list) / len(avg_temps_list)
        common_obj = Counter(avg_object_list).most_common()[0][0]
        for i in from_neighbors:
            for z in from_neighbors[i]:
                obj = system_data['object'][from_neighbors[i][z]['index']].values.tolist()[0]
                if obj == common_obj:
                    for q in system_data:
                        if q not in stationary_columns:
                            system_data[q][from_object_index] = system_data[q][from_neighbors[i][z]['index']].values.tolist()[0]
                    system_data['temperature'][from_object_index] = avg_temp
                    found = True
                    break
            if found is True:
                break

    def define_path(self, start, end, length, width, height):
        """
        Get the list of coordinates between two points in the box.
        :param start: a list of path start coordinates, [length, width, height] (i.e. [x, y, z])
        :param end: a list of path end coordinates, [length, width, height] (i.e. [x, y, z])
        :param length: x-coordinates
        :param width: y-coordinates
        :param height: z-coordinates
        :return: path, the list of lists of coordinates in the path
        """
        start_x = start[0]
        start_y = start[1]
        start_z = start[2]
        end_x = end[0]
        end_y = end[1]
        end_z = end[2]
        path_x = end_x - start_x
        path_y = end_y - start_y
        path_z = end_z - start_z



    def merge_objects(self, to_object_index, from_object_index, system_data, update_space):
        """
        When two objects of the same type occupy the same point in coordinate space, they will merge.
        The deepest object (i.e. the "to" destination of the "from" diapir) will inherit all properties.
        The object doing the sinking will merge to the deeper object + disappear.
        :param to_object_index:
        :param from_object_index:
        :param system_data:
        :return:
        """
        stationary_columns = ['x_coords', 'y_coords', 'z_coords', 'coord_index',
                              'nearest_neighbors']  # columns that are not swapped
        additive_columns = ['mass', 'volume']  # columns that contain additive data when diapirs merge
        console.pm_stat("Objects {} and {} will merge to object {}!".format(system_data['object_id'][from_object_index],
                                                                            system_data['object_id'][
                                                                                to_object_index], system_data['object_id'][
                                                                                to_object_index]))
        for i in system_data:
            # makes sure that the column is an additive property
            if (i not in stationary_columns) and (i in additive_columns):
                # adds the values
                system_data[i][to_object_index] = system_data[i][to_object_index] + system_data[i][
                    from_object_index]
        # takes an average of temperatures of the merging objects for the temperature of the merged object
        # this should eventually be weighted by object radius
        system_data['object_radius'][to_object_index] = ((system_data['volume'][to_object_index] * 3) / (4 * pi))**(1/3) # V = (4/3)*pi*r^3 --> r = ((3V)/(4pi))^(1/3)
        system_data['temperature'][to_object_index] = (system_data['temperature'][from_object_index] +
                                                                        system_data['temperature'][to_object_index]) / 2
        self.replace_fromobject(system_data=system_data, update_space=system_data, from_object_index=from_object_index,
                                stationary_columns=stationary_columns, to_object_index=to_object_index)
        return system_data

    # TODO: seperate velocity calculations from system movement so space dataframe can be updated and moved according to velocity contents
    @classmethod
    def calculate_velocities(cls):
        pass

    # # TODO: calculate the distance between the two points, and then find nearest coordinate neighbors along the path to account for x,y,z
    # # TODO: right now, just operates in z-direction. will have to rewrite entire method if lateral motion is to occur
    # @classmethod
    # def gather_path_coords(cls, system_data, from_zcoord, to_zcoord, x_coord, y_coord, from_xcoord=None, to_xcoord=None,
    #                        from_ycoord=None, to_ycoord=None):
    #     path_coords = []
    #     for row in system_data.index:
    #         unique_path_coords = [] # x coord at index 0, y coord at 1, z coord at 2
    #         if float(system_data['x_coord'][row]) == float(x_coord) and float(system_data['y_coord']) == float(y_coord):
    #             if float(system_data['z_coord'][row]) >= float(from_zcoord) and float(system_data['z_coord'][row]) <= to_zcoord:
    #                 # assumption is that z axis is inverted, but z values increase with decreasing depth
    #                 unique_path_coords.append(float(x_coord))
    #                 unique_path_coords.append(float(y_coord))
    #                 unique_path_coords.append(float(system_data['z_coord'][row]))
    #         path_coords.append(unique_path_coords)
    #     return path_coords

    def move_systems(self, system_data, update_space, deltaTime, box_height, space_resolution,
                     default_matrix_material='Silicate Liquid'):
        """
        Allows for the movement of objects to occur within the box if objects are gravitationally unstable
        :param system_data:
        :param update_space:
        :param deltaTime:
        :param box_height:
        :param space_resolution:
        :param default_matrix_material:
        :return: update_space_copy, a copy of the self.space dataframe with updated object/matrix positions
        """
        # update_space_copy = update_space.copy(deep=True)
        inactive_objects = []
        update_space_copy = self.space
        for row in system_data.itertuples():
            index = row.Index
            # object_id's that begin with 'A' are objects and will be free to move
            if str(system_data['object_id'][index][0]) == 'A' and str(system_data['object_id'][index]) not in inactive_objects:
                inactive_objects.append(str(system_data['object_id'][index]))
                curr_x_coords = system_data['x_coords'][index]
                curr_y_coords = system_data['y_coords'][index]
                curr_z_coords = system_data['z_coords'][index]
                object_velocity = 0

                matrix_material = default_matrix_material  # the default matrix matrial until overwritten
                matrix_material_temp = 0.0
                matrix_material_pressure = 0.0
                # assumption is that object will travel through matrix most like that occupying z coord below it.
                # code block below attempts to idenfity that material
                if (system_data['z_coords'][index] + space_resolution) in system_data['z_coords']:
                    searchfor_coord = (system_data['z_coords'][index] + space_resolution)
                    for row2 in system_data.itertuples():
                        index2 = row2.Index
                        if system_data['z_coords'][index2] == searchfor_coord and system_data['y_coords'][index2] \
                                == curr_y_coords and system_data['x_coords'][index2] == curr_x_coords:
                            matrix_material = system_data['object'][index2]
                            matrix_material_temp = system_data['temperature'][index2]
                            matrix_material_pressure = system_data['pressure'][index2]
                            break
                object_velocity = move_particle(body_type=system_data['object'][index],
                                                system_params=system_data).stokes_settling(
                    object=system_data['object'][index], matrix_material=matrix_material,
                    matrix_material_temp=matrix_material_temp, matrix_material_pressure=matrix_material_pressure,
                    object_radius=system_data['object_radius'][index])
                z_dis_obj_travel = object_velocity * deltaTime
                updated_x_coord = round(system_data['x_coords'][index], len(str(space_resolution)))
                updated_y_coord = round(system_data['y_coords'][index], len(str(space_resolution)))
                # round the z-coordinate to the nearest point within the spatial resolution
                updated_z_coord = round(self.round_coord_arbitrary(
                    coordinate=(z_dis_obj_travel + system_data['z_coords'][index]),
                    system_data=system_data, coordinate_type='z_coords'), len(str(space_resolution)))
                rounded_z_distance_travelled = round(updated_z_coord - curr_z_coords, len(str(
                    space_resolution)))  # use this distance for distance travelled, as it is more self-consistent within the model
                # check to see if object travels into boundary layer.  if so, put it in nearest point within spatial resolution ABOVE boundary layer
                if round(rounded_z_distance_travelled + curr_z_coords, len(str(space_resolution))) >= self.model_base:
                    updated_z_coord = round(self.model_base - self.space_resolution,
                                            len(str(space_resolution)))  # fix the z-coord
                    rounded_z_distance_travelled = round(updated_z_coord - curr_z_coords,
                                                         len(str(space_resolution)))  # fix the distance travelled
                rounded_object_velocity = rounded_z_distance_travelled / deltaTime  # makes object velocity self-consistent with model
                system_data['rounded_object_velocity'][index] = rounded_object_velocity  # makes object velocity self-consistent with model
                # checks to make sure that the space/time resolution was big enough for the object to move.  if not, velocity/distance_travelled = 0
                if rounded_z_distance_travelled == 0:
                    object_velocity = 0
                    rounded_object_velocity = 0
                    system_data['rounded_object_velocity'][index] = rounded_object_velocity
                    z_dis_obj_travel = 0

                # get the index of the coordinate point where the object will travel to
                to_row_index = self.grab_row_index_by_coord(system_data=system_data,
                                                            x_coord=updated_x_coord,
                                                            y_coord=updated_y_coord,
                                                            z_coord=updated_z_coord)
                from_row_index = self.grab_row_index_by_coord(system_data=system_data,
                                                              x_coord=system_data['x_coords'][index],
                                                              y_coord=system_data['y_coords'][index],
                                                              z_coord=system_data['z_coords'][index])
                # update the copy of the dataframe with the appropriate changes
                if rounded_object_velocity != 0:
                    console.pm_flush(
                        "Object {} will move! ({},{},{} to {},{},{} - {}m)".format(system_data['object_id'][index],
                                                                             curr_x_coords, curr_y_coords,
                                                                             curr_z_coords, updated_x_coord,
                                                                             updated_y_coord, updated_z_coord,
                                                                                   rounded_z_distance_travelled))
                # stokes_data returns degK, F_g, F_b, F_d
                stokes_data = energy().stokes_frictional_energy(
                    object=system_data['object'][index], matrix_material=matrix_material,
                    body_radius=system_data['object_radius'][index],
                    body_mass=system_data['mass'][index], distance_travelled=rounded_z_distance_travelled,
                    object_velocity=system_data['rounded_object_velocity'][index])
                system_data['heat_generated'][index] = float(stokes_data[0])  # grabs degK from stokes_data and stores it as the heat generated due to viscous dissipation
                system_data['temperature'][index] = float(
                    system_data['temperature'][index]) + stokes_data[
                                                        0]  # grabs degK from stokes_data & adjusts the temperature
                system_data['drag_force'][index] = float(stokes_data[1])  # gets drag force and adds it to the dataframe
                system_data['buoyant_force'][index] = float(
                    stokes_data[2])  # gets buoyant force and adds it to the dataframe
                system_data['gravitational_force'][index] = float(
                    stokes_data[3])  # gets gravitational force and adds it to the dataframe
                system_data['object_velocity'][index] = object_velocity
                # system_data['z_direct'][index] = object_velocity
                system_data['potential_energy'][index] = energy().potential_energy(mass=system_data['mass'][index],
                                                                                   height=system_data['z_coords'][
                                                                                       index],
                                                                                   box_height=box_height)
                system_data['kinetic_energy'][index] = energy().kinetic_energy(mass=system_data['mass'][index],
                                                                               velocity=system_data['rounded_object_velocity'][
                                                                                   index])
                if object_velocity != 0:
                    console.pm_stat("{} ({}) will move from x:{} y:{} z:{} to x:{} y:{} z:{} (real velocity: {}, rounded velocity: {})".format(
                        system_data['object_id'][index], system_data['object'][index], system_data['x_coords'][index],
                        system_data['y_coords'][index], system_data['z_coords'][index], updated_x_coord, updated_y_coord,
                        updated_z_coord, system_data['object_velocity'][index], system_data['rounded_object_velocity'][index]))
                # check to see if two objects of the same type will collide
                # if two objects of the same type collide, they will merge
                # else, just swap points with the matrix material at the destination coordinate point
                if (system_data['object'][from_row_index] == system_data['object'][to_row_index]) and \
                        (system_data['object_id'][from_row_index] != system_data['object_id'][to_row_index]):
                    update_space_copy = self.merge_objects(to_object_index=to_row_index, from_object_index=from_row_index, system_data=system_data, update_space=update_space)
                else:
                    if object_velocity != 0:
                        update_space_copy = self.swap_rows(system_data=system_data, update_space=update_space,
                                                           from_row_index=from_row_index, to_row_index=to_row_index)
        print("")
        return update_space_copy


    def certify_box(self):
        for row in self.space.itertuples():
            index = row.Index
            try:
                if 'A' in self.space['object_id'][index]:
                    pass
            except:
                console.pm_err(
                    "Box integrity check failed.  Please check your z-ranges to make sure all "
                    "coordinate spaces are filled..")
                sys.exit(1)
        res = [self.width, self.length, self.height]
        for i in res:
            if (i % self.space_resolution) - self.space_resolution >= 0:
                console.pm_err("Box integrity check failed.  Your space resolution is not a multiple of "
                               "the box length, width, and/or height.")
                sys.exit(1)
        for i in self.boundary_vals:
            if (i % self.space_resolution) - self.space_resolution >= 0:
                console.pm_err("Box integrity check failed.  Your space resolution is not a multiple of "
                               "the boundary layer limit(s).")
                sys.exit(1)
        console.pm_stat("Box integrity confirmed.  Calculations allowed to proceed.")




    # TODO: update x and y coords
    def update_system(self, auto_update=True, deltaTime=1.0):
        """
        Updates the system thermal/dynamic/chemical state at each time interval
        :param auto_update:
        :param deltaTime:
        :return: self.model_time, self.space
        """
        console.pm_stat("Model time at: {}".format(self.model_time))
        # update_space = self.space.copy(deep=True)
        # this section only executes at the initial time--no object or thermal movement occurs here
        if self.model_time == self.initial_time:
            # check the integrity of the box before time and neighbor identifification allowed to progress
            self.certify_box()
            # if box integrity confirmed, proceed to nearest neighbor identification
            self.classify_neighbors(visualize_neighbors=self.visualize_neighbors,
                                    animate_neighbors=self.animate_neighbors)
            # create an initial snapshot of the box
            self.visualize_box()
            # writes an object history output file if flagged in box setup
            if self.object_history is True:
                for row in self.space.itertuples():
                    index = row.Index
                    if 'A' in self.space['object_id'][index]:
                        contents = []
                        contents.append(str(self.model_time))
                        for i in self.space:
                            contents.append(str(self.space[i][index]))
                        formatted_contents = ",".join(i.replace(",", ":") for i in contents)
                        self.object_output.write("{}\n".format(formatted_contents))
        # executes when the model time is exhausted--writes output files and animations and then ends the simulation
        elif self.model_time <= 0:
            self.visualize_box()
            console.pm_stat("Model at minimum time!")
            if self.visualize_system is True:
                console.pm_stat("Writing animations...")

                # dynamics animation
                os.chdir(os.getcwd() + '/object_dynamics')
                animation = mpy.ImageSequenceClip(self.movie_frames1,
                                                  fps=round((self.initial_time / (self.initial_time / 3))),
                                                  load_images=True)
                os.chdir('..')
                animation.write_videofile('object_dynamics.mp4',
                                          fps=round((self.initial_time / (self.initial_time / 3))), audio=False)
                animation.write_gif('object_dynamics.gif',
                                    fps=round((self.initial_time / (self.initial_time / 3))))
                console.pm_stat("Animation created & available in {}!".format(os.getcwd()))

                # 3d heatmap animation
                os.chdir(os.getcwd() + '/thermal_equilibrium_heatmap')
                animation = mpy.ImageSequenceClip(self.movie_frames2,
                                                  fps=round((self.initial_time / (self.initial_time / 3))),
                                                  load_images=True)
                os.chdir('..')
                animation.write_videofile('thermal_equilibrium_heatmap.mp4',
                                          fps=round((self.initial_time / (self.initial_time / 3))), audio=False)
                animation.write_gif('thermal_equilibrium_heatmap.gif',
                                    fps=round((self.initial_time / (self.initial_time / 3))))
                console.pm_stat("Animation created & available in {}!".format(os.getcwd()))

                # 3d model base heat distribution animation
                os.chdir(os.getcwd() + '/temp_distrib_floor')
                animation = mpy.ImageSequenceClip(self.movie_frames4,
                                                  fps=round((self.initial_time / (self.initial_time / 3))),
                                                  load_images=True)
                os.chdir('..')
                animation.write_videofile('temp_distrib_floor.mp4',
                                          fps=round((self.initial_time / (self.initial_time / 3))), audio=False)
                animation.write_gif('temp_distrib_floor.gif',
                                    fps=round((self.initial_time / (self.initial_time / 3))))
                console.pm_stat("Animation created & available in {}!".format(os.getcwd()))

                # writes the central pandas dataframe to 'space.csv'.  most critical model info contained here
                self.space.to_csv("space.csv")
                # writes the chemical compositions to 'solution.csv'
                self.solution.get_solution().to_csv("solution.csv")
                # writes the object history output file
                if self.object_history is True:
                    for row in self.space.itertuples():
                        index = row.Index
                        if 'A' in self.space['object_id'][index]:
                            contents = []
                            contents.append(str(self.model_time))
                            for i in self.space:
                                contents.append(str(self.space[i][index]))
                            formatted_contents = ",".join(i.replace(",", ":") for i in contents)
                            self.object_output.write("{}\n".format(formatted_contents))
                if self.object_output is True:
                    self.object_output.close()
                return self.model_time, self.space
        else:
            # models the object movement
            self.move_systems(system_data=self.space, update_space=None, deltaTime=deltaTime,
                                             box_height=self.height, space_resolution=self.space_resolution)
            update_space = self.space.copy(deep=True)
            # updates chemical compositions
            update_solution = self.solution.update_solution(deltaTime=deltaTime)
            # models thermal equilibrium
            therm_eq_update_space = thermal_eq().D3_thermal_eq(system_data=update_space, deltaTime=deltaTime,
                                                               space_resolution=self.space_resolution)
            for row in update_space.itertuples():
                index = row.Index
                if 'A' in update_space['object_id'][index]:
                    self.velocity_output.write("\n{}".format(update_space['object_velocity'][index]))
            self.visualize_box()
            self.space = update_space
            if self.object_history is True:
                for row in self.space.itertuples():
                    index = row.Index
                    if 'A' in self.space['object_id'][index]:
                        contents = []
                        contents.append(str(self.model_time))
                        for i in self.space:
                            contents.append(str(self.space[i][index]))
                        formatted_contents = ",".join(i.replace(",", ":") for i in contents)
                        self.object_output.write("{}\n".format(formatted_contents))
        # auto-update calculates the appropriate deltaTime, if one is not defined
        if auto_update is True:
            if self.model_time == deltaTime:
                self.model_time -= deltaTime
                self.update_system(auto_update=False, deltaTime=deltaTime)
            elif self.model_time > deltaTime:
                self.model_time -= deltaTime
                self.update_system(auto_update=auto_update, deltaTime=deltaTime)
            else:
                return self.model_time, self.space
        else:
            return self.model_time, self.space


