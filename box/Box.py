import matplotlib as mpl
mpl.use('Qt5Agg')
import os
import numpy as np
import pandas as pd
from random import randint
import moviepy.editor as mpy
os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from dynamics.Movement import move_particle
os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from dynamics.Energy import energy, thermal_eq
os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from thermodynamics.Solution import solution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import sys
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.colorbar
import time
from math import pi

# TODO: update some methods to class methods to avoid outside interference
class box:

    def __init__(self, length, width, height, space_resolution, model_time, visualize_system=False, object_history=False):
        """
        :param length: length of the system, in m
        :param width: width of the system, in m
        :param height: height of the system, in m
        :param space_resolution: spatial resolution of the system, in m
        :param model_time: initial time of the system, in 'years ago'
        :param visualize_system: optional parameter to turn on movie generation of the evolution of the system

        """
        self.length = length
        self.width = width
        self.height = height
        self.space_resolution = space_resolution
        self.model_time = float(model_time)
        self.initial_time = float(model_time)
        self.coords = self.generate_coordinate_points(length=self.length, width=self.width, height=self.height,
                                                 space_resolution=self.space_resolution)
        self.visualize_system = visualize_system
        self.object_history = object_history
        self.space = pd.DataFrame({
            'object_id': np.NAN, # randomly generated object id tag to identify unique elements in box
            'object': np.NAN, # name of the object, as defined in self.physical_parameters
            'x_coords': [float(i[0]) for i in self.coords],
            'y_coords': [float(i[1]) for i in self.coords],
            'z_coords': [float(i[2]) for i in self.coords],
            'nearest_neighbors': np.NAN, # x, y, and z neighbors in each direction
            'object_radius': np.NAN, # in m
            'density': np.NAN, # in kg/m^3
            'temperature': np.NAN, # in K
            'pressure': [(1*10**9) for i in self.coords], # pressure in pascals, in order to work with ideal gas law
            'object_velocity': [float(0) for i in self.coords],
            'x_direct': np.NAN,
            'y_direct': np.NAN,
            'z_direct': np.NAN,
            'potential_energy': np.NAN,
            'kinetic_energy': np.NAN, # in J
            'total_energy_released': np.NAN, # in J
            'mass': np.NAN, # in kg
            'volume': np.NAN # in m^3
        })
        self.num_coords = len(self.coords)
        self.solution = solution(box_length=self.num_coords)
        self.physical_parameters = pd.read_csv(os.path.dirname(os.path.abspath('.')) + "/fEquilibrium/dynamics/physical_parameters.csv", index_col='Material')
        self.move_frames1 = []
        self.move_frames2 = []
        self.move_frames3 = []
        self.move_frames4 = []
        if os.path.exists('mpl_animation1'):
            shutil.rmtree('mpl_animation1')
        if os.path.exists('mpl_animation2'):
            shutil.rmtree('mpl_animation2')
        if os.path.exists('mpl_animation3'):
            shutil.rmtree('mpl_animation3')
        if os.path.exists('mpl_animation4'):
            shutil.rmtree('mpl_animation4')
        os.mkdir('mpl_animation1')
        os.mkdir('mpl_animation2')
        os.mkdir('mpl_animation3')
        os.mkdir('mpl_animation4')
        if self.object_history == True:
            if "object_history.csv" in os.listdir(os.getcwd()):
                os.remove("object_history.csv")
            self.object_output = open("object_history.csv", 'w')
            header = ['Model Time']
            for i in self.space.columns.tolist():
                header.append(str(i))
            formatted_header = ",".join(i for i in header)
            self.object_output.write("{}\n".format(formatted_header))
        if 'object_velocities.csv' in os.listdir(os.getcwd()):
            os.remove('object_velocities.csv')
        self.velocity_output = open('object_velocities.csv', 'a')


    def get_box(self):
        return self.space

    def classify_neighbors(self, animate_neighbors):
        loop_count = 1
        loop_total = len(self.space.index.tolist())
        print("Finding nearest neighbors for all points.  This may take several minutes...")
        for row in self.space.index:
            neighbors = thermal_eq.explicit_nearest_neighboor(system_data=self.space, x_coord=self.space['x_coords'][row],
                    y_coord=self.space['y_coords'][row], z_coord=self.space['z_coords'][row], space_resolution=self.space_resolution)
            self.space['nearest_neighbors'][row] = str(neighbors)
            print("Found neighbors for {}/{} coordinate points.".format(loop_count, loop_total))
            loop_count += 1
            if animate_neighbors == True:
                self.move_frames3.append('snap_{}-{}-{}.png'.format(self.space['x_coords'][row],
                                                        self.space['y_coords'][row], self.space['z_coords'][row]))
        if animate_neighbors == True:
            self.space.to_csv("space2_coords_check.csv")
            import moviepy.editor as mpy
            import os, time
            os.chdir(os.getcwd() + "/mpl_animation3")
            animation = mpy.ImageSequenceClip(self.move_frames3,
                                              fps=5,
                                              load_images=True)
            animation.write_gif('neighbors.gif', fps=5)
            os.chdir("..")
        return None



    @staticmethod
    def generate_coordinate_points(length, width, height, space_resolution):
        """
        :param length: length of the system, in m
        :param width: width of the system, in m
        :param height: height of the system, in m
        :param space_resolution: spatial resolution of the system, in m
        :return: coords, a list of all coordinate points available in the system
        """
        print("Generating coordinates...")
        coords = []
        x_coords_range = np.arange(0, length + 1, space_resolution)
        y_coords_range = np.arange(0, width + 1, space_resolution)
        z_coords_range = np.arange(0, height + 1, space_resolution)
        for i in x_coords_range:
            for j in y_coords_range:
                for q in z_coords_range:
                    temp_coords = []
                    temp_coords.append(round(i, len(str(space_resolution))))
                    temp_coords.append(round(j, len(str(space_resolution))))
                    temp_coords.append(round(q, len(str(space_resolution))))
                    coords.append(temp_coords)
        print("Coordinates generated!")
        return coords

    @staticmethod
    def round_coord_arbitrary(coordinate, system_data, coordinate_type):
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
        print("Checking if coordinates are valid for object insertion...")
        x_min, x_max = self.space['x_coords'][0], self.space['x_coords'][len(self.coords) - 1]
        y_min, y_max = self.space['y_coords'][0], self.space['y_coords'][len(self.coords) -1]
        z_min, z_max = self.space['z_coords'][0], self.space['z_coords'][len(self.coords) -1]
        if x_coord >= x_min and x_coord <= x_max:
            if y_coord >= y_min and y_coord <= y_max:
                if z_coord >= z_min and z_coord <= z_max:
                    print("Coordinates validated for object insertion!")
                    return True
        else:
            print("Coordinates invalid!")
            return False

    def generate_object_id(self, matrix):

        def random_gen(object_identifier):
            object_id = object_identifier + str(randint(0, len(self.coords) + len(self.coords)))
            return object_id

        if matrix is True:
            object_id = random_gen(object_identifier='B') # matrix material objects begin with a B
            while object_id in self.space['object_id'].tolist():
                object_id = random_gen(object_identifier='B')  # matrix material objects begin with a B
            return object_id
        else:
            object_id = random_gen(object_identifier='A') # non-matrix material objects begin with a A
            while object_id in self.space['object_id'].tolist():
                object_id = random_gen(object_identifier='A')  # matrix material objects begin with a A
            return object_id

    def insert_object(self, object, x_coord, y_coord, z_coord, object_radius, composition, initial_temperature):
        print("Inserting object...")
        if object in self.physical_parameters.index:
            if self.check_coords(x_coord=x_coord, y_coord=y_coord, z_coord=z_coord) is True: # checks to verify that coordinates exist in space
                for row in self.space.index:
                    if self.space['x_coords'][row] == x_coord:
                        if self.space['y_coords'][row] == y_coord:
                            if self.space['z_coords'][row] == z_coord: # verifies that coordinates match to Dataframe
                                self.space['object'][row] = object # the name of the object, as defined in dynamics/physical_parameters.csv
                                self.space['object_id'][row] = self.generate_object_id(matrix=False) # generates object ID
                                self.space['object_radius'][row] = object_radius # in m
                                self.space['volume'][row] = (4/3) * pi * (object_radius)**3 # assume volume of object is a perfect sphere
                                self.space['mass'][row] = self.physical_parameters['Density'][object] * self.space['volume'][row] # mass = density * volume
                                self.space['temperature'][row] = initial_temperature
                                self.space['object_density'] = float(self.space['mass'][row]) / ((4/3) * pi *
                                                float(self.space['object_radius'][row])**3) # assume object is a perfect sphere
                                self.solution.create_solution(box=self.space, composition=composition, row=row, object=object)
                                print("Inserted object ({}) at coordinates: x:{} y:{}, z:{}".format(self.space['object'][row],
                                                                                               self.space['x_coords'][row],
                                                                                               self.space['y_coords'][row],
                                                                                               self.space['z_coords'][row]))
            else:
                print("Could not insert object!  Outside of defined coordinate points!")
                sys.exit(1)
        else:
            print("Object not defined in {}!  Cannot insert object!".format(
                                                    os.getcwd() + "/dynamics/physical_parameters.csv"))
            sys.exit(1)

    # TODO: allow for the definition of matrix temperature or a matrix temperature gradient (starting temp, temp gradient
    def insert_matrix(self, matrix_material, composition, initial_temperature, z_range=[0,0]):
        print("Inserting matrix...")
        if matrix_material in self.physical_parameters.index:
            if z_range[0] != 0 and z_range[1] != 0: # z range is a list of two numbers, the minimum depth at the index 0, and the maximum depth at index 1
                for row in self.space.index:
                    if self.space['z_coords'][row] >= z_range[0] and self.space['z_coords'][row] <= z_range[1]:
                        self.space['object_id'][row] = self.generate_object_id(matrix=True)
                        self.space['object'][row] = matrix_material
                        self.space['temperature'][row] = initial_temperature
                        self.solution.create_solution(box=self.space, composition=composition, row=row, object=matrix_material)
                        print("Inserted matrix ({}) at coordinates: x:{} y:{}, z:{}".format(self.space['object'][row], self.space['x_coords'][row],
                                                            self.space['y_coords'][row], self.space['z_coords'][row]))
            else:
                for row in self.space.index:
                    self.space['object_id'][row] = self.generate_object_id(matrix=True)
                    self.space['object'][row] = matrix_material
                    self.space['temperature'][row] = initial_temperature
                    self.solution.create_solution(box=self.space, composition=composition, row=row, object=matrix_material)
                    print("Inserted matrix ({}) at coordinates: x:{} y:{}, z:{}".format(self.space['object'][row], self.space['x_coords'][row],
                                                            self.space['y_coords'][row], self.space['z_coords'][row]))

            print("Matrix inserted!")

        else:
            print("Matrix material not defined in {}!  Cannot insert matrix material!".format(
                os.getcwd() + "/dynamics/physical_parameters.csv"))
            sys.exit(1)


    def visualize_box(self):
        if self.visualize_system != False:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlim(xmin=min(self.space['x_coords']), xmax=max(self.space['x_coords']))
            ax.set_ylim(ymin=min(self.space['y_coords']), ymax=max(self.space['y_coords']))
            ax.set_zlim(zmin=min(self.space['z_coords']), zmax=max(self.space['z_coords']))
            # XX, YY, ZZ = np.meshgrid(self.space['x_coords'], self.space['y_coords'], self.space['z_coords'])
            for row in self.space.index:
                x = self.space['x_coords'][row]
                y = self.space['y_coords'][row]
                z = self.space['z_coords'][row]
                # velocity_x = self.space['x_direct'][row]
                # velocity_y = self.space['y_direct'][row]
                # velocity_z = self.space['z_direct'][row]
                if str(self.space['object_id'][row][0]) == 'A':
                    print("Plotted object at: x:{} y:{} z:{}.".format(x, y, z))
                    ax.scatter3D(x, y, z, color='b')
            ax.set_title("Sinking diapirs at Time {}".format(self.model_time))
            ax.set_xlabel("Box Length")
            ax.set_ylabel("Box Width")
            ax.set_zlabel("Box Height")
            ax.invert_zaxis()
            fig.savefig(os.getcwd()+'/mpl_animation1/snap_{}.png'.format(self.model_time), format='png')
            fig.clf()
            self.move_frames1.append('snap_{}.png'.format(self.model_time))
            print("System snapshot created: {}".format('snap_{}.png'.format(self.model_time)))


            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlim(xmin=min(self.space['x_coords']), xmax=max(self.space['x_coords']))
            ax.set_ylim(ymin=min(self.space['y_coords']), ymax=max(self.space['y_coords']))
            ax.set_zlim(zmin=min(self.space['z_coords']), zmax=max(self.space['z_coords']))
            # XX, YY, ZZ = np.meshgrid(self.space['x_coords'], self.space['y_coords'], self.space['z_coords'])
            for row in self.space.index:
                x = self.space['x_coords'][row]
                y = self.space['y_coords'][row]
                z = self.space['z_coords'][row]
                # velocity_x = self.space['x_direct'][row]
                # velocity_y = self.space['y_direct'][row]
                # velocity_z = self.space['z_direct'][row]
                if str(self.space['object_id'][row][0]) == 'A':
                    print("Plotted object at: x:{} y:{} z:{}.".format(x, y, z))
                    ax.scatter3D(x, y, z, color='b')
            # norm_colors = mpl.colors.Normalize(vmin=self.space['temperature'].min(), vmax=self.space['temperature'].max())
            norm_colors = mpl.colors.Normalize(vmin=1400, vmax=3000)
            colorsmap = matplotlib.cm.ScalarMappable(norm=norm_colors, cmap='jet')
            colorsmap.set_array(self.space['temperature'])
            ax.scatter(self.space['x_coords'], self.space['y_coords'], self.space['z_coords'], marker='s', s=140,
                       c=self.space['temperature'], cmap='jet', alpha=0.50)
            cb = fig.colorbar(colorsmap)
            ax.set_title("Sinking diapirs at Time {}".format(self.model_time))
            ax.set_xlabel("Box Length")
            ax.set_ylabel("Box Width")
            ax.set_zlabel("Box Height")
            ax.invert_zaxis()
            fig.savefig(os.getcwd()+'/mpl_animation2/snap_{}.png'.format(self.model_time), format='png')
            self.move_frames2.append('snap_{}.png'.format(self.model_time))
            fig.clf()

            x_coords = []
            y_coords = []
            temperature = []
            for row in self.space.index:
                if float(self.space['z_coords'][row]) == float(self.height):
                    x_coords.append(self.space['x_coords'][row])
                    y_coords.append(self.space['y_coords'][row])
                    temperature.append(self.space['temperature'][row])
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_trisurf(x_coords, y_coords, temperature)
            ax.set_xlabel("Box Length")
            ax.set_ylabel("Box Width")
            ax.set_zlabel("Temperature (degK)")
            ax.set_zlim(zmin=min(self.space['z_coords']), zmax=max(self.space['z_coords']))
            ax.set_title("Temperature Distribution at Time {} At Base of Model".format(self.model_time))
            fig.savefig(os.getcwd() + '/mpl_animation4/snap_{}.png'.format(self.model_time), format='png')
            self.move_frames4.append('snap_{}.png'.format(self.model_time))
            fig.clf()



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
        for row in system_data.index:
            if system_data['x_coords'][row] == x_coord:
                if system_data['y_coords'][row] == y_coord:
                    if system_data['z_coords'][row] == z_coord:
                        return row

    @staticmethod
    def swap_rows(system_data, updated_system_data, from_row_index, to_row_index):
        updated_system = updated_system_data
        for i in updated_system_data:
            if i != 'x_coords' and i != 'y_coords' and i != 'z_coords':
                # print("From index: {}       To index: {}".format(from_row_index, to_row_index))
                updated_system[i][to_row_index] = system_data[i][from_row_index]
                updated_system[i][from_row_index] = system_data[i][to_row_index]
        return updated_system

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



    @classmethod
    def move_systems(clf, system_data, update_space, deltaTime, box_height, space_resolution, default_matrix_material='Silicate Liquid'):
        update_space_copy = update_space.copy(deep=True)
        for row in system_data.index:
            if str(system_data['object_id'][row][0]) == 'A':
                curr_x_coords = system_data['x_coords'][row]
                curr_y_coords = system_data['y_coords'][row]
                curr_z_coords = system_data['z_coords'][row]
                object_velocity = 0

                matrix_material = default_matrix_material # the default matrix matrial until overwritten
                matrix_material_temp = 0.0
                matrix_material_pressure = 0.0
                # assumption is that object will travel through matrix most like that occupying z coord below it.
                # code block below attempts to idenfity that material
                if (system_data['z_coords'][row] + space_resolution) in system_data['z_coords']:
                    searchfor_coord = (system_data['z_coords'][row] + space_resolution)
                    for row2 in system_data.index:
                        if system_data['z_coords'][row2] == searchfor_coord and system_data['y_coords'][row2] \
                        == curr_y_coords and system_data['x_coords'][row2] == curr_x_coords:
                            matrix_material = system_data['object'][row2]
                            matrix_material_temp = system_data['temperature'][row2]
                            matrix_material_pressure = system_data['pressure'][row2]
                            break
                object_velocity = move_particle(body_type=system_data['object'][row],
                        system_params=system_data).stokes_settling(object=system_data['object'][row], matrix_material=matrix_material,
                        matrix_material_temp=matrix_material_temp, matrix_material_pressure=matrix_material_pressure)

                z_dis_obj_travel = object_velocity * deltaTime
                updated_x_coords = system_data['x_coords'][row]
                updated_y_coords = system_data['y_coords'][row]
                updated_z_coords = clf.round_coord_arbitrary(coordinate=(z_dis_obj_travel + system_data['z_coords'][row]),
                                                             system_data=system_data, coordinate_type='z_coords')
                rounded_distance_travelled = updated_z_coords - curr_z_coords
                if rounded_distance_travelled == 0:
                    object_velocity = 0
                    z_dis_obj_travel = 0
                system_data['temperature'][row] = float(
                    system_data['temperature'][row]) + energy().stokes_frictional_energy(
                    body_material=system_data['object'][row], matrix_material=matrix_material,
                    body_radius=system_data['object_radius'][row],
                    body_mass=system_data['mass'][row], distance_travelled=z_dis_obj_travel
                )
                system_data['object_velocity'][row] = object_velocity
                system_data['z_direct'][row] = object_velocity
                system_data['potential_energy'][row] = energy().potential_energy(mass=system_data['mass'][row],
                                                                                 height=system_data['z_coords'][row],
                                                                                 box_height=box_height)
                system_data['kinetic_energy'][row] = energy().kinetic_energy(mass=system_data['mass'][row],
                                                                             velocity=system_data['object_velocity'][
                                                                                 row])
                print("Object will move! {} ({}) will move from x:{} y:{} z:{} to x:{} y:{} z:{}".format(
                    system_data['object_id'][row], system_data['object'][row], system_data['x_coords'][row],
                    system_data['y_coords'][row], system_data['z_coords'][row], updated_x_coords, updated_y_coords,
                    updated_z_coords))
                from_row_index = clf.grab_row_index_by_coord(system_data=system_data,
                                                              x_coord=system_data['x_coords'][row],
                                                              y_coord=system_data['y_coords'][row],
                                                              z_coord=system_data['z_coords'][row])
                to_row_index = clf.grab_row_index_by_coord(system_data=system_data,
                                                            x_coord=updated_x_coords,
                                                            y_coord=updated_y_coords,
                                                            z_coord=updated_z_coords)
                update_space_copy = clf.swap_rows(system_data=system_data, updated_system_data=update_space,
                                              from_row_index=from_row_index, to_row_index=to_row_index)
        return update_space_copy



    # TODO: update x and y coords
    def update_system(self, auto_update=True, deltaTime=1.0):
        print("Model time at: {}".format(self.model_time))
        update_space = self.space.copy(deep=True)
        if self.model_time == self.initial_time:
            self.classify_neighbors(animate_neighbors=False)
            self.visualize_box()
            if self.object_history == True:
                for row in self.space.index:
                    if 'A' in self.space['object_id'][row]:
                        contents = []
                        contents.append(str(self.model_time))
                        for i in self.space:
                            contents.append(str(self.space[i][row]))
                        formatted_contents = ",".join(i for i in contents)
                        self.object_output.write("{}\n".format(formatted_contents))
        elif self.model_time <= 0:
            self.visualize_box()
            print("Model at minimum time!")
            if self.visualize_system == True:
                print("Writing animations...")

                # dynamics animation
                os.chdir(os.getcwd()+'/mpl_animation1')
                animation = mpy.ImageSequenceClip(self.move_frames1, fps=round((self.initial_time/(self.initial_time/3))), load_images=True)
                os.chdir('..')
                animation.write_videofile('fEquilibrium_animation.mp4',
                                          fps=round((self.initial_time / (self.initial_time / 3))), audio=False)
                animation.write_gif('fEquilibrium_animation.gif', fps=round((self.initial_time/(self.initial_time/3))))
                print("Animation created & available in {}!".format(os.getcwd()))

                # 3d heatmap animation
                os.chdir(os.getcwd() + '/mpl_animation2')
                animation = mpy.ImageSequenceClip(self.move_frames2,
                                                  fps=round((self.initial_time / (self.initial_time / 3))),
                                                  load_images=True)
                os.chdir('..')
                animation.write_videofile('thermal_fEquilibrium_animation.mp4',
                                          fps=round((self.initial_time / (self.initial_time / 3))), audio=False)
                animation.write_gif('thermal_fEquilibrium_animation.gif',
                                    fps=round((self.initial_time / (self.initial_time / 3))))
                print("Animation created & available in {}!".format(os.getcwd()))

                # 3d surface heat distribution animation
                os.chdir(os.getcwd() + '/mpl_animation4')
                animation = mpy.ImageSequenceClip(self.move_frames4,
                                                  fps=round((self.initial_time / (self.initial_time / 3))),
                                                  load_images=True)
                os.chdir('..')
                animation.write_videofile('time_t_distrib.mp4',
                                          fps=round((self.initial_time / (self.initial_time / 3))), audio=False)
                animation.write_gif('time_t_distrib.gif',
                                    fps=round((self.initial_time / (self.initial_time / 3))))
                print("Animation created & available in {}!".format(os.getcwd()))

                self.space.to_csv("space.csv")
                self.solution.get_solution().to_csv("solution.csv")
                if self.object_history == True:
                    for row in self.space.index:
                        if 'A' in self.space['object_id'][row]:
                            contents = []
                            contents.append(str(self.model_time))
                            for i in self.space:
                                contents.append(str(self.space[i][row]))
                            formatted_contents = ",".join(i for i in contents)
                            self.object_output.write("{}\n".format(formatted_contents))
                if self.object_output == True:
                    self.object_output.close()
                return self.model_time, self.space
        else:
            update_space = self.move_systems(system_data=self.space, update_space=update_space, deltaTime=deltaTime,
                                             box_height=self.height, space_resolution=self.space_resolution)
            update_solution = self.solution.update_solution(deltaTime=deltaTime)
            therm_eq_update_space = thermal_eq().D3_thermal_eq(system_data=update_space, deltaTime=deltaTime, space_resolution=self.space_resolution)
            for row in update_space.index:
                if update_space['object'][row] == 'Metal Liquid':
                    self.velocity_output.write("\n{}".format(update_space['object_velocity'][row]))
            self.visualize_box()
            self.space = update_space
            if self.object_history == True:
                for row in self.space.index:
                    if 'A' in self.space['object_id'][row]:
                        contents = []
                        contents.append(str(self.model_time))
                        for i in self.space:
                            contents.append(str(self.space[i][row]))
                        formatted_contents = ",".join(i for i in contents)
                        self.object_output.write("{}\n".format(formatted_contents))
        if auto_update == True:
            if self.model_time == 1:
                self.model_time -= deltaTime
                self.update_system(auto_update=False, deltaTime=1)
            elif self.model_time > 1:
                self.model_time -= deltaTime
                self.update_system(auto_update=auto_update, deltaTime=(self.model_time/self.model_time))
            else:
                return self.model_time, self.space
        else:
            return self.model_time, self.space


