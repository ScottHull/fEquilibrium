import matplotlib as mpl
mpl.use('Qt5Agg')
import os
import numpy as np
import pandas as pd
from random import randint
import moviepy.editor as mpy
os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from dynamics.Movement import move_particle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import re

# TODO: update some methods to class methods to avoid outside interference
class box:

    def __init__(self, length, width, height, space_resolution, model_time, visualize_system=False):
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
        self.model_time = model_time
        self.initial_time = model_time
        self.coords = self.generate_coordinate_points(length=self.length, width=self.width, height=self.height,
                                                 space_resolution=self.space_resolution)
        self.visualize_system = visualize_system
        self.space = pd.DataFrame({
            'object_id': [np.NAN for i in list(range(len(self.coords)))], 'object': [np.NAN for i in list(range(len(self.coords)))],
            'x_coords': [float(i[0]) for i in self.coords], 'y_coords': [float(i[1]) for i in self.coords],
            'z_coords': [float(i[2]) for i in self.coords], 'object_size': [np.NAN for i in list(range(len(self.coords)))],
            'density': [np.NAN for i in list(range(len(self.coords)))], 'temperature': [np.NAN for i in list(range(len(self.coords)))],
            'pressure': [np.NAN for i in list(range(len(self.coords)))],
            'object_velocity': [np.NAN for i in list(range(len(self.coords)))],
            'x_direct': [np.NAN for i in list(range(len(self.coords)))], 'y_direct': [np.NAN for i in list(range(len(self.coords)))],
            'z_direct': [np.NAN for i in list(range(len(self.coords)))], 'potential_energy': [np.NAN for i in list(range(len(self.coords)))],
            'kinematic_energy': [np.NAN for i in list(range(len(self.coords)))]
        })
        self.mov_frames = []
        if os.path.exists('mpl_pics'):
            shutil.rmtree('mpl_pics')
        os.mkdir('mpl_pics')


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
                    temp_coords.append(i)
                    temp_coords.append(j)
                    temp_coords.append(q)
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
        print("Checking if coordinates are valid...")
        x_min, x_max = self.space['x_coords'][0], self.space['x_coords'][len(self.coords) - 1]
        y_min, y_max = self.space['y_coords'][0], self.space['y_coords'][len(self.coords) -1]
        z_min, z_max = self.space['z_coords'][0], self.space['z_coords'][len(self.coords) -1]
        if x_coord >= x_min and x_coord <= x_max:
            if y_coord >= y_min and y_coord <= y_max:
                if z_coord >= z_min and z_coord <= z_max:
                    print("Coordinates validated!")
                    return True
        else:
            print("Coordinates invalid!")
            return False

    def generate_object_id(self, matrix):
        unref_object_id = randint(0, len(self.coords))
        str_unref_object_id = str(unref_object_id)
        if matrix is True:
            object_id = 'B' + str_unref_object_id # matrix material objects begin with a B
            if object_id in self.space['object_id']:
                self.generate_object_id(matrix=matrix)
            else:
                return object_id
        else:
            object_id = 'A' + str_unref_object_id # non-matrix material objects begin with a A
            if object_id in self.space['object_id']:
                self.generate_object_id(matrix=matrix)
            else:
                return object_id

    def insert_object(self, object, x_coord, y_coord, z_coord, object_size):
        print("Inserting object...")
        if self.check_coords(x_coord=x_coord, y_coord=y_coord, z_coord=z_coord) is True: # checks to verify that coordinates exist in space
            for row in self.space.index:
                if self.space['x_coords'][row] == x_coord:
                    if self.space['y_coords'][row] == y_coord:
                        if self.space['z_coords'][row] == z_coord: # verifies that coordinates match to Dataframe
                            self.space['object'][row] = object
                            self.space['object_id'][row] = self.generate_object_id(matrix=False) # generates object ID
                            self.space['object_size'][row] = object_size

    def insert_matrix(self, matrix_material):
        print("Inserting matrix...")
        for row in self.space.index:
            self.space['object_id'][row] = self.generate_object_id(matrix=True)
            self.space['object'][row] = matrix_material
            print("Inserted matrix at coordinates: x:{} y:{}, z:{}".format(self.space['x_coords'][row], self.space['y_coords'][row], self.space['z_coords'][row]))
        print("Matrix inserted!")


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
                # else:
                #     print("Plotted quiver at: x:{} y:{} z:{}.".format(x, y, z))
                #     for i in list(range(6))[1:]:
                #         if 'A' in self.space['object_id'][row + i]:
                #             ax.quiver(XX, YY, ZZ, velocity_x, velocity_y, velocity_z, length=0.4)
            ax.set_title("Sinking diapirs at {} years ago".format(self.model_time))
            ax.set_xlabel("Box Length")
            ax.set_ylabel("Box Width")
            ax.set_zlabel("Box Height")
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.invert_zaxis()
            plt.savefig(os.getcwd()+'/mpl_pics/snap_{}.png'.format(self.model_time), format='png')


            # mlab.clf()
            # for row in self.space.index:
            #     x = self.space['x_coords'][row]
            #     y = self.space['y_coords'][row]
            #     z = self.space['z_coords'][row]
            #     velocity_x = self.space['x_direct'][row]
            #     velocity_y = self.space['y_direct'][row]
            #     velocity_z = self.space['z_direct'][row]
            #     object_size = self.space['object_size'][row]
            #     if str(self.space['object_id'][row][0]) == 'A':
            #         mlab.points3d(x, y, z, object_size, scale_factor=1)
            #     else:
            #         pass
            #         # mlab.flow(velocity_x, velocity_y, velocity_z)
            # mlab.view(distance=12)
            # # scene = mlab.screenshot(antialiased=True)
            # fig = plt.figure()
            # scene = mlab.screenshot()
            # pl.imshow(scene)
            # plt.grid()
            # plt.show()
            # plt.close()
            # self.mov_frames.append(scene)


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
            # print("Swapping coordinate data from x:{} y:{} z:{} with x:{} y:{} z:{}".format(system_data['x_coords'][to_row_index],
            #                                                                                 system_data['y_coords'][
            #                                                                                     to_row_index],
            #                                                                                 system_data['z_coords'][
            #                                                                                     to_row_index], system_data['x_coords'][from_row_index],
            #                                                                                 system_data['y_coords'][from_row_index], system_data['z_coords'][from_row_index]))
            if i != 'x_coords' and i != 'y_coords' and i != 'z_coords':
                # print("From index: {}       To index: {}".format(from_row_index, to_row_index))
                updated_system[i][to_row_index] = system_data[i][from_row_index]
                updated_system[i][from_row_index] = system_data[i][to_row_index]
        return updated_system
    
    # TODO: seperate velocity calculations from system movement so space dataframe can be updated and moved according to velocity contents
    @classmethod
    def calculate_velocities(cls):
        pass
    
    @classmethod
    def move_systems(clf, system_data, update_space, deltaTime):
        update_space_copy = update_space.copy(deep=True)
        for row in system_data.index:
            if str(system_data['object_id'][row][0]) == 'A':
                object_velocity = move_particle(body_type='fe alloy',
                                                  system_params=system_data).stokes_settling()
                system_data['object_velocity'][row] = object_velocity
                system_data['z_direct'][row] = object_velocity
                z_dis_obj_travel = object_velocity * deltaTime
                updated_x_coords = system_data['x_coords'][row]
                updated_y_coords = system_data['y_coords'][row]
                updated_z_coords = clf.round_coord_arbitrary(coordinate=(z_dis_obj_travel + system_data['z_coords'][row]),
                                                             system_data=system_data, coordinate_type='z_coords')
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

    def atof(self, text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    def natural_keys(self, text):
        return [self.atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

    # TODO: update x and y coords
    def update_system(self, deltaTime):
        print("Model time at: {}".format(self.model_time))
        update_space = self.space.copy(deep=True)
        if self.model_time == self.initial_time:
            self.visualize_box()
        elif self.model_time <= 0:
            self.visualize_box()
            print("Model at minimum time!")
            if self.visualize_system == True:
                self.space.to_csv('space.csv')
                self.mov_frames.append(i for i in os.listdir(os.getcwd()+'/mpl_pics'))
                os.chdir(os.getcwd()+'/mpl_pics')
                animation = mpy.ImageSequenceClip(list(reversed([z for z in os.listdir(os.getcwd())])).sort(key=self.natural_keys), fps=round((self.model_time)/15), load_images=True)
                os.chdir('..')
                animation.write_videofile('fEquilibrium_animation.mp4', fps=1, audio=False)
                animation.write_gif('Equilibrium_animation.gif', fps=round((self.model_time)/20))
        else:
            update_space = self.move_systems(system_data=self.space, update_space=update_space, deltaTime=deltaTime)
            self.visualize_box()
        self.space = update_space
        self.model_time -= deltaTime


