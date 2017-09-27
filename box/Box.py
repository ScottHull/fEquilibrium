import os
import numpy as np
import pandas as pd
from random import randint
import mayavi.mlab as mlab
import moviepy.editor as mpy
os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from dynamics.Movement import move_particle


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
            'object_id': ['NaN' for i in list(range(len(self.coords)))], 'object': ['NaN' for i in list(range(len(self.coords)))],
            'x_coords': [i[0] for i in self.coords], 'y_coords': [i[1] for i in self.coords],
            'z_coords': [i[2] for i in self.coords], 'object_size': [np.NAN for i in list(range(len(self.coords)))],
            'density': [0 for i in list(range(len(self.coords)))], 'temperature': [0 for i in list(range(len(self.coords)))],
            'pressure': [0 for i in list(range(len(self.coords)))],
            'object_velocity': [0 for i in list(range(len(self.coords)))],
            'x_direct': [0 for i in list(range(len(self.coords)))], 'y_direct': [0 for i in list(range(len(self.coords)))],
            'z_direct': [0 for i in list(range(len(self.coords)))], 'potential_energy': [0 for i in list(range(len(self.coords)))],
            'kinematic_energy': [0 for i in list(range(len(self.coords)))]
        })
        self.mov_frames = []


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


    def check_coords(self, x_coord, y_coord, z_coord):
        print("Checking if coordinates are valid...")
        x_min, x_max = self.space['x_coords'][0], self.space['x_coords'][len(self.coords) - 1]
        y_min, y_max = self.space['y_coords'][0], self.space['y_coords'][len(self.coords) -1]
        z_min, z_max = self.space['z_coords'][0], self.space['z_coords'][len(self.coords) -1]
        if x_coord >= x_min and x_coord <= x_max and y_coord >= y_min and y_coord <= y_max and \
                        z_coord >= z_min and z_coord <= z_max:
            print("Coordinates validated!")
            return True
        else:
            print("Coordinates invalid!")
            return False

    def generate_object_id(self, matrix):
        unref_object_id = randint(0, 99999999999999999999999)
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
                if self.space['x_coord'][row] == x_coord:
                    if self.space['y_coord'][row] == y_coord:
                        if self.space['z_coord'][row] == z_coord: # verifies that coordinates match to Dataframe
                            self.space['object'][row] = object
                            self.space['object_id'][row] = self.generate_object_id(matrix=False) # generates object ID
                            self.space['object_size'] = object_size
                            self.space['object_velocity']['z_direct'] = move_particle(body_type='fe alloy',
                                                                        system_params=self.space).stokes_settling()
        print("Object inserted!")

    def insert_matrix(self, matrix_material):
        print("Inserting matrix...")
        for row in self.space.index:
            self.space['object_id'][row] = self.generate_object_id(matrix=True)
            self.space['object'][row] = matrix_material
            print("Inserted matrix at coordinates: x:{} y:{}, z:{}".format(self.space['x_coords'][row], self.space['y_coords'][row], self.space['z_coords'][row]))
        print("Matrix inserted!")


    def visualize_box(self):
        if self.visualize_system != False:
            mlab.clf()
            for row in self.space.index:
                if str(self.space['object_id'][row][0]) == 'A':
                    x = self.space['x_coords'][row]
                    y = self.space['x_coords'][row]
                    z = self.space['x_coords'][row]
                    object_size = self.space['object_size'][row]
                    mlab.points3d(x, y, z, object_size, scale_factor=1)
                    mlab.view(distance=12)
                    self.mov_frames.append(mlab.screenshot(antialiased=True))


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
        for row in system_data.index():
            if system_data['x_coords'][row] == x_coord:
                if system_data['y_coords'][row] == y_coord:
                    if system_data['z_coords'][row] == z_coord:
                        return row

    @staticmethod
    def swap_rows(system_data, updated_system_data, from_row_index, to_row_index):
        updated_system = updated_system_data
        for i in updated_system_data:
            if i != 'x_coords' and i != 'y_coords' and i != 'z_coords':
                updated_system[i][to_row_index] = system_data[i][from_row_index]
        return updated_system





    # TODO: update x and y coords
    def update_system(self, deltaTime):
        self.model_time -= deltaTime
        print("Model time at: {}")
        update_space = self.space
        if self.model_time == self.initial_time:
            self.visualize_box()
        elif self.model_time <= 0:
            print(self.space)
            if self.visualize_system == True:
                animation = mpy.ImageSequenceClip(self.mov_frames, fps=30)
                animation.write_videofile('fEquilibrium_animation.mp4', fps=30, audio=False)
        else:
            for row in self.space.index:
                row_object = self.space.index[row]
                if str(self.space['object_id'][row][0]) == 'A':
                    z_dis_obj_travel = (move_particle(body_type='fe alloy',
                                        system_params=self.space).stokes_settling()) * deltaTime
                    # updated_x_coords = 0
                    # updated_y_coords = 0
                    updated_z_coords = round(z_dis_obj_travel, -len(str(self.space_resolution)))
                    from_row_index = self.grab_row_index_by_coord(system_data=self.space, x_coord=self.space['x_coords'][row],
                                                                    y_coord=self.space['y_coords'][row],
                                                                    z_coord=self.space['z_coords'][row])
                    to_row_index = self.grab_row_index_by_coord(system_data=update_space, x_coord=self.space['x_coords'][row],
                                                                    y_coord=self.space['y_coords'][row],
                                                                    z_coord=updated_z_coords)
                    update_space = self.swap_rows(system_data=self.space, updated_system_data=update_space,
                                                  from_row_index=from_row_index, to_row_index=to_row_index)
                    self.visualize_box()
            self.visualize_box()
        self.space = update_space
        return self.model_time


