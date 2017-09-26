import os
import numpy as np
import pandas as pd
from random import randint
import mayavi.mlab as mlab
import moviepy.editor as mpy
os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from dynamics.Movement import move_particle

class box:

    def __init__(self, length, width, height, space_resolution, model_time, initial_time, visualize_system=False):
        self.length = length
        self.width = width
        self.height = height
        self.space_resolution = space_resolution
        self.model_time = model_time
        self.initial_time = initial_time
        self.visualize_system = visualize_system
        self.x_coords = np.linspace(0, self.length, endpoint=True, retstep=self.space_resolution)
        self.y_coords = np.linspace(0, self.width, endpoint=True, retstep=self.space_resolution)
        self.z_coords = np.linspace(0, self.height, endpoint=True, retstep=self.space_resolution)
        self.space = pd.DataFrame({
            'object_id': [].append(self.generate_obj_id(matrix=True) for i in list(range(len(self.x_coords.tolist())))),
            'object': [], 'x_coords': self.x_coords.tolist(), 'y_coords': self.y_coords.tolist(),
            'z_coords': self.z_coords.tolist(), 'object_size': [], 'temperature': [], 'pressure': [],
            'object_velocity': {'x_direct': [], 'y_direct': [], 'z_direct': []},
        }, index='object_id')
        self.mov_frames = []

    def check_coords(self, x_coord, y_coord, z_coord):
        x_min, x_max = self.x_coords[0], self.x_coords[:-1]
        y_min, y_max = self.y_coords[0], self.y_coords[:-1]
        z_min, z_max = self.z_coords[0], self.z_coords[:-1]
        if x_coord >= x_min and x_coord <= x_max and y_coord >= y_min and y_coord <= y_max and \
                        z_coord >= z_min and z_coord <= z_max:
            return True
        else:
            return False

    def generate_obj_id(self, matrix):
        unref_obj_id = randint(1000, 999999)
        str_unref_obj_id = str(unref_obj_id)
        if matrix is True:
            obj_id = int('0' + str_unref_obj_id) # matrix material objects begin with a 0
            if obj_id in self.space['obj_id']:
                self.generate_obj_id(matrix=matrix)
            else:
                return obj_id
        else:
            obj_id = int('1' + str_unref_obj_id) # non-matrix material objects begin with a 1
            if obj_id in self.space['obj_id']:
                self.generate_obj_id(matrix=matrix)
            else:
                return obj_id


    def insert_object(self, object, x_coord, y_coord, z_coord, matrix, object_size):
        if self.check_coords(x_coord=x_coord, y_coord=y_coord, z_coord=z_coord) is True: # checks to verify that coordinates exist in space
            for row in self.space.index:
                if self.space['x_coord'][row] == x_coord:
                    if self.space['y_coord'][row] == y_coord:
                        if self.space['z_coord'][row] == z_coord: # verifies that coordinates match to Dataframe
                            self.space['object'][row] = object
                            self.space['object_id'][row] = self.generate_obj_id(matrix=matrix) # generates object ID
                            self.space['object_size'] = object_size
                            self.space['object_velocity']['z_direct'] = move_particle(body_type='fe alloy',
                                                                        system_params=self.space).stokes_settling()


    def visualize_box(self):
        if self.visualize_system != False:
            mlab.clf()
            for row in self.space.index:
                if str(self.space['object_id'][row][0]) == '0':
                    x = self.space['x_coords'][row]
                    y = self.space['x_coords'][row]
                    z = self.space['x_coords'][row]
                    object_size = self.space['object_size'][row]
                    mlab.points3d(x, y, z, object_size, scale_factor=1)
                    mlab.view(distance=12)
                    self.mov_frames.append(mlab.screenshot(antialiased=True))


    def update_system(self, deltaTime):

            update_space = self.space
            if self.model_time == self.initial_time:
                self.visualize_box()
            elif self.model_time <= 0:
                animation = mpy.ImageSequenceClip(self.mov_frames, fps=30)
                animation.write_videofile('fEquilibrium_animation.mp4', fps=30, audio=False)
            else:
                self.model_time = self.model_time - deltaTime
                for row in self.space.index:
                    row_object = self.space.index[row]
                    if str(self.space['object_id'][row][0]) == '1':
                        z_dis_obj_travel = (move_particle(body_type='fe alloy',
                                            system_params=self.space).stokes_settling()) * deltaTime
                        updated_z_cords = z_dis_obj_travel.round(len(str(self.space_resolution)))

                self.visualize_box()


