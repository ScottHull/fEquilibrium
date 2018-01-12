import os
import sys
import shutil
import pandas as pd
import matplotlib
import matplotlib as mpl

mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from math import pi
import numpy as np
from datetime import datetime
from decimal import getcontext, Decimal
import warnings

warnings.filterwarnings('ignore')
import ast
import moviepy.editor as mpy


class line:

    def __init__(self, max_r, radial_mesh_res, model_time, time_interval, partitioning_coefficient):
        self.partitioning_coefficient = Decimal(partitioning_coefficient)
        self.max_time = model_time
        self.model_time = model_time
        self.time_interval = time_interval
        getcontext().prec = len(str(radial_mesh_res)) - 1
        self.radial_mesh_res = Decimal(radial_mesh_res)
        self.max_r = Decimal(max_r)
        self.min_boundary_range = [Decimal(0), Decimal(0) + Decimal(self.radial_mesh_res)]
        self.max_boundary_range = [Decimal(self.max_r) - Decimal(self.radial_mesh_res), Decimal(self.max_r)]
        self.diapir_r = np.NAN
        self.diapir_theta = Decimal(Decimal(2) * Decimal(pi))
        self.diapir_phi = Decimal(Decimal(2) * Decimal(pi))
        self.coords = self.generate_coords()
        self.space = pd.DataFrame(
            {
                'object': np.NAN,
                'r_coords': [i[0] for i in self.coords],
                'coord_index': [str(i) for i in self.coords],
                'neighbors': np.NAN,
                'concentration': [0 for i in self.coords],
                'diffusivity': np.NAN,
                'dc/dt': np.NAN,
            }
        )
        self.max_r_diapir = 0
        self.max_r_diapir_index = 0

        self.neighbors = self.generate_neighbors()

        self.conc_frames = []
        if 'concentration_frames' in os.listdir(os.getcwd()):
            shutil.rmtree(os.getcwd() + '/concentration_frames')
        os.mkdir(os.getcwd() + '/concentration_frames')

    def generate_coords(self):
        console.pm_stat("Generating coordinates...")
        r_range = np.arange(0, self.max_r + self.radial_mesh_res, self.radial_mesh_res)

        coords = []
        for r in r_range:
            temp_coords = []
            temp_coords.append(Decimal(r))
            coords.append(temp_coords)
        console.pm_stat("Coordinates generated!")

        return coords

    def generate_neighbors(self):
        curr_coord = 1
        total_coords = len(self.space.index.tolist())
        for row in self.space.itertuples():
            index = row.Index
            curr_r = Decimal(self.space['r_coords'][index])
            console.pm_flush("Finding neighbors for {}/{} (curr_r: {})".format(curr_coord, total_coords,
                                                                               self.space['r_coords'][index]))
            r_plus = curr_r + Decimal(self.radial_mesh_res)
            r_minus = curr_r - Decimal(self.radial_mesh_res)
            if not (self.min_boundary_range[0] <= curr_r <= self.min_boundary_range[1]) and not \
                    ((self.max_boundary_range[0] <= curr_r <= self.max_boundary_range[1])):
                neighbors = {
                    'r': {
                        'r': {
                            'index': index
                        },
                        'r+': {
                        },
                        'r-': {
                        }
                    },
                }
                r_plus_index = self.space.index[self.space['r_coords'] == r_plus].values[0]
                r_minus_index = self.space.index[self.space['r_coords'] == r_minus].values[0]
                neighbors['r']['r+'].update({'index': r_plus_index})
                neighbors['r']['r-'].update({'index': r_minus_index})

                self.space['neighbors'][index] = str(neighbors)
                curr_coord += 1
            else:
                curr_coord += 1
        return None

    def insert_diapir(self, radius, concentration, diffusivity):
        console.pm_stat("Inserting diapir...")
        radius = Decimal(radius)
        diapir_range = np.arange(0, radius + self.radial_mesh_res, self.radial_mesh_res)
        for i in diapir_range:
            index = self.space.index[self.space['r_coords'] == Decimal(i)].values[0]
            console.pm_flush("Inserting diapir at radius: {}".format(self.space['r_coords'][index]))
            self.space['object'][index] = 'diapir'
            self.space['concentration'][index] = Decimal(concentration)
            self.space['diffusivity'][index] = Decimal(diffusivity)
        self.diapir_df = self.space.query('object == "diapir"')
        self.max_r_diapir = self.space['r_coords'][
            self.space.index[self.space['r_coords'] == self.diapir_df['r_coords'].max()].values[0]]
        self.max_r_diapir_index = self.space.index[self.space['r_coords'] == self.diapir_df['r_coords'].max()].values[0]
        console.pm_stat("Diapir inserted!")

    def insert_melt(self, concentration, diffusivity):
        console.pm_stat("Inserting melt...")
        for row in self.space.itertuples():
            index = row.Index
            if self.space['object'][index] != 'diapir':
                console.pm_flush("Inserting melt at radius: {}".format(self.space['r_coords'][index]))
                self.space['object'][index] = 'silicate_melt'
                self.space['concentration'][index] = Decimal(concentration)
                self.space['diffusivity'][index] = Decimal(diffusivity)
        console.pm_stat("Melt inserted!")

    def insert_boundary(self, concentration_diapir, concentration_melt):
        console.pm_stat("Inserting boundary conditions...")
        for i in self.min_boundary_range:
            index = self.space.index[self.space['r_coords'] == i].values[0]
            self.space['object'][index] = 'boundary'
            self.space['concentration'][index] = concentration_diapir
        for i in self.max_boundary_range:
            index = self.space.index[self.space['r_coords'] == i].values[0]
            self.space['object'][index] = 'boundary'
            self.space['concentration'][index] = concentration_melt

    def partition(self, coefficient):
        diapir_df = self.space.query('object == "diapir"')
        index = self.space.index[self.space['r_coords'] == diapir_df['r_coords'].max()].values[0]
        melt_boundary_index = index + 1
        melt_boundary_abundance = self.space['concentration'][melt_boundary_index]
        partition_abundance = coefficient * melt_boundary_abundance
        if partition_abundance >= melt_boundary_abundance:
            partition_abundance = self.space['concentration'][melt_boundary_index]
        self.space['concentration'][index] = self.space['concentration'][index] + partition_abundance
        self.space['concentration'][melt_boundary_index] = self.space['concentration'][
                                                               melt_boundary_index] - partition_abundance
        return None

    def diffusivity(self):
        self.partition(coefficient=self.partitioning_coefficient)
        update_space = self.space.copy(deep=True)
        for row in self.space.itertuples():
            index = row.Index
            if self.space['object'][index] != 'boundary' and self.space['object'][index] != 'diapir':
                neighbors = ast.literal_eval(self.space['neighbors'][index])
                r_plus_index = neighbors['r']['r+']['index']
                r_minus_index = neighbors['r']['r-']['index']
                r_conc = self.space['concentration'][index]
                r_plus_conc = self.space['concentration'][r_plus_index]
                r_minus_conc = self.space['concentration'][r_minus_index]
                # fick's 2nd law
                c_laplacian = (r_plus_conc - (2 * r_conc) + r_minus_conc) / (self.radial_mesh_res ** 2)
                r_diffusivity = self.space['diffusivity'][index]
                dC_dt = Decimal(r_diffusivity) * Decimal(c_laplacian)
                update_space['dc/dt'][index] = dC_dt
                dC = dC_dt * self.time_interval
                update_space['concentration'][index] = r_conc + dC

        # homogenize diapir concentration
        diapir_indecies = []
        diapir_df = self.space.query('object == "diapir"')
        for row in diapir_df.itertuples():
            index = row.Index
            diapir_indecies.append(index)
        for i in diapir_indecies:
            conc_diapir = update_space['concentration'][self.max_r_diapir_index]
            update_space['concentration'][i] = conc_diapir



        # adjust the diapir boundary condition to share a nearby gradient
        diapir_boundary_index = self.space.index[self.space['r_coords'] == self.min_boundary_range[1]].values[0]
        diapir_boundary_index_inner = self.space.index[self.space['r_coords'] == self.min_boundary_range[0]].values[0]
        diapir_boundary_index_neighbor = self.space.index[self.space['r_coords'] == self.min_boundary_range[1]
                                            + self.radial_mesh_res].values[0]
        diapir_boundary_index_neighbor_rplus = self.space.index[
            self.space['r_coords'] == self.min_boundary_range[1] + (Decimal(2) * self.radial_mesh_res)].values[0]
        diff = self.space['concentration'][diapir_boundary_index_neighbor] - self.space['concentration'][
            diapir_boundary_index_neighbor_rplus]
        update_space['concentration'][diapir_boundary_index] = self.space['concentration'][diapir_boundary_index] + diff
        update_space['concentration'][diapir_boundary_index_inner] = update_space['concentration'][
            diapir_boundary_index]
        self.space = update_space
        return None

    def visualize_results(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("W Concentration Over Diapir Radius {} at Time {}s".format(self.max_r, self.model_time))
        ax.set_xlabel('Radius (m)')
        ax.set_ylabel('W Concentration (%)')
        ax.plot(self.space['r_coords'], self.space['concentration'])
        ax.axvspan(0, self.max_r_diapir, alpha=0.2, color='red')
        ax.grid()
        fig.savefig(os.getcwd() + "/concentration_frames/{}.png".format(self.model_time), format='png')
        self.conc_frames.append("{}.png".format(self.model_time))
        fig.clear()

    def write_animation(self):
        os.chdir(os.getcwd() + '/concentration_frames')
        animation = mpy.ImageSequenceClip(self.conc_frames,
                                          fps=round((self.max_time / (self.max_time / 3))),
                                          load_images=True)
        animation.write_videofile('chemical_diffusivity.mp4',
                                  fps=round((self.max_time / (self.max_time / 3))), audio=False)
        animation.write_gif('chemical_diffusivity.gif',
                            fps=round((self.max_time / (self.max_time / 3))))
        os.chdir('..')

    def update_model(self):
        while True:
            console.pm_stat("Model time: {}s".format(self.model_time))
            if self.model_time == self.max_time:
                self.visualize_results()
                self.model_time -= self.time_interval
            elif self.model_time == self.max_time - self.time_interval:
                self.diffusivity()
                self.visualize_results()
                self.space.to_csv('space_{}.csv'.format(self.model_time))
                self.model_time -= self.time_interval
            elif self.model_time <= self.max_time and self.model_time != 0:
                self.diffusivity()
                self.visualize_results()
                self.model_time -= self.time_interval
            elif self.model_time <= 0:
                self.visualize_results()
                console.pm_stat("Model time at 0!")
                self.space.to_csv('space.csv')
                self.write_animation()
                sys.exit(0)


class console:

    def __init__(clf):
        pass

    # print message that can be updated in the console
    @classmethod
    def pm_flush(clf, message):
        t = datetime.now()
        sys.stdout.write('\r[!] ({})      {}'.format(t, message))
        sys.stdout.flush()
        return None

    @classmethod
    def pm_stat(clf, message):
        t = datetime.now()
        print('\r[!] ({})      {}'.format(t, message))
        return None

    @classmethod
    def pm_err(clf, message):
        t = datetime.now()
        print('[X] ({})      {}'.format(t, message))
        return None

    @classmethod
    def pm_header(clf, message):
        print('{}'.format(message))
        return None
