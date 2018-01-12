import os
import sys
import pandas as pd
import matplotlib
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from math import pi
import numpy as np
from datetime import datetime
from decimal import getcontext, Decimal



class sphere:

    def __init__(self, max_r, max_theta, max_phi, radial_mesh_res, angular_mesh_res, diapir_r):
        self.prec = getcontext().prec = len(str(radial_mesh_res)) - 1
        self.max_r = Decimal(max_r)
        self.max_theta = Decimal(max_theta)
        self.max_phi = Decimal(max_phi)
        self.radial_mesh_res = Decimal(radial_mesh_res)
        self.angular_mesh_res = Decimal(angular_mesh_res)
        self.diapir_r = Decimal(diapir_r)
        self.diapir_theta = Decimal(Decimal(2) * Decimal(pi))
        self.diapir_phi = Decimal(Decimal(2) * Decimal(pi))
        self.coords = self.generate_coords()

        self.space = pd.DataFrame({
            'r_coords': [i[0] for i in self.coords],
            'theta_coords': [i[1] for i in self.coords],
            'phi_coords': [i[2] for i in self.coords],
            'coord_index': [str(i) for i in self.coords],
            'neighbors': np.NAN,
            'concentration': np.NAN,
        })

        self.neighbors = self.generate_neighbors()



    def generate_coords(self):
        console.pm_stat("Generating coordinates...")
        r_range = np.arange(0, self.max_r + self.radial_mesh_res, self.radial_mesh_res)
        theta_range = np.arange(0, self.max_theta + self.angular_mesh_res, self.angular_mesh_res)
        phi_range = np.arange(0, self.max_phi + self.angular_mesh_res, self.angular_mesh_res)

        coords = []
        for r in r_range:
            for t in theta_range:
                for p in phi_range:
                    temp_coords = []
                    temp_coords.append(Decimal(r))
                    temp_coords.append(Decimal(t))
                    temp_coords.append(Decimal(p))
                    coords.append(temp_coords)
        console.pm_stat("Coordinates generated!")

        return coords

    def generate_neighbors(self):
        curr_coord = 1
        total_coords = len(self.space['coord_index'].tolist())
        for row in self.space.itertuples():
            index = row.Index
            console.pm_flush("Finding neighbors for {}/{}".format(curr_coord, total_coords))
            curr_coord += 1

            curr_r = self.space['r_coords'][index]
            curr_theta = self.space['theta_coords'][index]
            curr_phi = self.space['phi_coords'][index]
            neighbors = {
                'r': {
                    'r+': {
                        'index': []
                    },
                    'r-': {
                        'index': []
                    }
                },
                'theta': {
                    'theta+': {
                        'index': []
                    },
                    'theta-': {
                        'index': []
                    }
                },
                'phi': {
                    'phi+': {
                        'index': []
                    },
                    'phi-': {
                        'index': []
                    }
                }
            }

            if Decimal(curr_r) + Decimal(self.radial_mesh_res) > self.max_r:
                r_plus = [Decimal(self.max_r), curr_theta + Decimal(pi), curr_phi + Decimal(pi)]






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
