from math import pi, sqrt
import os
import pandas as pd
import numpy as np
import ast
import time
from multiprocessing import Pool

os.sys.path.append(os.path.dirname(os.path.abspath('.')));
from meta.Console import console


class thermal_eq:
    def __init__(self):
        pass

    @classmethod
    def explicit_nearest_neighboor(self, system_data, x_coord, y_coord, z_coord, space_resolution, minx, maxx, miny,
                                   maxy,
                                   minz, maxz, visualize_neighbors, animate_neighbors):
        neighbors = []

        potential_xplus_neighbor = round(x_coord + space_resolution, len(str(space_resolution)))
        potential_yplus_neighbor = round(y_coord + space_resolution, len(str(space_resolution)))
        potential_zplus_neighbor = round(z_coord + space_resolution, len(str(space_resolution)))
        potential_xminus_neighbor = round(x_coord - space_resolution, len(str(space_resolution)))
        potential_yminus_neighbor = round(y_coord - space_resolution, len(str(space_resolution)))
        potential_zminus_neighbor = round(z_coord - space_resolution, len(str(space_resolution)))

        if minx <= potential_xplus_neighbor <= maxx:
            neighbors.append([potential_xplus_neighbor, y_coord, z_coord])
        elif potential_xplus_neighbor > maxx:
            neighbors.append([minx, y_coord, z_coord])
        if minx <= potential_xminus_neighbor <= maxx:
            neighbors.append([potential_xminus_neighbor, y_coord, z_coord])
        elif potential_xminus_neighbor < minx:
            neighbors.append([maxx, y_coord, z_coord])
        if miny <= potential_yplus_neighbor <= maxy:
            neighbors.append([x_coord, potential_yplus_neighbor, z_coord])
        elif potential_yplus_neighbor > maxy:
            neighbors.append([x_coord, miny, z_coord])
        if miny <= potential_yminus_neighbor <= maxy:
            neighbors.append([x_coord, potential_yminus_neighbor, z_coord])
        elif potential_yminus_neighbor < miny:
            neighbors.append([x_coord, maxy, z_coord])
        if minz <= potential_zplus_neighbor <= maxz:
            neighbors.append([x_coord, y_coord, potential_zplus_neighbor])
        elif potential_zplus_neighbor > maxz:
            neighbors.append([x_coord, y_coord, minz])
        if minz <= potential_zminus_neighbor <= maxz:
            neighbors.append([x_coord, y_coord, potential_zminus_neighbor])
        elif potential_zplus_neighbor < maxz:
            neighbors.append([x_coord, y_coord, maxz])

        # eliminate potential floating points:
        # temp_neighbors = []
        # for i in neighbors:
        #     temp = []
        #     for j in i: # access elements in sub-lists
        #         rounded_coord = round(j, len(str(space_resolution))) # round the coordinates to the spatial resolution
        #         temp.append(rounded_coord)
        #     temp_neighbors.append(temp)
        # neighbors.clear() # delete the old list with potential floating point contamination
        # for i in temp_neighbors:
        #     neighbors.append(i) # replace values in the list



        if visualize_neighbors is True:
            import matplotlib as mpl
            mpl.use('Qt5Agg')
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            import os
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlim(xmin=min(system_data['x_coords']), xmax=max(system_data['x_coords']))
            ax.set_ylim(ymin=min(system_data['y_coords']), ymax=max(system_data['y_coords']))
            ax.set_zlim(zmin=min(system_data['z_coords']), zmax=max(system_data['z_coords']))
            ax.scatter3D(x_coord, y_coord, z_coord, color='b')
            for i in neighbors:
                ax.scatter3D(i[0], i[1], i[2], color='r')
            ax.set_title("Nearest neighbors for x:{}, y:{}, z:{}".format(x_coord, y_coord, z_coord))
            ax.set_xlabel("Box Length")
            ax.set_ylabel("Box Width")
            ax.set_zlabel("Box Height")
            ax.invert_zaxis()
            if visualize_neighbors is True and animate_neighbors is False:
                fig.show()
            if animate_neighbors is True:
                fig.savefig(os.getcwd() + '/nearest_neighbors/snap_{}-{}-{}.png'.format(x_coord, y_coord, z_coord),
                            format='png')
            fig.clf()
        classified_neighbors = self.explicit_classify_neighbors(x_coord=x_coord, y_coord=y_coord, z_coord=z_coord,
                                                                neighbors=neighbors, space_resolution=space_resolution,
                                                                system_data=system_data,
                                                                minx=minx, maxx=maxx, miny=miny, maxy=maxy, minz=minz,
                                                                maxz=maxz)
        return classified_neighbors

    @classmethod
    def explicit_classify_neighbors(cls, x_coord, y_coord, z_coord, system_data, neighbors, space_resolution,
                                    minx, maxx, miny, maxy, minz, maxz):

        # this dictionary will be stored for reference in the dataframe for quick index access
        neighbors_dict = {'x': {'x+': {'coords': [], 'index': []}, 'x': {'coords': [], 'index': []},
                                'x-': {'coords': [], 'index': []}}, 'y': {'y+': {'coords': [], 'index': []},
                                                                          'y': {'coords': [], 'index': []},
                                                                          'y-': {'coords': [], 'index': []}},
                          'z': {'z+': {'coords': [], 'index': []}, 'z': {'coords': [], 'index': []},
                                'z-': {'coords': [], 'index': []}}}  # for each dict, x,y,z,index

        # round coordinates to avoid floating point conflictions
        x_coord = round(x_coord, len(str(space_resolution)))
        y_coord = round(y_coord, len(str(space_resolution)))
        z_coord = round(z_coord, len(str(space_resolution)))

        # add coordinates to the dictionary
        neighbors_dict['x']['x']['coords'].append(x_coord)
        neighbors_dict['x']['x']['coords'].append(y_coord)
        neighbors_dict['x']['x']['coords'].append(z_coord)
        neighbors_dict['y']['y']['coords'].append(x_coord)
        neighbors_dict['y']['y']['coords'].append(y_coord)
        neighbors_dict['y']['y']['coords'].append(z_coord)
        neighbors_dict['z']['z']['coords'].append(x_coord)
        neighbors_dict['z']['z']['coords'].append(y_coord)
        neighbors_dict['z']['z']['coords'].append(z_coord)

        # iterate over the neighbors to classify their direction, specifically for directional gradients
        for set in neighbors:
            if round(x_coord + space_resolution, len(str(space_resolution))) == round(set[0],
                                                                                      len(str(space_resolution))) and \
                            y_coord == round(set[1], len(str(space_resolution))) and \
                            z_coord == round(set[2], len(str(space_resolution))):
                neighbors_dict['x']['x+']['coords'].append(
                    round(x_coord + space_resolution, len(str(space_resolution))))
                neighbors_dict['x']['x+']['coords'].append(round(y_coord, len(str(space_resolution))))
                neighbors_dict['x']['x+']['coords'].append(round(z_coord, len(str(space_resolution))))
            if round(x_coord - space_resolution, len(str(space_resolution))) == round(set[0],
                                                                                      len(str(space_resolution))) and \
                            y_coord == round(set[1], len(str(space_resolution))) and \
                            z_coord == round(set[2], len(str(space_resolution))):
                neighbors_dict['x']['x-']['coords'].append(
                    round(x_coord - space_resolution, len(str(space_resolution))))
                neighbors_dict['x']['x-']['coords'].append(round(y_coord, len(str(space_resolution))))
                neighbors_dict['x']['x-']['coords'].append(round(z_coord, len(str(space_resolution))))
            if x_coord == round(set[0], len(str(space_resolution))) and round(y_coord + space_resolution,
                                                                              len(str(space_resolution))) == \
                    round(set[1], len(str(space_resolution))) and \
                            z_coord == round(set[2], len(str(space_resolution))):
                neighbors_dict['y']['y+']['coords'].append(round(x_coord, len(str(space_resolution))))
                neighbors_dict['y']['y+']['coords'].append(
                    round(y_coord + space_resolution, len(str(space_resolution))))
                neighbors_dict['y']['y+']['coords'].append(round(z_coord, len(str(space_resolution))))
            if x_coord == round(set[0], len(str(space_resolution))) and round(y_coord - space_resolution,
                                                                              len(str(space_resolution))) == \
                    round(set[1], len(str(space_resolution))) and \
                            z_coord == round(set[2], len(str(space_resolution))):
                neighbors_dict['y']['y-']['coords'].append(round(x_coord, len(str(space_resolution))))
                neighbors_dict['y']['y-']['coords'].append(
                    round(y_coord - space_resolution, len(str(space_resolution))))
                neighbors_dict['y']['y-']['coords'].append(round(z_coord, len(str(space_resolution))))
            if x_coord == round(set[0], len(str(space_resolution))) and y_coord == round(set[1],
                                                                                         len(str(space_resolution))) and \
                            round(z_coord + space_resolution, len(str(space_resolution))) == round(set[2], len(
                        str(space_resolution))):
                neighbors_dict['z']['z+']['coords'].append(round(x_coord, len(str(space_resolution))))
                neighbors_dict['z']['z+']['coords'].append(round(y_coord, len(str(space_resolution))))
                neighbors_dict['z']['z+']['coords'].append(
                    round(z_coord + space_resolution, len(str(space_resolution))))
            if x_coord == round(set[0], len(str(space_resolution))) and y_coord == round(set[1],
                                                                                         len(str(space_resolution))) and \
                            round(z_coord - space_resolution, len(str(space_resolution))) == round(set[2], len(
                        str(space_resolution))):
                neighbors_dict['z']['z-']['coords'].append(round(x_coord, len(str(space_resolution))))
                neighbors_dict['z']['z-']['coords'].append(round(y_coord, len(str(space_resolution))))
                neighbors_dict['z']['z-']['coords'].append(
                    round(z_coord - space_resolution, len(str(space_resolution))))




        # need to multi-index for quick neighbor data scooping
        # x, y, z coords will be index values. copy the original form quickly
        x_coords_copy = system_data['x_coords'].values.tolist()
        y_coords_copy = system_data['y_coords'].values.tolist()
        z_coords_copy = system_data['z_coords'].values.tolist()
        system_data['index'] = list(range(len(system_data['x_coords'])))
        system_data.set_index(['x_coords', 'y_coords', 'z_coords'], inplace=True)  # x, y, z coords now index values
        for i in neighbors_dict:
            for r in neighbors_dict[i]:
                if len(neighbors_dict[i][r]['coords']) == 3:
                    x = neighbors_dict[i][r]['coords'][0]  # neighbor x-coord
                    y = neighbors_dict[i][r]['coords'][1]  # neighbor y-coord
                    z = neighbors_dict[i][r]['coords'][2]  # neighbor z-coord
                    row = (system_data['index'].loc[[(x, y, z)]]).values[
                        0]  # get the original index value of the coords
                    neighbors_dict[i][r]['index'].append(row)  # append that value to the neighbors dictionary

        # resets indexing to original form
        system_data.set_index('index', inplace=True)
        system_data['x_coords'] = x_coords_copy
        system_data['y_coords'] = y_coords_copy
        system_data['z_coords'] = z_coords_copy

        return neighbors_dict

    @classmethod
    def gradient(cls, classified_neighbors):
        gradient = {'grad_x': [], 'grad_y': [], 'grad_z': []}
        for i in classified_neighbors:
            if i == 'x':
                if len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['temperature'][0] -
                            classified_neighbors[i]["{}-".format(i)]['temperature'][0]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][0] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][0])
                    gradient['grad_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) == 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}".format(i)]['temperature'][0] -
                            classified_neighbors[i]["{}-".format(i)]['temperature'][0]) / \
                           (classified_neighbors[i]["{}".format(i)]['coords'][0] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][0])
                    gradient['grad_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) == 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['temperature'][0] -
                            classified_neighbors[i]["{}".format(i)]['temperature'][0]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][0] -
                            classified_neighbors[i]["{}".format(i)]['coords'][0])
                    gradient['grad_{}'.format(i)].append(dT_d)
            elif i == 'y':
                if len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['temperature'][0] -
                            classified_neighbors[i]["{}-".format(i)]['temperature'][0]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][1] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][1])
                    gradient['grad_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) == 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}".format(i)]['temperature'][0] -
                            classified_neighbors[i]["{}-".format(i)]['temperature'][0]) / \
                           (classified_neighbors[i]["{}".format(i)]['coords'][1] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][1])
                    gradient['grad_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) == 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['temperature'][0] -
                            classified_neighbors[i]["{}".format(i)]['temperature'][0]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][1] -
                            classified_neighbors[i]["{}".format(i)]['coords'][1])
                    gradient['grad_{}'.format(i)].append(dT_d)
            elif i == 'z':
                if len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['temperature'][0] -
                            classified_neighbors[i]["{}-".format(i)]['temperature'][0]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][2] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][2])
                    gradient['grad_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) == 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}".format(i)]['temperature'][0] -
                            classified_neighbors[i]["{}-".format(i)]['temperature'][0]) / \
                           (classified_neighbors[i]["{}".format(i)]['coords'][2] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][2])
                    gradient['grad_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) == 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['temperature'][0] -
                            classified_neighbors[i]["{}".format(i)]['temperature'][0]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][2] -
                            classified_neighbors[i]["{}".format(i)]['coords'][2])
                    gradient['grad_{}'.format(i)].append(dT_d)
        return gradient

    @classmethod
    def laplacian(cls, classified_neighbors):
        laplacian = {'laplacian_x': [], 'laplacian_y': [], 'laplacian_z': []}
        for i in classified_neighbors:
            if i == 'x':
                if len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['T_gradient'][0] -
                            classified_neighbors[i]["{}-".format(i)]['T_gradient'][0]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][0] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][0])
                    laplacian['laplacian_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) == 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}".format(i)]['T_gradient'][0] -
                            classified_neighbors[i]["{}-".format(i)]['T_gradient'][0]) / \
                           (classified_neighbors[i]["{}".format(i)]['coords'][0] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][0])
                    laplacian['laplacian_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) == 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['T_gradient'][0] -
                            classified_neighbors[i]["{}".format(i)]['T_gradient'][0]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][0] -
                            classified_neighbors[i]["{}".format(i)]['coords'][0])
                    laplacian['laplacian_{}'.format(i)].append(dT_d)
            elif i == 'y':
                if len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['T_gradient'][1] -
                            classified_neighbors[i]["{}-".format(i)]['T_gradient'][1]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][1] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][1])
                    laplacian['laplacian_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) == 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}".format(i)]['T_gradient'][1] -
                            classified_neighbors[i]["{}-".format(i)]['T_gradient'][1]) / \
                           (classified_neighbors[i]["{}".format(i)]['coords'][1] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][1])
                    laplacian['laplacian_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) == 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['T_gradient'][1] -
                            classified_neighbors[i]["{}".format(i)]['T_gradient'][1]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][1] -
                            classified_neighbors[i]["{}".format(i)]['coords'][1])
                    laplacian['laplacian_{}'.format(i)].append(dT_d)
            elif i == 'z':
                if len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['T_gradient'][2] -
                            classified_neighbors[i]["{}-".format(i)]['T_gradient'][2]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][2] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][2])
                    laplacian['laplacian_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) == 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) != 0:
                    dT_d = (classified_neighbors[i]["{}".format(i)]['T_gradient'][2] -
                            classified_neighbors[i]["{}-".format(i)]['T_gradient'][2]) / \
                           (classified_neighbors[i]["{}".format(i)]['coords'][2] -
                            classified_neighbors[i]["{}-".format(i)]['coords'][2])
                    laplacian['laplacian_{}'.format(i)].append(dT_d)
                elif len(classified_neighbors[i]["{}+".format(i)]['coords']) != 0 and \
                                len(classified_neighbors[i]["{}-".format(i)]['coords']) == 0:
                    dT_d = (classified_neighbors[i]["{}+".format(i)]['T_gradient'][2] -
                            classified_neighbors[i]["{}".format(i)]['T_gradient'][2]) / \
                           (classified_neighbors[i]["{}+".format(i)]['coords'][2] -
                            classified_neighbors[i]["{}".format(i)]['coords'][2])
                    laplacian['laplacian_{}'.format(i)].append(dT_d)
        return laplacian

    @classmethod
    def change_temperature(cls, laplacian, point_temperature, deltaTime, thermal_diffusivity):
        # heat equation (time-dependent): dT/dt = K*laplacian(T)
        # dT = (K*K*laplacian(T)) * dt
        # T = T_not + dT
        format_laplacian = []
        for i in laplacian:
            format_laplacian.append(laplacian[i][0])
        dT = (thermal_diffusivity * sum(format_laplacian)) * deltaTime
        T = point_temperature + dT
        return T

    def D3_thermal_eq(self, system_data, deltaTime, space_resolution):
        """
        The main class for handling thermal equilibrium in 3D.

        :param system_data:
        :param deltaTime:
        :param space_resolution:
        :return: new_thermal_df, a new dataframe with updated temperatures to replace self.space
        """
        console.pm_stat("Thermally equilibrating system...")
        system_data['neighbors'] = np.NAN
        system_data['T_gradient'] = [[] for i in list(range(len(system_data['x_coords'])))]
        system_data['T_laplacian'] = [[] for i in list(range(len(system_data['x_coords'])))]
        system_data['dT/dt'] = np.NAN
        system_data['K'] = np.NAN
        new_thermal_df = system_data.copy(deep=True)  # copy the dataframe so that new values can be written
        material_properties = pd.read_csv("dynamics/physical_parameters.csv",
                                          index_col='Material')  # load in material properties from spreadsheet
        # calculate & store temperature gradients
        for row in system_data.itertuples():
            index = row.Index
            sample_xcoord = round(system_data['x_coords'][index], len(str(space_resolution)))
            sample_ycoord = round(system_data['y_coords'][index], len(str(space_resolution)))
            sample_zcoord = round(system_data['z_coords'][index], len(str(space_resolution)))
            console.pm_flush("Calculating temperature gradient for x:{} y:{} z:{}".format(sample_xcoord, sample_ycoord,
                                                                                          sample_zcoord))
            neighbors = ast.literal_eval(
                system_data['nearest_neighbors'][index])  # interpret the dictionary stored in the dataframe
            for i in neighbors:
                for z in neighbors[i]:
                    if len(neighbors[i][z]['coords']) != 0:
                        ind = neighbors[i][z]['index']
                        temperature = [system_data['temperature'][ind].item()]
                        neighbors[i][z].update({'temperature': temperature})
            gradient = self.gradient(classified_neighbors=neighbors)
            system_data['T_gradient'][index] = gradient
        print("")
        # calculate & store temperature laplacians
        for row in system_data.itertuples():
            index = row.Index
            sample_xcoord = system_data['x_coords'][index]
            sample_ycoord = system_data['y_coords'][index]
            sample_zcoord = system_data['z_coords'][index]
            console.pm_flush("Calculating temperature laplacian for x:{} y:{} z:{}".format(sample_xcoord, sample_ycoord,
                                                                                           sample_zcoord))
            neighbors = ast.literal_eval(system_data['nearest_neighbors'][index])
            for i in neighbors:
                for z in neighbors[i]:
                    if len(neighbors[i][z]['coords']) != 0:
                        structured_grad = []
                        ind = neighbors[i][z]['index']
                        grad = system_data['T_gradient'][ind].item()
                        structured_grad.append(grad['grad_x'][0])
                        structured_grad.append(grad['grad_y'][0])
                        structured_grad.append(grad['grad_z'][0])
                        neighbors[i][z].update({'T_gradient': structured_grad})
            laplacian = self.laplacian(classified_neighbors=neighbors)
            system_data['T_laplacian'][index] = laplacian
            system_data['temperature'][index] = self.change_temperature(laplacian=laplacian,
                                                                        point_temperature=system_data['temperature'][
                                                                            index], deltaTime=deltaTime,
                                                                        thermal_diffusivity=
                                                                        material_properties['Thermal Diffusivity'][
                                                                            system_data['object'][index]])
        print("")

        #     if animate_neighbors == True:
        #         frames.append('snap_{}-{}-{}.png'.format(sample_xcoord, sample_ycoord, sample_zcoord))
        # if animate_neighbors == True:
        #     import moviepy.editor as mpy
        #     import os, time
        #     os.chdir(os.getcwd() + "/nearest_neighbors")
        #     animation = mpy.ImageSequenceClip(frames,
        #                                       fps=5,
        #                                       load_images=True)
        #     animation.write_gif('neighbors.gif', fps=5)
        #     os.chdir("..")

        return new_thermal_df


class energy:

    def __init__(self):
        self.gravity = 9.81

    def stokes_frictional_energy(self, object, object_velocity, matrix_material, body_radius, body_mass,
                                 distance_travelled):
        """
        The Stokes settling velocity calculates a terminal velocity, so there is no acceleration.  The frictional force
        (drag force) must therefore balance the force due to gravity, Fg=Fd.  All work (units=J) assumed to be
        converted to heat (units=degK).

        Temperature = degK
        Body Mass = g (automatically scaled from the user input in kg)
        Specific Heat, cp = J/g
        Distance Travelled = m

        Modeling temperature change: K = W / (cp * mass) = J / [(((J / K) / mass) * mass)] = JK / J = K

        :return:
        """
        df = pd.read_csv('dynamics/physical_parameters.csv', index_col='Material')
        body_cp = df['C_p'][object]  # specific heat of the body
        body_density = df['Density'][object]
        matrix_density = df['Density'][matrix_material]
        drag_coeffient = df['Drag Coefficient'][object]
        F_d = drag_coeffient * 0.5 * matrix_density * (object_velocity ** 2) * (
        pi * (body_radius ** 2))  # F_d = drag (frictional) force, F_d = Cd*0.5*rho*velocity*A (source: NASA)
        # Frictional drag must be converted to energy and added to the body temperature.
        # W = F * d, Units = [J]
        W = F_d * distance_travelled  # convert joules to degK, Q[J]=M[g] * Cp * T[degK] --> T=Q/(M*Cp)
        degK = W / (body_cp * (body_mass * 1000))  # the temperature to be added to the body
        # Checking units for calculation above: K = W / (cp * mass) = J / [(((J / K) / mass) * mass)] = JK / J = K
        return degK

    # def distribute_stokes_frictional_energy(self, frictional_energy, path_coords):
    #     """
    #     Frictional energy shall be distributed along the path of movement evenly.
    #     :param frictional_energy:
    #     :param path_coords:
    #     :return: energy_unit, energy per grid unit
    #     """
    #     path_length = len(path_coords)
    #     energy_unit = frictional_energy / path_length
    #     return energy_unit

    def kinetic_energy(self, mass, velocity):
        ke = 0.5 * mass * (velocity ** 2)  # ke = 1/2*m*v^2
        return ke

    #
    def potential_energy(self, mass, height, box_height):
        pe = mass * self.gravity * abs((height - box_height))  # pe = m*g*h
        return pe
        #
        # def release_energy(self, former_pe, current_pe):
        #     friction_energy = abs(current_pe - former_pe) # assume that all energy release as frictional energy, or the
        #     #       difference in the potential energies at position 2 and position 1
        #     return friction_energy