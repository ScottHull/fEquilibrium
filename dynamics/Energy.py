from math import pi, sqrt
import pandas as pd
import numpy as np
import ast
import time


class thermal_eq:

    def __init__(self):
        pass


    @classmethod
    def explicit_nearest_neighboor(self, system_data, x_coord, y_coord, z_coord, space_resolution,
                          visualize_neighbors=False, animate_neighbors=False):
        neighbors = []
        # minimum = 0 # Uncomment this code if you'd like explicit distance calculations
        # potential_neighbors = {} # Uncomment this code if you'd like explicit distance calculations

        potential_xplus_neighbor = x_coord + space_resolution
        potential_yplus_neighbor = y_coord + space_resolution
        potential_zplus_neighbor = z_coord + space_resolution
        potential_xminus_neighbor = x_coord - space_resolution
        potential_yminus_neighbor = y_coord - space_resolution
        potential_zminus_neighbor = z_coord - space_resolution

        if min(system_data['x_coords']) <= potential_xplus_neighbor <= max(system_data['x_coords']):
            neighbors.append([potential_xplus_neighbor, y_coord, z_coord])
        if min(system_data['x_coords']) <= potential_xminus_neighbor <= max(system_data['x_coords']):
            neighbors.append([potential_xminus_neighbor, y_coord, z_coord])
        if min(system_data['y_coords']) <= potential_yplus_neighbor <= max(system_data['y_coords']):
            neighbors.append([x_coord, potential_yplus_neighbor, z_coord])
        if min(system_data['y_coords']) <= potential_yminus_neighbor <= max(system_data['y_coords']):
            neighbors.append([x_coord, potential_yminus_neighbor, z_coord])
        if min(system_data['z_coords']) <= potential_zplus_neighbor <= max(system_data['z_coords']):
            neighbors.append([x_coord, y_coord, potential_zplus_neighbor])
        if min(system_data['z_coords']) <= potential_zminus_neighbor <= max(system_data['z_coords']):
            neighbors.append([x_coord, y_coord, potential_zminus_neighbor])

        if visualize_neighbors == True:
            import matplotlib as mpl
            mpl.use('Qt5Agg')
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            import os, shutil
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
            if visualize_neighbors == True and animate_neighbors == False:
                fig.show()
            if animate_neighbors == True:
                fig.savefig(os.getcwd()+'/mpl_animation3/snap_{}-{}-{}.png'.format(x_coord, y_coord, z_coord), format='png')
            fig.clf()
        classified_neighbors = self.explicit_classify_neighbors(x_coord=x_coord, y_coord=y_coord, z_coord=z_coord,
                                neighbors=neighbors, space_resolution=space_resolution, system_data=system_data)
        # print("Classified neighbors: {}".format(classified_neighbors))
        return classified_neighbors


    @classmethod
    def explicit_classify_neighbors(cls, x_coord, y_coord, z_coord, system_data, neighbors, space_resolution):
        neighbors_dict = {'x': {'x+': {'coords': [], 'index': []}, 'x': {'coords': [], 'index': []},
                    'x-': {'coords': [], 'index': []}}, 'y': {'y+': {'coords': [], 'index': []},
                    'y': {'coords': [], 'index': []},'y-': {'coords': [], 'index': []}},
                    'z': {'z+': {'coords': [], 'index': []}, 'z': {'coords': [], 'index': []},
                    'z-': {'coords': [], 'index': []}}} # for each dict, x,y,z,index
        neighbors_dict['x']['x']['coords'].append(x_coord)
        neighbors_dict['x']['x']['coords'].append(y_coord)
        neighbors_dict['x']['x']['coords'].append(z_coord)
        neighbors_dict['y']['y']['coords'].append(x_coord)
        neighbors_dict['y']['y']['coords'].append(y_coord)
        neighbors_dict['y']['y']['coords'].append(z_coord)
        neighbors_dict['z']['z']['coords'].append(x_coord)
        neighbors_dict['z']['z']['coords'].append(y_coord)
        neighbors_dict['z']['z']['coords'].append(z_coord)
        for set in neighbors:
            if x_coord + space_resolution == set[0] and y_coord == set[1] and z_coord == set[2]:
                neighbors_dict['x']['x+']['coords'].append(x_coord+space_resolution)
                neighbors_dict['x']['x+']['coords'].append(y_coord)
                neighbors_dict['x']['x+']['coords'].append(z_coord)
            if x_coord - space_resolution == set[0] and y_coord == set[1] and z_coord == set[2]:
                neighbors_dict['x']['x-']['coords'].append(x_coord-space_resolution)
                neighbors_dict['x']['x-']['coords'].append(y_coord)
                neighbors_dict['x']['x-']['coords'].append(z_coord)
            if x_coord == set[0] and y_coord + space_resolution == set[1] and z_coord == set[2]:
                neighbors_dict['y']['y+']['coords'].append(x_coord)
                neighbors_dict['y']['y+']['coords'].append(y_coord+space_resolution)
                neighbors_dict['y']['y+']['coords'].append(z_coord)
            if x_coord == set[0] and y_coord - space_resolution == set[1] and z_coord == set[2]:
                neighbors_dict['y']['y-']['coords'].append(x_coord)
                neighbors_dict['y']['y-']['coords'].append(y_coord-space_resolution)
                neighbors_dict['y']['y-']['coords'].append(z_coord)
            if x_coord == set[0] and y_coord == set[1] and z_coord + space_resolution == set[2]:
                neighbors_dict['z']['z+']['coords'].append(x_coord)
                neighbors_dict['z']['z+']['coords'].append(y_coord)
                neighbors_dict['z']['z+']['coords'].append(z_coord+space_resolution)
            if x_coord == set[0] and y_coord == set[1] and z_coord - space_resolution == set[2]:
                neighbors_dict['z']['z-']['coords'].append(x_coord)
                neighbors_dict['z']['z-']['coords'].append(y_coord)
                neighbors_dict['z']['z-']['coords'].append(z_coord-space_resolution)

        index_count = 0
        for row in system_data.index:
            if index_count == 9:
                break
            else:
                for i in neighbors_dict:
                    for z in neighbors_dict[i]:
                        if len(neighbors_dict[i][z]['coords']) == 3:
                            if system_data['x_coords'][row] == neighbors_dict[i][z]['coords'][0] and \
                            system_data['y_coords'][row] == neighbors_dict[i][z]['coords'][1] and \
                            system_data['z_coords'][row] == neighbors_dict[i][z]['coords'][2]:
                                neighbors_dict[i][z]['index'].append(row)
                                index_count += 1
        return neighbors_dict
        
        

    @classmethod
    def classify_neighbors(cls, x_coord, y_coord, z_coord, system_data, neighbors, space_resolution, data_type):
        neighbors_dict = {'x': {'x+': {'coords': [], 'index': [], '{}'.format(data_type): []}, 'x': {'coords': [], 'index': [], '{}'.format(data_type): []},
                    'x-': {'coords': [], 'index': [], '{}'.format(data_type): []}}, 'y': {'y+': {'coords': [], 'index': [], '{}'.format(data_type): []},
                    'y': {'coords': [], 'index': [], '{}'.format(data_type): []},'y-': {'coords': [], 'index': [], '{}'.format(data_type): []}},
                    'z': {'z+': {'coords': [], 'index': [], '{}'.format(data_type): []}, 'z': {'coords': [], 'index': [], '{}'.format(data_type): []},
                    'z-': {'coords': [], 'index': [], '{}'.format(data_type): []}}} # for each dict, x,y,z,index
        neighbors_dict['x']['x']['coords'].append(x_coord)
        neighbors_dict['x']['x']['coords'].append(y_coord)
        neighbors_dict['x']['x']['coords'].append(z_coord)
        neighbors_dict['y']['y']['coords'].append(x_coord)
        neighbors_dict['y']['y']['coords'].append(y_coord)
        neighbors_dict['y']['y']['coords'].append(z_coord)
        neighbors_dict['z']['z']['coords'].append(x_coord)
        neighbors_dict['z']['z']['coords'].append(y_coord)
        neighbors_dict['z']['z']['coords'].append(z_coord)
        for set in neighbors:
            if x_coord + space_resolution == set[0] and y_coord == set[1] and z_coord == set[2]:
                neighbors_dict['x']['x+']['coords'].append(x_coord+space_resolution)
                neighbors_dict['x']['x+']['coords'].append(y_coord)
                neighbors_dict['x']['x+']['coords'].append(z_coord)
            if x_coord - space_resolution == set[0] and y_coord == set[1] and z_coord == set[2]:
                neighbors_dict['x']['x-']['coords'].append(x_coord-space_resolution)
                neighbors_dict['x']['x-']['coords'].append(y_coord)
                neighbors_dict['x']['x-']['coords'].append(z_coord)
            if x_coord == set[0] and y_coord + space_resolution == set[1] and z_coord == set[2]:
                neighbors_dict['y']['y+']['coords'].append(x_coord)
                neighbors_dict['y']['y+']['coords'].append(y_coord+space_resolution)
                neighbors_dict['y']['y+']['coords'].append(z_coord)
            if x_coord == set[0] and y_coord - space_resolution == set[1] and z_coord == set[2]:
                neighbors_dict['y']['y-']['coords'].append(x_coord)
                neighbors_dict['y']['y-']['coords'].append(y_coord-space_resolution)
                neighbors_dict['y']['y-']['coords'].append(z_coord)
            if x_coord == set[0] and y_coord == set[1] and z_coord + space_resolution == set[2]:
                neighbors_dict['z']['z+']['coords'].append(x_coord)
                neighbors_dict['z']['z+']['coords'].append(y_coord)
                neighbors_dict['z']['z+']['coords'].append(z_coord+space_resolution)
            if x_coord == set[0] and y_coord == set[1] and z_coord - space_resolution == set[2]:
                neighbors_dict['z']['z-']['coords'].append(x_coord)
                neighbors_dict['z']['z-']['coords'].append(y_coord)
                neighbors_dict['z']['z-']['coords'].append(z_coord-space_resolution)

        for i in neighbors_dict:
            for z in neighbors_dict[i]:
                if len(neighbors_dict[i][z]['coords']) == 3:
                    for row in system_data.index:
                        if system_data['x_coords'][row] == neighbors_dict[i][z]['coords'][0] and \
                        system_data['y_coords'][row] == neighbors_dict[i][z]['coords'][1] and system_data['z_coords'][row] == \
                        neighbors_dict[i][z]['coords'][2]:
                            neighbors_dict[i][z]['index'].append(row)
                            neighbors_dict[i][z]['{}'.format(data_type)].append(system_data['{}'.format(data_type)][row])

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
        dT = (thermal_diffusivity*sum(format_laplacian)) * deltaTime
        T = point_temperature + dT
        return T






    @classmethod
    def nearest_neighboor(self, system_data, x_coord, y_coord, z_coord, space_resolution, data_type,
                          visualize_neighbors=False, animate_neighbors=False):
        neighbors = []
        minimum = 0
        potential_neighbors = {}
        for row in system_data.index:
            if system_data['x_coords'][row] == x_coord and system_data['y_coords'][row] == y_coord and system_data['z_coords'][row] == z_coord:
                pass
            elif len(list(potential_neighbors.keys())) == 6:
                pass
            else:
                sample_xcoord = system_data['x_coords'][row]
                sample_ycoord = system_data['y_coords'][row]
                sample_zcoord = system_data['z_coords'][row]
                distance = round(sqrt(((sample_xcoord - x_coord)**2) + ((sample_ycoord - y_coord)**2) +
                                      ((sample_zcoord - z_coord)**2)), len(str(space_resolution)))
                if minimum == 0:
                    minimum = distance
                    potential_neighbors.update({system_data['object_id'][row]: distance})
                elif distance < minimum:
                    potential_neighbors.clear()
                    minimum = distance
                    potential_neighbors.update({system_data['object_id'][row]: distance})
                elif distance == minimum:
                    potential_neighbors.update({system_data['object_id'][row]: distance})
        for row in system_data.index:
            for i in potential_neighbors:
                if str(i) == str(system_data['object_id'][row]):
                    sample_neighbors = []
                    sample_neighbors.append(round(system_data['x_coords'][row], len(str(space_resolution))))
                    sample_neighbors.append(round(system_data['y_coords'][row], len(str(space_resolution))))
                    sample_neighbors.append(round(system_data['z_coords'][row], len(str(space_resolution))))
                    neighbors.append(sample_neighbors)

        if visualize_neighbors == True:
            import matplotlib as mpl
            mpl.use('Qt5Agg')
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            import os, shutil
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
            if visualize_neighbors == True and animate_neighbors == False:
                fig.show()
            if animate_neighbors == True:
                fig.savefig(os.getcwd()+'/mpl_animation3/snap_{}-{}-{}.png'.format(x_coord, y_coord, z_coord), format='png')
            fig.clf()
        classified_neighbors = self.classify_neighbors(x_coord=x_coord, y_coord=y_coord, z_coord=z_coord,
                                neighbors=neighbors, space_resolution=space_resolution, system_data=system_data, data_type=data_type)
        return classified_neighbors
        # return neighbors # for each coordinate set in neighbors, x at position 0, y at 1, z and 2


    def D3_thermal_eq(self, system_data, deltaTime, space_resolution):
        print("Thermally equilibrating system...")
        system_data['neighbors'] = np.NAN
        system_data['T_gradient'] = [[] for i in list(range(len(system_data['x_coords'])))]
        system_data['T_laplacian'] = [[] for i in list(range(len(system_data['x_coords'])))]
        system_data['dT/dt'] = np.NAN
        system_data['K'] = np.NAN
        new_thermal_df = system_data.copy(deep=True)
        frames = []
        material_properties = pd.read_csv("dynamics/physical_parameters.csv", index_col='Material')
        for row in system_data.index:
            sample_xcoord = system_data['x_coords'][row]
            sample_ycoord = system_data['y_coords'][row]
            sample_zcoord = system_data['z_coords'][row]
            print("Calculating temperature gradient for x:{} y:{} z:{}".format(sample_xcoord, sample_ycoord, sample_zcoord))
            # neighbors = self.nearest_neighboor(system_data=system_data, x_coord=sample_xcoord, y_coord=sample_ycoord,
            #                                    z_coord=sample_zcoord, space_resolution=space_resolution,
            #                                    visualize_neighbors=visualize_neighbors,
            #                                    animate_neighbors=animate_neighbors, data_type='temperature')
            # gradient = [] # x gradient at index 0, y at 1, z at 2
            # unsorted_gradient = self.gradient(classified_neighbors=neighbors)
            neighbors = ast.literal_eval(system_data['nearest_neighbors'][row])
            for i in neighbors:
                for z in neighbors[i]:
                    if len(neighbors[i][z]['coords']) != 0:
                        index = neighbors[i][z]['index']
                        temperature = [system_data['temperature'][index].item()]
                        neighbors[i][z].update({'temperature': temperature})
            gradient = self.gradient(classified_neighbors=neighbors)
            system_data['T_gradient'][row] = gradient
            # gradient.append([unsorted_gradient['grad_x'][0]])
            # gradient.append([unsorted_gradient['grad_y'][0]])
            # gradient.append([unsorted_gradient['grad_z'][0]])

        for row in system_data.index:
            sample_xcoord = system_data['x_coords'][row]
            sample_ycoord = system_data['y_coords'][row]
            sample_zcoord = system_data['z_coords'][row]
            print("Calculating temperature laplacian for x:{} y:{} z:{}".format(sample_xcoord, sample_ycoord,
                                                                               sample_zcoord))
            neighbors = ast.literal_eval(system_data['nearest_neighbors'][row])
            for i in neighbors:
                for z in neighbors[i]:
                    if len(neighbors[i][z]['coords']) != 0:
                        structured_grad = []
                        index = neighbors[i][z]['index']
                        grad = system_data['T_gradient'][index].item()
                        structured_grad.append(grad['grad_x'][0])
                        structured_grad.append(grad['grad_y'][0])
                        structured_grad.append(grad['grad_z'][0])
                        neighbors[i][z].update({'T_gradient': structured_grad})
            laplacian = self.laplacian(classified_neighbors=neighbors)
            system_data['T_laplacian'][row] = laplacian
            system_data['temperature'][row] = self.change_temperature(laplacian=laplacian,
                point_temperature=system_data['temperature'][row], deltaTime=deltaTime,
                                thermal_diffusivity=material_properties['Thermal Diffusivity'][system_data['object'][row]])
            
            
            
        #     if animate_neighbors == True:
        #         frames.append('snap_{}-{}-{}.png'.format(sample_xcoord, sample_ycoord, sample_zcoord))
        # if animate_neighbors == True:
        #     import moviepy.editor as mpy
        #     import os, time
        #     os.chdir(os.getcwd() + "/mpl_animation3")
        #     animation = mpy.ImageSequenceClip(frames,
        #                                       fps=5,
        #                                       load_images=True)
        #     animation.write_gif('neighbors.gif', fps=5)
        #     os.chdir("..")

        return new_thermal_df


class energy:

    def __init__(self):
        self.gravity = 9.81

    def stokes_frictional_energy(self, body_material, matrix_material, body_radius, body_mass, distance_travelled):
        """
        The Stokes settling velocity calculates a terminal velocity, so there is no acceleration.  The frictional force
        (drag force) must therefore balance the force due to gravity, Fg=Fd.
        :return:
        """
        df = pd.read_csv('dynamics/physical_parameters.csv', index_col='Material')
        body_cp = df['C_p'][body_material] # specific heat of the body
        body_density = df['Density'][body_material]
        matrix_density = df['Density'][matrix_material]
        F_d = (body_density - matrix_density) * self.gravity * (4/3) * pi * (body_radius**3) # the force generated by
        # frictional drag.  Must be converted to energy and added to the body temperature.
        # W = F * d, Units = [J]
        W = F_d * distance_travelled
        # convert joules to degK, Q[J]=M[g] * Cp * T[degK] --> T=Q/(M*Cp)
        degK = W / (body_cp * body_mass) # the temperature to be added to the body
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
        ke = 0.5 * mass * (velocity**2) # ke = 1/2*m*v^2
        return ke
    #
    def potential_energy(self, mass, height, box_height):
        pe = mass * self.gravity * abs((height - box_height)) # pe = m*g*h
        return pe
    #
    # def release_energy(self, former_pe, current_pe):
    #     friction_energy = abs(current_pe - former_pe) # assume that all energy release as frictional energy, or the
    #     #       difference in the potential energies at position 2 and position 1
    #     return friction_energy