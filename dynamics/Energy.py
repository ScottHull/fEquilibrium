from math import pi, sqrt
import pandas as pd
import numpy as np


class thermal_eq:

    def __init__(self):
        pass


    @classmethod
    def gradient(cls, system_data, neighbors, x_coord, y_coord, z_coord):
        gradient = [] # d/dx at index 0, d/dy at 1, d/dz at 2
        neighbor_temps_x_minus1 = ''
        neighbor_temps_x_plus1 = ''
        neighbor_temps_y_minus1 = ''
        neighbor_temps_y_plus1 = ''
        neighbor_temps_z_minus1 = ''
        neighbor_temps_z_plus1 = ''
        delT_delx = 0
        delT_dely = 0
        delT_delz = 0
        for set in neighbors:
            # print("At coord: x:{} y:{} z:{}".format(x_coord, y_coord, z_coord))
            # print("Working on set: {}".format(set))
            for row in system_data.index:
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (x_coord + 1) == set[0]:
                    # print("Identified a x+1!")
                    neighbor_temps_x_plus1 = system_data['temperature'][row]
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (x_coord - 1) == set[0]:
                    # print("Identified a x-1!")
                    neighbor_temps_x_minus1 = system_data['temperature'][row]
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (y_coord + 1) == set[1]:
                    # print("Identified a y+1!")
                    neighbor_temps_y_plus1 = system_data['temperature'][row]
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (y_coord - 1) == set[1]:
                    neighbor_temps_y_minus1 = system_data['temperature'][row]
                    # print("Identified a y-1!")
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (z_coord + 1) == set[2]:
                    # print("Identified a z+1!")
                    neighbor_temps_z_plus1 = system_data['temperature'][row]
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (z_coord - 1) == set[2]:
                    # print("Identified a z-1!")
                    neighbor_temps_z_minus1 = system_data['temperature'][row]
        if isinstance(neighbor_temps_x_plus1, str):
            for row in system_data.index:
                if system_data['x_coords'][row] == x_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_temps_x_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_temps_x_minus1, str):
            for row in system_data.index:
                if system_data['x_coords'][row] == x_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_temps_x_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_temps_y_plus1, str):
            for row in system_data.index:
                if system_data['y_coords'][row] == y_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_temps_y_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_temps_y_minus1, str):
            for row in system_data.index:
                if system_data['y_coords'][row] == y_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_temps_y_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_temps_z_plus1, str):
            for row in system_data.index:
                if system_data['z_coords'][row] == z_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_temps_z_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_temps_z_minus1, str):
            for row in system_data.index:
                if system_data['z_coords'][row] == z_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_temps_z_plus1 = system_data['temperature'][row]
        if not isinstance(neighbor_temps_x_minus1, str) and not isinstance(neighbor_temps_x_plus1, str):
            # print("x+1: {}K, x-1: {}K (x:{})".format(neighbor_temps_x_plus1, neighbor_temps_x_minus1, x_coord))
            delT_delx = (neighbor_temps_x_plus1 - neighbor_temps_x_minus1) / ((x_coord + 1) - (x_coord - 1))
        if not isinstance(neighbor_temps_y_minus1, str) and not isinstance(neighbor_temps_y_plus1, str):
            # print("y+1: {}K, y-1: {}K (y:{})".format(neighbor_temps_y_plus1, neighbor_temps_y_minus1, y_coord))
            delT_dely = (neighbor_temps_y_plus1 - neighbor_temps_y_minus1) / ((y_coord + 1) - (y_coord - 1))
        if not isinstance(neighbor_temps_z_minus1, str) and not isinstance(neighbor_temps_z_plus1, str):
            # print("z+1: {}K, z-1: {}K (z:{})".format(neighbor_temps_z_plus1, neighbor_temps_z_minus1, z_coord))
            delT_delz = (neighbor_temps_z_plus1 - neighbor_temps_z_minus1) / ((z_coord + 1) - (z_coord - 1))
        gradient.append(delT_delx)
        gradient.append(delT_dely)
        gradient.append(delT_delz)
        # print("x+1:{} x-1:{} y+1:{} y-1:{} z+1:{} z-1:{}".format(neighbor_temps_x_plus1, neighbor_temps_x_minus1,
        #             neighbor_temps_y_plus1, neighbor_temps_y_minus1, neighbor_temps_z_plus1, neighbor_temps_z_minus1))
        # print("Gradient: {}".format(gradient))
        return gradient

    @classmethod
    def laplace(cls, system_data, neighbors, x_coord, y_coord, z_coord):
        laplace = [] # d2/dx2 at index 0, d2/dy2 at 1, d2/dz2 at 2
        neighbor_gradients_x_minus1 = ''
        neighbor_gradients_x_plus1 = ''
        neighbor_gradients_y_minus1 = ''
        neighbor_gradients_y_plus1 = ''
        neighbor_gradients_z_minus1 = ''
        neighbor_gradients_z_plus1 = ''
        del2T_delx2 = 0
        del2T_dely2 = 0
        del2T_delz2 = 0
        for set in neighbors:
            for row in system_data.index:
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (x_coord + 1) == set[0]:
                    neighbor_gradients_x_plus1 = system_data['T_gradient'][row]
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (x_coord - 1) == set[0]:
                    neighbor_gradients_x_minus1 = system_data['T_gradient'][row]
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (y_coord + 1) == set[1]:
                    neighbor_gradients_y_plus1 = system_data['T_gradient'][row]
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (y_coord - 1) == set[1]:
                    neighbor_gradients_y_minus1 = system_data['T_gradient'][row]
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (z_coord + 1) == set[2]:
                    neighbor_gradients_z_plus1 = system_data['T_gradient'][row]
                if system_data['x_coords'][row] == set[0] and system_data['y_coords'][row] == set[1] and \
                                system_data['z_coords'][row] == set[2] and (z_coord - 1) == set[2]:
                    neighbor_gradients_z_minus1 = system_data['T_gradient'][row]
        if isinstance(neighbor_gradients_x_plus1, str):
            for row in system_data.index:
                if system_data['x_coords'][row] == x_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_gradients_x_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_gradients_x_minus1, str):
            for row in system_data.index:
                if system_data['x_coords'][row] == x_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_gradients_x_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_gradients_y_plus1, str):
            for row in system_data.index:
                if system_data['y_coords'][row] == y_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_gradients_y_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_gradients_y_minus1, str):
            for row in system_data.index:
                if system_data['y_coords'][row] == y_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_gradients_y_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_gradients_z_plus1, str):
            for row in system_data.index:
                if system_data['z_coords'][row] == z_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_gradients_z_plus1 = system_data['temperature'][row]
        if isinstance(neighbor_gradients_z_minus1, str):
            for row in system_data.index:
                if system_data['z_coords'][row] == z_coord and system_data['y_coords'][row] == y_coord and \
                                system_data['z_coords'][row] == z_coord:
                    neighbor_gradients_z_plus1 = system_data['temperature'][row]
        if not isinstance(neighbor_gradients_x_minus1, str) and not isinstance(neighbor_gradients_x_plus1, str):
            # print("x+1: {}K, x-1: {}K (x:{})".format(neighbor_gradients_x_plus1, neighbor_gradients_x_minus1, x_coord))
            del2T_delx2 = (neighbor_gradients_x_plus1 - neighbor_gradients_x_minus1) / ((x_coord + 1) - (x_coord - 1))
        if not isinstance(neighbor_gradients_y_minus1, str) and not isinstance(neighbor_gradients_y_plus1, str):
            # print("y+1: {}K, y-1: {}K (y:{})".format(neighbor_gradients_y_plus1, neighbor_gradients_y_minus1, y_coord))
            del2T_dely2 = (neighbor_gradients_y_plus1 - neighbor_gradients_y_minus1) / ((y_coord + 1) - (y_coord - 1))
        if not isinstance(neighbor_gradients_z_minus1, str) and not isinstance(neighbor_gradients_z_plus1, str):
            # print("z+1: {}K, z-1: {}K (z:{})".format(neighbor_gradients_z_plus1, neighbor_gradients_z_minus1, z_coord))
            del2T_delz2 = (neighbor_gradients_z_plus1 - neighbor_gradients_z_minus1) / ((z_coord + 1) - (z_coord - 1))
        laplace.append(del2T_delx2)
        laplace.append(del2T_dely2)
        laplace.append(del2T_delz2)
        # print("x+1:{} x-1:{} y+1:{} y-1:{} z+1:{} z-1:{}".format(neighbor_gradients_x_plus1, neighbor_gradients_x_minus1,
        #             neighbor_gradients_y_plus1, neighbor_gradients_y_minus1, neighbor_gradients_z_plus1, neighbor_gradients_z_minus1))
        # print("laplace: {}".format(laplace))
        return laplace
                
            



    @classmethod
    def nearest_neighboor(self, system_data, x_coord, y_coord, z_coord, visualize_neighbors=False, animate_neighbors=False):
        neighbors = []
        minimum = 0
        for row in system_data.index:
            if system_data['x_coords'][row] == x_coord and system_data['y_coords'][row] == y_coord and system_data['z_coords'][row] == z_coord:
                pass
            else:
                sample_xcoord = system_data['x_coords'][row]
                sample_ycoord = system_data['y_coords'][row]
                sample_zcoord = system_data['z_coords'][row]
                distance = sqrt(((sample_xcoord - x_coord)**2) + ((sample_ycoord - y_coord)**2) + ((sample_zcoord - z_coord)**2))
                if minimum == 0:
                    minimum = distance
                elif distance < minimum:
                    minimum = distance

        for row in system_data.index:
            if system_data['x_coords'][row] == x_coord and system_data['y_coords'][row] == y_coord and \
                            system_data['z_coords'][row] == z_coord:
                pass
            else:
                sample_xcoord = system_data['x_coords'][row]
                sample_ycoord = system_data['y_coords'][row]
                sample_zcoord = system_data['z_coords'][row]
                distance = sqrt(((sample_xcoord - x_coord) ** 2) + ((sample_ycoord - y_coord) ** 2) + (
                (sample_zcoord - z_coord) ** 2))
                if distance == minimum:
                    sample_neighbors = []
                    sample_neighbors.append(sample_xcoord)
                    sample_neighbors.append(sample_ycoord)
                    sample_neighbors.append(sample_zcoord)
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
        return neighbors # for each coordinate set in neighbors, x at position 0, y at 1, z and 2


    def D3_thermal_eq(self, system_data, animate_neighbors=False):
        system_data['neighbors'] = [[] for i in list(range(len(system_data['x_coords'])))]
        system_data['T_gradient'] = np.nan
        system_data['T_laplace'] = np.nan
        new_thermal_df = system_data.copy(deep=True)
        if animate_neighbors == True:
            import os, shutil
            if os.path.exists('mpl_animation3'):
                shutil.rmtree('mpl_animation3')
            os.mkdir('mpl_animation3')
        frames = []
        material_properts = pd.read_csv("physical_parameters.csv", index_col='Material')
        for row in system_data.index:
            sample_xcoord = system_data['x_coords'][row]
            sample_ycoord = system_data['y_coords'][row]
            sample_zcoord = system_data['z_coords'][row]
            neighbors = self.nearest_neighboor(system_data=system_data, x_coord=sample_xcoord, y_coord=sample_ycoord,
                                               z_coord=sample_zcoord, animate_neighbors=animate_neighbors)
            system_data['neighbors'][row] = neighbors
            system_data['T_gradient'][row] = sum(self.gradient(system_data=system_data, neighbors=neighbors,
                                                x_coord=sample_xcoord, y_coord=sample_ycoord, z_coord=sample_zcoord))
        for row in system_data.index:
            sample_xcoord = system_data['x_coords'][row]
            sample_ycoord = system_data['y_coords'][row]
            sample_zcoord = system_data['z_coords'][row]
            neighbors = self.nearest_neighboor(system_data=system_data, x_coord=sample_xcoord,
                                               y_coord=sample_ycoord,
                                               z_coord=sample_zcoord, animate_neighbors=animate_neighbors)
            system_data['neighbors'][row] = neighbors
            system_data['T_laplace'][row] = sum(self.laplace(system_data=system_data, neighbors=neighbors,
                                            x_coord=sample_xcoord, y_coord=sample_ycoord, z_coord=sample_zcoord))
            K = material_properts["Thermal Diffusivity"][system_data['object'][row]]
            dT_dt = K * system_data['T_laplace'][row]

            new_thermal_df["temperature"][row] = new_thermal_df["temperature"][row] + dT_dt
            new_thermal_df['neighbors'][row] = system_data['neighbors'][row]
            new_thermal_df['T_gradient'][row] = system_data['T_gradient'][row]
            new_thermal_df['T_laplace'][row] = system_data['T_laplace'][row]

            if animate_neighbors == True:
                frames.append('snap_{}-{}-{}.png'.format(sample_xcoord, sample_ycoord, sample_zcoord))
        if animate_neighbors == True:
            import moviepy.editor as mpy
            import os, time
            os.chdir(os.getcwd() + "/mpl_animation3")
            animation = mpy.ImageSequenceClip(frames,
                                              fps=5,
                                              load_images=True)
            animation.write_gif('neighbors.gif', fps=5)
            os.chdir("..")
            time.sleep(20)
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
        df = pd.read_csv('physical_parameters.csv', index_col='Material')
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
    def potential_energy(self, mass, height):
        pe = mass * self.gravity * height # pe = m*g*h
        return pe
    #
    # def release_energy(self, former_pe, current_pe):
    #     friction_energy = abs(current_pe - former_pe) # assume that all energy release as frictional energy, or the
    #     #       difference in the potential energies at position 2 and position 1
    #     return friction_energy