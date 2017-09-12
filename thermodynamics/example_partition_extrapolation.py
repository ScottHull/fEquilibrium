# import matplotlib as mpl
# mpl.use("Qt4Agg")
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import pandas as pd
# import sys, os
# import scipy.linalg as linalg
# os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from stats.regression import ls_regression, mult_lin_regression
# from mayavi import mlab
#
#
#
# w_df = pd.read_csv('w_partitioning_play.csv')
# temperature = w_df['Temperature']
# pressure = w_df['Pressure']
# fO2 = w_df['Corrected_fO2']
# D = w_df['Corrected_logD']
#
# w_df_isoT = pd.read_csv('w_partitioning_isoT.csv')
# w_df_isoP = pd.read_csv('w_partitioning_isoP.csv')
# w_df_isofO2 = pd.read_csv('w_partitioning_isofO2.csv')
#
# isoT_pressure = w_df_isoT['Pressure']
# isoT_fO2 = w_df_isofO2['Corrected_fO2']
# isoT_D = w_df_isoT['Corrected_logD']
# isoP_T = w_df_isoP['Temperature']
# isoP_fO2 = w_df_isoP['Corrected_fO2']
# isoP_D = w_df_isoP['Corrected_logD']
# isofO2_T = w_df_isofO2['Temperature']
# isofO2_P = w_df_isofO2['Pressure']
# isofO2_D = w_df_isofO2['Corrected_logD']
#
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # x = [1,2,3,4,5]
# # y = [2,4,5,4,5]
# # ax.scatter(x,y)
# # thing = ls_regression(x=x, y=y)
# # sloper, intercepter, = thing.lin_ls_regression()
# # x_min1, x_max1 = ax.get_xlim()
# # y_min1, y_max1 = intercepter, intercepter + sloper*(x_max1)
# # ax.plot([0, x_max1], [y_min1, y_max1])
# # ax.axvline(np.mean(x), c='g')
# # ax.axhline(np.mean(y), c='g')
# #
# # plt.show()
# # plt.close()
#
#
#
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(311)
# ax1.set_title("logD(Temperature)")
# ax1.set_xlabel("Temperature")
# ax1.set_ylabel("logD")
# ax1.scatter(temperature, D)
# ax1_regress = ls_regression(x=temperature, y=D)
# ax1_slope, ax1_intercept = ax1_regress.lin_ls_regression()
# x_min1, x_max1 = ax1.get_xlim()
# y_min1, y_max1 = ax1_intercept, ax1_intercept + ax1_slope*(x_max1)
# ax1.plot([0, x_max1], [y_min1, y_max1])
# ax1.axvline(np.mean(temperature), c='g')
# ax1.axhline(np.mean(D), c='g')
#
# ax2 = fig1.add_subplot(312)
# ax2.set_title("logD(Pressure)")
# ax2.set_xlabel("Pressure")
# ax2.set_ylabel("logD")
# ax2.scatter(pressure, D)
# ax2_regress = ls_regression(x=pressure, y=D)
# ax2_slope, ax2_intercept = ax2_regress.lin_ls_regression()
# x_min2, x_max2 = ax2.get_xlim()
# y_min2, y_max2 = ax2_intercept, ax2_intercept + ax2_slope*(x_max2)
# ax2.plot([0, x_max2], [y_min2, y_max2])
# ax2.axvline(np.mean(pressure), c='g')
# ax2.axhline(np.mean(D), c='g')
#
# ax3 = fig1.add_subplot(313)
# ax3.set_title("logD(fO2)")
# ax3.set_xlabel("fO2 (delta IW)")
# ax3.set_ylabel("logD")
# ax3.scatter(fO2, D)
# ax3_regress = ls_regression(x=fO2, y=D)
# ax3_slope, ax3_intercept = ax3_regress.lin_ls_regression()
# x_min3, x_max3 = ax3.get_xlim()[::-1]
# y_min3, y_max3 = ax3_intercept, ax3_intercept + ax3_slope*(x_max3)
# ax3.plot([0, x_max3], [y_min3, y_max3])
# ax3.axvline(np.mean(fO2), c='g')
# ax3.axhline(np.mean(D), c='g')
#
# plt.show()
# plt.close()
#
#
# fig2 = plt.figure()
# ax4 = fig2.add_subplot(111, projection='3d')
# ax4.set_title("logD(Pressure, fO2)")
# ax4.set_xlabel('Pressure (GPa)')
# ax4.set_ylabel("fO2 (IW)")
# ax4.set_zlabel('logD')
# ax4.set_zlim(zmin=0, zmax=5)
# x = pressure
# y = fO2
# z = D
# ax4.scatter3D(x, y, z, c='r')
# ax4.plot([np.mean(pressure)] * len(pressure), [np.mean(fO2)] * len(fO2), D, c='g')
# ax4.plot(pressure, [np.mean(fO2)] * len(fO2), [np.mean(D)] * len(D), c='g')
# ax4.plot([np.mean(pressure)] * len(pressure), fO2, [np.mean(D)] * len(D), c='g')
# data = np.c_[x, y, z]
# A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
# x_min, x_max = ax4.get_xlim()
# y_min, y_max = ax4.get_ylim()
# X, Y = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
# XX = X.flatten()
# YY = Y.flatten()
# C,_,_,_ = linalg.lstsq(A, data[:,2])
# Z = C[0]*X + C[1]*Y + C[2]
# ax4.plot_surface(X, Y, X, alpha=0.6, vmax=5)
#
# plt.show()
# plt.close()
#
#
#
#
# fig3 = plt.figure()
# ax5 = fig3.add_subplot(111, projection='3d')
# ax5.set_title("logD(Temperature, fO2)")
# ax5.set_xlabel('Temperature (degC)')
# ax5.set_ylabel("fO2 (IW)")
# ax5.set_zlabel('logD')
# ax5.set_zlim(zmin=0, zmax=5)
# x = temperature
# y = fO2
# z = D
# ax5.scatter3D(x, y, z, c='r')
# ax5.plot([np.mean(temperature)] * len(temperature), [np.mean(fO2)] * len(fO2), D, c='g')
# ax5.plot(temperature, [np.mean(fO2)] * len(fO2), [np.mean(D)] * len(D), c='g')
# ax5.plot([np.mean(temperature)] * len(temperature), fO2, [np.mean(D)] * len(D), c='g')
# data = np.c_[x, y, z]
# A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
# x_min, x_max = ax5.get_xlim()
# y_min, y_max = ax5.get_ylim()
# X, Y = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
# XX = X.flatten()
# YY = Y.flatten()
# C,_,_,_ = linalg.lstsq(A, data[:,2])
# Z = C[0]*X + C[1]*Y + C[2]
# ax5.plot_surface(X, Y, X, alpha=0.6)
#
# plt.show()
# plt.close()
#
#
#
#
# fig4 = plt.figure()
# ax6 = fig4.add_subplot(111, projection='3d')
# ax6.set_title("logD(Temperature, Pressure)")
# ax6.set_xlabel('Temperature (degC)')
# ax6.set_ylabel("Pressure (GPa)")
# ax6.set_zlabel('logD')
# ax6.set_zlim(zmin=0, zmax=5)
# x = temperature
# y = pressure
# z = D
# ax6.scatter3D(x, y, z, c='r')
# ax6.plot([np.mean(temperature)] * len(temperature), [np.mean(pressure)] * len(pressure), D, c='g')
# ax6.plot(temperature, [np.mean(pressure)] * len(pressure), [np.mean(D)] * len(D), c='g')
# ax6.plot([np.mean(temperature)] * len(temperature), pressure, [np.mean(D)] * len(D), c='g')
# data = np.c_[x, y, z]
# A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
# x_min, x_max = ax6.get_xlim()
# y_min, y_max = ax6.get_ylim()
# X, Y = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
# XX = X.flatten()
# YY = Y.flatten()
# C,_,_,_ = linalg.lstsq(A, data[:,2])
# Z = C[0]*X + C[1]*Y + C[2]
# ax6.plot_surface(X, Y, X, alpha=0.6)
#
# plt.show()
# plt.close()
#
# x = isofO2_T
# y = isofO2_P
# z = isofO2_D
# fig5 = plt.figure()
# ax7 = fig5.add_subplot(111)
# ax7.scatter(x, z)
# ax7_regress = ls_regression(x=x, y=z)
# ax7_slope, ax7_intercept = ax7_regress.lin_ls_regression()
# x_min3, x_max3 = ax7.get_xlim()
# y_min3, y_max3 = ax7_intercept, ax7_intercept + ax7_slope*(x_max3)
# ax7.plot([0, x_max3], [y_min3, y_max3])
# ax7.axvline(np.mean(x), c='g')
# ax7.axhline(np.mean(z), c='g')
#
# plt.show()
# plt.close()
#
#
#
#
#
# reg = mult_lin_regression(temperature=temperature, pressure=pressure, fO2=fO2, partitioncoeff=D)
# fit = reg.mult_lin_regress()
# print(fit)
