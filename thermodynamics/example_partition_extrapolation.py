import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import sys, os
os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from stats.regression import ls_regression



w_df = pd.read_csv('w_partitioning_play.csv')
temperature = w_df['Temperature']
pressure = w_df['Pressure']
fO2 = w_df['Corrected_fO2']
D = w_df['Corrected_logD']

fig = plt.figure()
ax = fig.add_subplot(111)
x = [1,2,3,4,5]
y = [2,4,5,4,5]
ax.scatter(x,y)
thing = ls_regression(x=x, y=y)
sloper, intercepter, = thing.lin_ls_regression()
print("***{}***{}***".format(sloper, intercepter))
x_min1, x_max1 = ax.get_xlim()
print("{}/{}".format(x_min1, x_max1))
y_min1, y_max1 = intercepter, intercepter + sloper*(x_max1)
ax.plot([0, x_max1], [y_min1, y_max1])
ax.axvline(np.mean(x), c='g')
ax.axhline(np.mean(y), c='g')
plt.show()
plt.close()



fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.set_title("logD(Temperature)")
ax1.set_xlabel("Temperature")
ax1.set_ylabel("logD")
ax1.scatter(temperature, D)
ax1_regress = ls_regression(x=temperature, y=D)
ax1_slope, ax1_intercept = ax1_regress.lin_ls_regression()
print("***{}***{}***".format(ax1_slope, ax1_intercept))
x_min1, x_max1 = ax1.get_xlim()
y_min1, y_max1 = ax1_intercept, ax1_intercept + ax1_slope*(x_max1)
ax1.plot([0, x_max1], [y_min1, y_max1])
ax1.axvline(np.mean(temperature), c='g')
ax1.axhline(np.mean(D), c='g')

ax2 = fig.add_subplot(312)
ax2.set_title("logD(Pressure)")
ax2.set_xlabel("Pressure")
ax2.set_ylabel("logD")
ax2.scatter(pressure, D)
ax2_regress = ls_regression(x=pressure, y=D)
ax2_slope, ax2_intercept = ax2_regress.lin_ls_regression()
print("***{}***{}***".format(ax2_slope, ax2_intercept))
x_min2, x_max2 = ax2.get_xlim()
y_min2, y_max2 = ax2_intercept, ax2_intercept + ax2_slope*(x_max2)
ax2.plot([0, x_max2], [y_min2, y_max2])
ax2.axvline(np.mean(pressure), c='g')
ax2.axhline(np.mean(D), c='g')

ax3 = fig.add_subplot(313)
ax3.set_title("logD(fO2)")
ax3.set_xlabel("fO2 (delta IW)")
ax3.set_ylabel("logD")
ax3.scatter(fO2, D)
ax3_regress = ls_regression(x=fO2, y=D)
ax3_slope, ax3_intercept = ax3_regress.lin_ls_regression()
print("***{}***{}***".format(ax3_slope, ax3_intercept))
x_min3, x_max3 = ax3.get_xlim()[::-1]
y_min3, y_max3 = ax3_intercept, ax3_intercept + ax3_slope*(x_max3)
ax3.plot([0, x_max3], [y_min3, y_max3])
ax3.axvline(np.mean(fO2), c='g')
ax3.axhline(np.mean(D), c='g')

plt.show()
plt.close()
