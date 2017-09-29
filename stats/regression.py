import numpy as np
from statsmodels.formula.api import ols
import pandas as pd
import sys
import traceback



class ls_regression:

    """
    Linear least squares regression, given x and y lists.
    Returns slope, intercept
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def int_to_float(self, x):  # converts integer values into floats in a Pandas dataframe
        try:
            return np.float(x) # if a number, it is converted to a float
        except:
            return np.nan # if not a number, it is NaN

    def lin_ls_regression(self):
        try: # attempts to convert values to floats
            self.x.apply(self.int_to_float)
            self.y.apply(self.int_to_float)
        except:
            pass
        try:
            x_mean = sum(self.x) / len(self.x) # takes the mean of x values
            y_mean = sum(self.y) / len(self.y) # takes the mean of y values
            dist_x = [float(i - x_mean) for i in self.x] # x - x_bar
            dist_y = [float(i - y_mean) for i in self.y] # y - y_bar
            sqr_dist_x = [(i**2) for i in dist_x]
            prod_dists = [a * b for a, b in zip(dist_x, dist_y)] # (x - x_bar) * (y - y_bar)
            b_1 = sum(prod_dists) / sum(sqr_dist_x) # b_1 = [sum(x - x_bar)(y - y_bar))] / sum((x-x_bar)^2)
            b_0 = y_mean - (b_1 * x_mean) # y_hat = b_0 + b_1 * x_mean
            slope = b_1
            intercept = b_0
            return slope, intercept

        except:
            traceback.print_exc()


class mult_lin_regression:

    """
    Multiple linear regression.
    """

    def __init__(self, temperature, pressure, fO2, partitioncoeff):
        self.temperature = temperature
        self.pressure = pressure
        self.fO2 = fO2
        self.partitioncoeff = partitioncoeff

    def int_to_float(self, x):  # converts integer values into floats in a Pandas dataframe
        try:
            return np.float(x) # if a number, it is converted to a float
        except:
            return np.nan # if not a number, it is NaN

    def mult_lin_regress(self):
        try: # attempts to convert values to floats
            self.temperature.apply(self.int_to_float)
            self.pressure.apply(self.int_to_float)
            self.fO2.apply(self.int_to_float)
            self.partitioncoeff.apply(self.int_to_float)
        except:
            pass
        if len(self.temperature) == len(self.pressure) == len(self.fO2) == len(self.partitioncoeff):
            x = (self.temperature).values.tolist()
            y = (self.pressure).values.tolist()
            z = (self.fO2).values.tolist()
            d = (self.partitioncoeff).values.tolist()
            data = pd.DataFrame({'temperature': x, 'pressure': y, 'fO2': z, 'partitioncoeff': d})
            model = ols("partitioncoeff ~ temperature + pressure + fO2", data).fit()
            print("\nModel Summary:")
            print(model.summary())
            print("\nModel parameters:")
            print(model._results.params)
            coeffs = model._results.params
            intercept = coeffs[0]
            temperature_coeff = coeffs[1]
            pressure_coeff = coeffs[2]
            fO2_coeff = coeffs[3]
            return intercept, temperature_coeff, pressure_coeff, fO2_coeff

        else:
            print("Series x, y, z, d do not match in length!")
            sys.exit(1)





