import os
import numpy as np
import sys



class LengthError(Exception):
    pass


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
        try:
            self.x.apply(self.int_to_float)
            self.y.apply(self.int_to_float)
        except:
            pass
        if len(self.x) == len(self.y):
            x_mean = sum(self.x) / len(self.x) # takes the mean of x values
            y_mean = sum(self.y) / len(self.y) # takes the mean of y values
            dist_x = [float(i - x_mean) for i in self.x] # x - x_bar
            dist_y = [float(i - y_mean) for i in self.y] # y - y_bar
            sqr_dist_x = [(i**2) for i in dist_x]
            prod_dists = [a * b for a,b in zip(dist_x, dist_y)] # (x - x_bar) * (y - y_bar)
            b_1 = sum(prod_dists) / sum(sqr_dist_x) # b_1 = [sum(x - x_bar)(y - y_bar))] / sum((x-x_bar)^2)
            b_0 = y_mean - (b_1 * x_mean) # y_hat = b_0 + b_1 * x_mean
            slope = b_1
            intercept = b_0

            print("x: {}\ny: {}\nmeanx: {}\nmeany: {}\ndistx: {}\ndisty: {}\nsqrdistx: {}\nproddists: {}\nb1: {}\nb0: {}\n".format(
                self.x, self.y, x_mean, y_mean, dist_x, dist_y, sqr_dist_x, prod_dists, b_1, b_0
            ))

            return slope, intercept

        else:
            return LengthError("Lists x and y do not match in length!")
