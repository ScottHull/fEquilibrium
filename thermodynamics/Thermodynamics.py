import pandas as pd
from math import exp
import numpy as np
import os
os.sys.path.append(os.path.dirname(os.path.abspath('.'))); from stats.Regression import ls_regression, mult_lin_regression
import re



class partition:

    """
    Describes the behavior of elemental partitioning between chemical species in solution.
    """

    def __init__(self, element, phase):
        """
        :param element: the element being incorporated
        :param phase: the phase the element is being incorporated int0
        """
        self.element = element
        self.phase = phase

    def int_to_float(self, x):  # converts integer values into floats in a Pandas dataframe
        try:
            return np.float(x) # if a number, it is converted to a float
        except:
            return np.nan # if not a number, it is NaN


    def henry_partition(self, phase_df, deltaH, pressure, deltaV, temperature):

        """
        :param phase_df: phase dataframe
        :param deltaH: change in enthalpy
        :param pressure: pressure
        :param deltaV: change in volume
        :param temperature: temperature
        :return: D
        """

        # Note that this is the pressure-volume dependent relationship
        gas_const = 8.312 #J/mol*k




class avg_deltaGnot:

    """
    Given a Pandas dataframe of logD and temperature, calculates the average deltaG_0.
    """

    def __init__(self, logD, temperature):
        """
        :param logD: the logarithmic partition coefficient (in Pandas dataframe)
        :param temperature: the temperature at which logD was calculated (in Pandas dataframe)
        """
        self.logD = logD
        self.temperature = temperature

    def int_to_float(self, x):  # converts integer values into floats in a Pandas dataframe
        try:
            return np.float(x) # if a number, it is converted to a float
        except:
            return np.nan # if not a number, it is NaN

    def calc_avg_deltaGnot(self):
        """
        :return: average deltaG_not
        """
        gas_const = 8.312  # J/mol*k
        self.logD.apply(self.int_to_float)
        self.temperature.apply(self.int_to_float)
        logD = self.logD.values.tolist() # converts Pandas dataframe to list
        temperature = self.temperature.values.tolist() # converts Pandas dataframe to list
        deltaGnot_calcs = [] # list of calculated deltaGnot's
        loop = 0 # loop counter for indexing purposes
        for i in logD:
            corresponding_t = temperature[loop]
            deltaGnot = -i / (gas_const * corresponding_t)
            deltaGnot_calcs.append(deltaGnot)
        avg_deltaGnot = sum(deltaGnot_calcs) / len(deltaGnot_calcs)
        return avg_deltaGnot

