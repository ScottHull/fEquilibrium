import os
import numpy as np
import pandas as pd
from math import exp, log


pd.options.mode.chained_assignment = None # turns off Pandas warning about dataframe overwrites

isotopes = pd.read_csv('chem.csv', index_col=0) # loads isotope data from a local source
# Note that all chemical abundance data is given in ppm due to trace nature of Hf-Ta-W system
def f(x):
    try:
        return np.float(x)
    except:
        return np.nan
isotopes["Abundance"] = isotopes['Abundance'].apply(f) # convert Abundances column to floats or NaN
time = 0 # time, in years
time_resolution = 1000000.0 # time resolution, in years
time_limit = (4.5 * 10**9) # limits the time evolution to the age of the Earth
iteration = 0 # the number of iterations the model has been exposed to
max_iterations = round(time_limit / time_resolution) # the maximum number of iterations (default: age of Earth 4.5Gya)


class partition():

    def __init__(self, element):
        self.element = element

class decay:

    """
    Models decay of of a radioactive isotope family tree.
    """

    def __init__(self, element):
        self.element = element

    def rad_decay(self, isotope_df, time_resolution):
        daughters = [] # first element will be the primordial parent, all else daughters in descending order
        daughters.append(self.element) # adds primordial parent
        if isotope_df['Daughter'][self.element] != 'NONE': # checks to see that isotope has a daughter
            daughter = str(isotope_df['Daughter'][self.element])
            while daughter != 'NONE': # adds daughters to list in stepwise fashion
                daughters.append(daughter)
                daughter = str(isotope_df['Daughter'][daughter])
        if len(daughters) > 1: # check to see that primordial isotope has daughters, i.e. is radioactive
            iterdaughters = iter(reversed(daughters)) # now stable isotope is first in list, primordial parent last
            next(iterdaughters) # skips the stable isotope
            for i in iterdaughters: # iterates through daughters to model decay
                half_life = float(isotope_df['Half-Life'][i]) # extracts half-life
                daughter = isotope_df['Daughter'][i] # identifies daughter
                curr_amount = float(isotope_df['Abundance'][i]) # finds current abundance of isotope in question
                k = log(0.5) / half_life # gamma variable in decay equation
                remaining_amount = curr_amount * exp(k * time_resolution) # models remaining amount of decay, where time is the time resolution
                isotope_df['Abundance'][i] = remaining_amount
                if isotope_df['Daughter'][i] != 'NONE':
                    isotope_df['Abundance'][daughter] = float(isotope_df['Abundance'][daughter] + (curr_amount - remaining_amount))
                else:
                    pass
        else: # no daughters, doesn't decay into daughter products
            pass
        return isotope_df # returns the updated isotope dataframe for one iteration


hf_182 = decay(element='182-Hf')
while time < 55000000:
    hf_182.rad_decay(isotope_df=isotopes, time_resolution=time_resolution)
    time += time_resolution
    iteration += 1
    # print("\nIterations: {}\nTime: {} years\n182-Hf: {} mol\n182-Ta: {} mol\n182-W: {} mol".format(iteration, time,
    #                                                                                               isotopes['Abundance'][
    #                                                                                                   '182-Hf'],
    #                                                                                               isotopes['Abundance'][
    #                                                                                                   '182-Ta'],
    #                                                                                               isotopes['Abundance'][
    #                                                                                                   '182-W']))
print("\nIsotope Dataframe:")
print(isotopes)
print("Time: {}, Iterations: {}".format(time, iteration))






# hf_182 = decay(element='182-Hf') # opens an instance of the decay class for the primordial isotope, keep formatting!
# for i in range(3):
#     if isotopes['Abundance']['182-Hf'] != 0:
#         run_decay = hf_182.rad_decay(isotope_df=isotopes, time_resolution=time_resolution)
#         isotopes['Abundance'] = run_decay['Abundance']
#         time += time_resolution
#         iteration += 1
#         print("\nIterations: {}\nTime: {} years\n182-Hf: {} mol\n182-Ta: {} mol\n182-W: {} mol".format(iteration, time,
#                                                                                                       isotopes['Abundance'][
#                                                                                                           '182-Hf'],
#                                                                                                       isotopes['Abundance'][
#                                                                                                           '182-Ta'],
#                                                                                                       isotopes['Abundance'][
#                                                                                                           '182-W']))







