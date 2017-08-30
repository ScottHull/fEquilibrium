import os
import numpy as np
import pandas as pd
from math import exp, log

isotopes = pd.read_csv('chem.csv', index_col=0) # loads isotope data from a local source
time = 0 # time, in years
time_resolution = 1000.0 # time resolution, in years
time_limit = (4.5 * 10**9) # limits the time evolution to the age of the Earth
iteration = 0 # the number of iterations the model has been exposed to
max_iterations = round(time_limit / time_resolution) # the maximum number of iterations (default: age of Earth 4.5Gya)


class partition():

    def __init__(self, element):
        self.element = element

class decay:

    def __init__(self, element):
        self.element = element

    def rad_decay(self, isotope_df, time_resolution):
        daughters = [] # first element will be
        daughters.append(self.element)
        if isotope_df['Daughter'][self.element] != 'NONE':
            daughter = str(isotope_df['Daughter'][self.element])
            while daughter != 'NONE':
                daughters.append(daughter)
                daughter = str(isotope_df['Daughter'][daughter])
        if len(daughters) > 1:
            iterdaughters = iter(reversed(daughters))
            next(iterdaughters)
            for i in iterdaughters:
                half_life = float(isotope_df['Half-Life'][i])
                curr_amount = isotope_df['Abundance'][i]
                k = log(0.5) / half_life
                remaining_amount = curr_amount * exp(k * time_resolution)
                isotope_df['Abundance'][i] = remaining_amount
                print("\nAbundance of {}: {}".format(i, isotope_df['Abundance'][i]))
                if isotope_df['Daughter'][i] != 'NONE':
                    isotope_df['Abundance'][isotope_df['Daughter'][i]] = float(isotope_df['Abundance'][daughter]) + (curr_amount - remaining_amount)
                    print("Abundance of {}: {}".format(daughter, isotope_df['Abundance'][daughter]))
                else:
                    pass
        return isotope_df


hf_182 = decay(element='182-Hf')
for i in range(40):
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







