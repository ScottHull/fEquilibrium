import pandas as pd
from math import exp, log
import numpy as np



class decay:

    """
    Models decay of of a radioactive isotope family tree.  Takes arbitrary primordial isotope as parameter self, then
    models decay into daughter products based on the information provided in the passed isotope dataframe.
    Be sure that solution follows the formatting of 'chem.csv' if using a custom isotope table.
    Abundances of radioactive isotopes given in ppm.
    Abundances should reflect solar system initial abundances or initial Earth abundances, depending on how one adjusts
        time/time resolution/T0 for the model.
    """

    pd.options.mode.chained_assignment = None # turns off Pandas warning about dataframe overwrites

    def __init__(self):
        pass

    def int_to_float(self, x):  # converts integer values into floats in a Pandas dataframe
        try:
            return np.float(x) # if a number, it is converted to a float
        except:
            return np.nan # if not a number, it is NaN



    # TODO: come up with a better scheme for arbitrary decay chain detection and handling
    @classmethod
    def decay_chain(cls, solution, isotope_df):
        """
        Identifies the entire decay chain of elements given.
        :param solution:
        :param isotope_df:
        :return:
        """

        decay_chain = []

        for element in solution:
            if element in isotope_df['Isotope']:
                if isotope_df['Primordial Parent'][element] == 'TRUE': # checks if this is the first element in a decay chain
                    decay_chain.append(element)
                    daughter = isotope_df['Daughter'][element]
                    while daughter != 'NONE': # loops through the isotope df to identify entire daughter chain
                        decay_chain.append(isotope_df['Daughter'][element])
                        if daughter not in solution: # if daughter not a header in the solution dateframe, makes it one
                            solution[daughter] = np.NaN
                        daughter = isotope_df['Daughter'][daughter]

        return decay_chain


    def rad_energy(self, deltaTime): # energy produced by radioactivity
        pass

    def rad_decay(self, solution, deltaTime):
        """
        :param solution:
        :param deltaTime:
        :return: solution
        """

        isotope_df = pd.read_csv('/radioactivity/isotopes.csv', index='Isotope')
        chain = self.decay_chain(solution=solution, isotope_df=isotope_df)
        if len(chain) > 1:
            if len(chain) > 1: # check to see that primordial isotope has daughters, i.e. is radioactive
                iterdaughters = iter(reversed(chain)) # now stable isotope is first in list, primordial parent last
                next(iterdaughters) # skips the stable isotope
                for i in iterdaughters: # iterates through daughters to model decay
                    half_life = float(isotope_df['Half-Life'][i]) # extracts half-life
                    daughter = isotope_df['Daughter'][i] # identifies daughter
                    for row in solution:
                        curr_amount = float(isotope_df[i][row]) # finds current abundance of isotope in question
                        k = log(0.5) / half_life # gamma variable in decay equation
                        remaining_amount = curr_amount * exp(k * deltaTime) # models remaining amount of decay,
                                                # where time is the time resolution (N = N0*exp(gamma*t))
                        solution[i][row] = remaining_amount # the remaining amount of isotope after decay
                        if isotope_df['Daughter'][i] != 'NONE': # makes sure that a daughter exists
                            solution[daughter][row] = float(solution['Abundance'][daughter] +
                                                (curr_amount - remaining_amount)) # updates daughter abundance
                        else:
                            pass
            else: # no daughters, doesn't decay into daughter products
                pass
            return solution # returns the updated isotope dataframe for one iteration










        # solution["Abundance"] = solution['Abundance'].apply(self.int_to_float)  # convert 'Abundance' column to floats or NaN
        # # Note that all chemical abundance data is given in ppm due to trace nature of Hf-Ta-W system
        # daughters = [] # first element will be the primordial parent, all else daughters in descending order
        # daughters.append(self.element) # adds primordial parent
        # if solution['Daughter'][self.element] != 'NONE': # checks to see that isotope has a daughter
        #     daughter = str(solution['Daughter'][self.element])
        #     while daughter != 'NONE': # adds daughters to list in stepwise fashion
        #         daughters.append(daughter) # adds daughter to the list
        #         daughter = str(solution['Daughter'][daughter]) # finds the next daughter and repeats the loop
        # if len(daughters) > 1: # check to see that primordial isotope has daughters, i.e. is radioactive
        #     iterdaughters = iter(reversed(daughters)) # now stable isotope is first in list, primordial parent last
        #     next(iterdaughters) # skips the stable isotope
        #     for i in iterdaughters: # iterates through daughters to model decay
        #         half_life = float(solution['Half-Life'][i]) # extracts half-life
        #         daughter = solution['Daughter'][i] # identifies daughter
        #         curr_amount = float(solution['Abundance'][i]) # finds current abundance of isotope in question
        #         k = log(0.5) / half_life # gamma variable in decay equation
        #         remaining_amount = curr_amount * exp(k * deltaTime) # models remaining amount of decay,
        #                                 # where time is the time resolution (N = N0*exp(gamma*t))
        #         solution['Abundance'][i] = remaining_amount # the remaining amount of isotope after decay
        #         if solution['Daughter'][i] != 'NONE': # makes sure that a daughter exists
        #             solution['Abundance'][daughter] = float(solution['Abundance'][daughter] +
        #                                 (curr_amount - remaining_amount)) # updates daughter abundance
        #         else:
        #             pass
        # else: # no daughters, doesn't decay into daughter products
        #     pass
        # return solution # returns the updated isotope dataframe for one iteration




class isotopeconvert:

    """
    Converts isotopes from popular notation to abundances in ppm.
    """

    def __init__(self, isotope):
        self.isotope = isotope

    def epilon_tungsten(self, epislon_w):
        pass


