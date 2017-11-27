import pandas as pd
import numpy as np
from radioactivity.Radioactivity import decay
from collections import Counter
import time



class solution:

    """
    Manages solution chemistry and phases in variable dataframes.
    """

    def __init__(self, box_length):
        self.solution = pd.DataFrame({'object_id': [np.nan for i in list(range(box_length))],
                                      'object': [np.nan for i in list(range(box_length))]})
        self.decay = decay()

    def create_solution(self, object, composition, box, row):
        """
        Creates a dataframe to manage chemistry within the box.  This should be called when inserting a new type of object
        into the box, and not to update existing components of the solution dataframe.  Current assumption is homogenous
        composition of all objects of similar name within the box.  Individual components can have composition updated
        individually in accessory functions.

        :param object: the object inserted into the box
        :param composition: the composition of the object
        :param box: self.space, defined in module 'Box.py'
        :return: self.solution, the solution dataframe
        """
        self.solution['object_id'][row] = box['object_id'][row]
        self.solution['object'][row] = box['object'][row]

        # if the element/molecule does not exist in the specified dataframe, it is set to an abundance of 0
        # this prevents errors to to the overwriting of data
        if Counter(self.solution.columns.values.tolist()[2:]) != Counter(list(composition.keys())): # looks to see if both lists are equal
            for i in self.solution.columns.values.tolist()[2:]: # if not, replace with 0's
                self.solution[i][row] = float(0)

        # checks to see if the molecule exists as a column in dataframe. if not, creates column.
        for molecule in composition:
            if molecule not in self.solution:
                self.solution[molecule] = np.NaN
            else:
                pass

            self.solution[molecule][row] = float(composition[molecule]) # inserts the compositional value


    def update_solution(self, deltaTime):
        """
        Updates the solution according to chemical reactions.  Currently supports radioactivity.
        :param deltaTime: time step of each model iteration
        :return: self.solution
        """

        self.solution = self.decay.rad_decay(solution=self.solution, deltaTime=deltaTime)
        return self.solution

    def get_solution(self):
        """
        Returns the self.solution chemical dataframe.
        :return: self.solution
        """

        return self.solution
