import pandas as pd
import numpy as np
import time



class solution:

    """
    Manages solution chemistry and phases in variable dataframes.
    """

    def __init__(self, box_length):
        self.solution = pd.DataFrame({'object_id': [np.nan for i in list(range(box_length))],
                                      'object': [np.nan for i in list(range(box_length))]})

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

        # checks to see if the molecule exists as a column in dataframe. if not, creates column.
        for molecule in composition:
            if molecule not in self.solution:
                self.solution[molecule] = np.NAN
            else:
                pass

            self.solution[molecule][row] = float(composition[molecule])

        return self.solution



    def update_solution(self, box):
        pass