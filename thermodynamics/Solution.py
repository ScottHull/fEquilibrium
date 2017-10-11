import pandas as pd
import numpy as np



class solution:

    """
    Manages solution chemistry and phases in variable dataframes.
    """

    def __init__(self):
        self.solution = pd.DataFrame({'object_id': [], 'object': []})

    def create_solution(self, object, composition, box):
        for row in box.index:
            if box['object'][row] == object:
                self.solution['object_id'].append(box['object_id'][row])
                self.solution['object'].append((box['object'][row]))
                for molecule in composition:
                    if not self.solution[molecule]:
                        self.solution[molecule] = np.NAN
                    else:
                        self.solution[molecule].append(composition[molecule])



    def update_solution(self, box):
        pass