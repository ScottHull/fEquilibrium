import pandas as pd



class solution:

    """
    Describes species in solution at a given point in time.
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def create_solution(self):
        df = pd.DataFrame({
            "Stable Phases", "Mass", "Composition"
                           },
                          index='') # create dataframe from which solution chemistry can be handled

    def update_system(self):
        pass