import pandas as pd
from math import exp




class partition:

    """
    Describes the behavior of elemental partitioning between chemical species in solution.
    """

    def __init__(self, element):
        self.element = element


    def henry_partition(self, temperature):
        boltzmann = 1.38064852 * (10**(-23))
        deltaH = 1
        deltaS = 1
        deltaG = deltaH - temperature * deltaS
        k = exp(-deltaG / (boltzmann * temperature))
        return k # should return the updated chemical system based on partition behavior



class geotherm:

    def __init__(self, depth):
        self.depth = depth