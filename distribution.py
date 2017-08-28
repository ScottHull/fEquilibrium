import os
import numpy as np
import pandas as pd

isotopes = pd.read_csv('chem.csv', index_col=0)
time = 100


class partition:

    def __init__(self, element):
        self.element = element

class decay:
    def __init__(self, element, daughter, element_abun, daughter_abun):
        self.element = element
        self.daughter = daughter
        self.element_abun = element_abun
        self.daughter_abun = daughter_abun

    def hl_decay(self, element_abun, daughter_abun):
        decay_abundance = float(element_abun) * 0.5
        daughter_abun = float(daughter_abun) + (float(element_abun) - decay_abundance)
        return decay_abundance, daughter_abun


hf_182 = isotopes['Abundance']['182-Hf']
hf_182_decay = decay
hf_182_decay.hl_decay(hf_182)


