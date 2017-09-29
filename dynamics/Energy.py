



class energy:

    def __init__(self):
        self.gravity = 9.81

    def kinetic_energy(self, mass, velocity):
        ke = 0.5 * mass * (velocity**2) # ke = 1/2*m*v^2
        return ke

    def potential_energy(self, mass, height):
        pe = mass * self.gravity * height # pe = m*g*h
        return pe

    def release_energy(self, former_pe, current_pe):
        friction_energy = abs(current_pe - former_pe) # assume that all energy release as frictional energy, or the
        #       difference in the potential energies at position 2 and position 1
        return friction_energy