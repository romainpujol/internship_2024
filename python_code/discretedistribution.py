################################################################################
###################### DiscreteDistribution Class ##############################
################################################################################

"""
    Class to handle discrete distribution. Built on top of numpy for efficiency.
    Tailored to our needs compared to scipy.
"""
import numpy as np

class DiscreteDistribution:
    def __init__(self, atoms: np.ndarray, probabilities: np.ndarray):
        assert len(atoms) == len(probabilities), "Atoms and probabilities must have the same length"
        assert np.isclose(probabilities.sum(), 1), "Probabilities must sum to 1"
        
        self.atoms = atoms
        self.probabilities = probabilities

    @classmethod
    def from_lists(cls, atoms_list, probabilities_list):
        atoms = np.array(atoms_list)
        probabilities = np.array(probabilities_list)
        return cls(atoms, probabilities)

    def __len__(self):
        return len(self.atoms) 

    def __getitem__(self, index): 
        return self.atoms[index], self.probabilities[index]

    def __str__(self):
        return f"DiscreteDistribution(atoms={self.atoms.tolist()}, probabilities={self.probabilities.tolist()})"

    def __repr__(self):
        return f"DiscreteDistribution(atoms={self.atoms!r}, probabilities={self.probabilities!r})"

    def get_atoms(self): 
        return self.atoms

    def get_probabilities(self):
        return self.probabilities

    def sample(self, size: int = 1): 
        return np.random.choice(self.atoms, size=size, p=self.probabilities)

    def mean(self):
        return np.average(self.atoms, weights=self.probabilities, axis=0)

    def dim(self):
        if len(self.atoms) == 0:
            return 0
        return len(self.atoms[0])