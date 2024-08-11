################################################################################
###################### DiscreteDistribution Class ##############################
################################################################################

import numpy as np

class DiscreteDistribution:
    """
    A class to represent and manipulate a discrete probability distribution.

    This class is built on top of NumPy for efficiency and is designed to be
    tailored to specific needs compared to general-purpose libraries like SciPy.

    Attributes
    ----------
    atoms : np.ndarray
        An array representing the support of the distribution (the atoms).
    probabilities : np.ndarray
        An array representing the probabilities associated with each atom.

    Methods
    -------
    from_lists(cls, atoms_list, probabilities_list)
        Class method to create a DiscreteDistribution instance from Python lists.
    __len__()
        Returns the number of atoms in the distribution.
    __getitem__(index)
        Returns the atom and its associated probability at the given index.
    __str__()
        Returns a string representation of the DiscreteDistribution.
    __repr__()
        Returns a formal string representation of the DiscreteDistribution.
    get_atoms()
        Returns the array of atoms in the distribution.
    get_probabilities()
        Returns the array of probabilities in the distribution.
    sample(size=1)
        Generates a random sample from the distribution.
    mean()
        Computes the weighted mean of the distribution.
    dim()
        Returns the dimensionality of the space in which the atoms reside.
    """

    def __init__(self, atoms: np.ndarray, probabilities: np.ndarray):
        """
        Initializes the DiscreteDistribution with atoms and their associated 
        probabilities.

        Parameters
        ----------
        atoms : np.ndarray
            An array of shape (n, d) where n is the number of atoms and d is 
            the dimension of the space.
        probabilities : np.ndarray
            An array of shape (n,) where each entry corresponds to the 
            probability of the associated atom.

        Raises
        ------
        AssertionError
            If the lengths of atoms and probabilities do not match, or if the 
            probabilities do not sum to 1.
        """
        assert len(atoms) == len(probabilities), \
            "Atoms and probabilities must have the same length"
        assert np.isclose(probabilities.sum(), 1), \
            "Probabilities must sum to 1"

        self.atoms = atoms
        self.probabilities = probabilities

    @classmethod
    def from_lists(cls, atoms_list, probabilities_list):
        """
        Class method to create a DiscreteDistribution instance from Python lists.

        Parameters
        ----------
        atoms_list : list
            A list of atoms where each atom is a list or array representing a 
            point in space.
        probabilities_list : list
            A list of probabilities associated with each atom.

        Returns
        -------
        DiscreteDistribution
            A new instance of DiscreteDistribution initialized with the provided 
            atoms and probabilities.
        """
        atoms = np.array(atoms_list)
        probabilities = np.array(probabilities_list)
        return cls(atoms, probabilities)

    def __len__(self):
        """
        Returns the number of atoms in the distribution.

        Returns
        -------
        int
            The number of atoms.
        """
        return len(self.atoms) 

    def __getitem__(self, index):
        """
        Returns the atom and its associated probability at the given index.

        Parameters
        ----------
        index : int
            The index of the atom.

        Returns
        -------
        tuple
            A tuple containing the atom and its associated probability.
        """
        return self.atoms[index], self.probabilities[index]

    def __str__(self):
        """
        Returns a string representation of the DiscreteDistribution.

        Returns
        -------
        str
            A string representing the distribution.
        """
        return (f"DiscreteDistribution(atoms={self.atoms.tolist()}, "
                f"probabilities={self.probabilities.tolist()})")

    def __repr__(self):
        """
        Returns a formal string representation of the DiscreteDistribution.

        Returns
        -------
        str
            A detailed string representing the distribution for debugging 
            purposes.
        """
        return (f"DiscreteDistribution(atoms={self.atoms!r}, "
                f"probabilities={self.probabilities!r})")

    def get_atoms(self): 
        """
        Returns the array of atoms in the distribution.

        Returns
        -------
        np.ndarray
            The array of atoms.
        """
        return self.atoms

    def get_probabilities(self):
        """
        Returns the array of probabilities in the distribution.

        Returns
        -------
        np.ndarray
            The array of probabilities.
        """
        return self.probabilities

    def sample(self, size: int = 1): 
        """
        Generates a random sample from the distribution.

        Parameters
        ----------
        size : int, optional
            The number of samples to generate (default is 1).

        Returns
        -------
        np.ndarray
            An array of sampled atoms.
        """
        return np.random.choice(self.atoms, size=size, p=self.probabilities)

    def mean(self):
        """
        Computes the weighted mean of the distribution.

        Returns
        -------
        np.ndarray
            The weighted mean of the atoms, calculated using the associated 
            probabilities.
        """
        return np.average(self.atoms, weights=self.probabilities, axis=0)

    def dim(self):
        """
        Returns the dimensionality of the space in which the atoms reside.

        Returns
        -------
        int
            The dimension of the atoms' space. Returns 0 if the distribution 
            has no atoms.
        """
        if len(self.atoms) == 0:
            return 0
        return len(self.atoms[0])
    
################################################################################
############################### Helper functions ###############################
################################################################################

def discrete_reallocation(
        xi: DiscreteDistribution, indexes: list[int], 
        weights: np.ndarray) -> DiscreteDistribution:
    """
    Construct a reduced DiscreteDistribution by reallocating the atoms based 
    on specified indexes and weights.

    Parameters
    ----------
    xi : DiscreteDistribution
        The original distribution from which a subset of atoms is selected.
    indexes : list[int]
        A list of indices representing the atoms to include in the reduced 
        distribution.
    weights : np.ndarray
        An array of weights (probabilities) corresponding to the selected atoms.

    Returns
    -------
    DiscreteDistribution
        A new instance of DiscreteDistribution representing the reduced 
        distribution.
    """
    atoms = xi.get_atoms()
    return DiscreteDistribution(atoms[indexes], weights)

def dummy_DiscreteDistribution(n: int, dim: int) -> DiscreteDistribution:
    """
    Generate a dummy DiscreteDistribution with randomly generated atoms.

    Parameters
    ----------
    n : int
        The number of atoms to generate.
    dim : int
        The dimension of the space in which the atoms reside.

    Returns
    -------
    DiscreteDistribution
        A DiscreteDistribution instance with randomly generated atoms and 
        probabilities.
    """
    atoms = np.random.rand(n, dim)  
    probabilities = np.random.rand(n)
    probabilities /= probabilities.sum()
    return DiscreteDistribution(atoms, probabilities)
