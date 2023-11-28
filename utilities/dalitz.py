import numpy as np

class DalitzKinematics:
    def __init__(self, mother_mass, daughters_masses):
        """A class to calculate the kinematics of a decay

        Args:
            mother_mass (float): the mass of the mother particle
            daughters_masses (list): the masses of the daughters particles

        Raises:
            ValueError: Less than two daughters are provided
        """        
        if len(daughters_masses) != 3:
            raise ValueError('Three daughters are required in a Dalitz plot')
        self.mother_mass = mother_mass
        self.daughters_masses = daughters_masses
        return
    
    def otherDaughter(self, daughters_indices):
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        other_indices = np.arange(len(self.daughters_masses))
        other_indices = np.setdiff1d(other_indices, daughters_indices)
        return other_indices[0]
        
    def mSqMin(self, daughters_indices):
        """Calculate the minimum invariant mass squared of a combination of daughters

        Args:
            daughters_indices (list): the indices of the daughters of interest

        Raises:
            ValueError: no daughters indices are provided

        Returns:
            float: the minimum invariant mass squared of the combination
        """        
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        comb_mass = 0
        for i in daughters_indices:
            comb_mass += self.daughters_masses[i]
        return (comb_mass)**2
    
    def mSqMax(self, daughters_indices):
        """Calculate the maximum invariant mass squared of a combination of daughters

        Args:
            daughters_indices (list): the indices of the daughters of interest

        Raises:
            ValueError: no daughters indices are provided

        Returns:
            float: the maximum invariant mass squared of the combination
        """        
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        return (self.mother_mass - self.daughters_masses[self.otherDaughter(daughters_indices)])**2
    
    def EStar2(self, inv_mass, daughters_indices):
        """Calculates the energy squared of a combination of daughters

        Args:
            inv_mass (float): the invariant mass of the combination
            daughters_indices (list): the list of indices of the daughters of interest

        Raises:
            ValueError: two daughters indices are required

        Returns:
            float: the energy squared of the combination to calculate the Dalitz plot range
        """        
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        m1Sq = self.daughters_masses[daughters_indices[0]]**2
        m2Sq = self.daughters_masses[daughters_indices[1]]**2
        Est2 = inv_mass*inv_mass - m1Sq + m2Sq
        Est2/= 2*inv_mass
        return Est2
    
    def EStar3(self, inv_mass, daughters_indices):
        """Calculates the energy squared of a combination of daughters

        Args:
            inv_mass (float): the invariant mass of the combination
            daughters_indices (list): the list of indices of the daughters of interest

        Raises:
            ValueError: two daughters indices are required

        Returns:
            float: the energy squared of the decay remnants to calculate the Dalitz plot range
        """        
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        mMSq = self.mother_mass**2
        mOtherSq = self.daughters_masses[self.otherDaughter(daughters_indices)]**2
        Est3 = mMSq- inv_mass*inv_mass - mOtherSq
        Est3/= 2*inv_mass
        return Est3
    
    def Contour(self, daughters_indices, num_points=1000):
        """Calculate the Dalitz plot contour as mSq23 vs mSq12

        Args:
            daughters_indices (list): the list of indices of the daughters on the interest
            Npoints (int): the number of points for the contour

        Raises:
            ValueError: two daughters indices are required

        Returns:
            tuple: the x and y_low,high coordinates of the contour
        """        
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        #m1Sq = self.daughters_masses[daughters_indices[0]]**2
        m2Sq = self.daughters_masses[daughters_indices[1]]**2
        m3Sq = self.daughters_masses[self.otherDaughter(daughters_indices)]**2
        mSqMin = self.mSqMin(daughters_indices)
        mSqMax = self.mSqMax(daughters_indices)
        mSq = np.linspace(mSqMin, mSqMax, num_points)
        Est2 = self.EStar2(np.sqrt(mSq), daughters_indices)
        Est3 = self.EStar3(np.sqrt(mSq), daughters_indices)
        Ymin= np.power(Est2+Est3,2) - np.power(np.sqrt(Est2*Est2-m2Sq)+ np.sqrt(Est3*Est3-m3Sq),2)
        Ymax= np.power(Est2+Est3,2) - np.power(np.sqrt(Est2*Est2-m2Sq)- np.sqrt(Est3*Est3-m3Sq),2)
        return (mSq, Ymin, Ymax)