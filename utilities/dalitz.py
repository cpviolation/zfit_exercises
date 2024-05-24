import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

class DalitzKinematics:
    def __init__(self, mother_mass, daughters_masses, tensor=False):
        """A class to calculate the kinematics of a decay

        Args:
            mother_mass (float): the mass of the mother particle
            daughters_masses (list): the masses of the daughters particles
            tensor (bool): enables tensor calculations with tensorflow

        Raises:
            ValueError: Less than two daughters are provided
        """        
        if len(daughters_masses) != 3:
            raise ValueError('Three daughters are required in a Dalitz plot')
        self.mother_mass = mother_mass
        self.daughters_masses = daughters_masses
        self.tensor = tensor
        self.lib = tf if tensor else np
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
    
    def Inside(self, x, y, daughters_indices):
        """Checks whether a point is inside the Dalitz plot (squared masses of the daughters)

        Args:
            x (float): the x coordinate of the point
            y (float): the y coordinate of the point
            daughters_indices (list): the list of indices of the daughters on the interest

        Returns:
            bool: `True` if the point is inside the Dalitz plot, `False` otherwise
        """        
        m2Sq = self.daughters_masses[daughters_indices[1]]**2
        m3Sq = self.daughters_masses[self.otherDaughter(daughters_indices)]**2
        Est2 = self.EStar2(self.lib.sqrt(x), daughters_indices)
        Est3 = self.EStar3(self.lib.sqrt(x), daughters_indices)
        Ymin= (Est2+Est3)*(Est2+Est3) - (self.lib.sqrt(Est2*Est2-m2Sq)+ self.lib.sqrt(Est3*Est3-m3Sq))*(self.lib.sqrt(Est2*Est2-m2Sq)+ self.lib.sqrt(Est3*Est3-m3Sq))
        Ymax= (Est2+Est3)*(Est2+Est3) - (self.lib.sqrt(Est2*Est2-m2Sq)- self.lib.sqrt(Est3*Est3-m3Sq))*(self.lib.sqrt(Est2*Est2-m2Sq)- self.lib.sqrt(Est3*Est3-m3Sq))
        return (y > Ymin) & (y < Ymax)
    
    def KinematicallyAllowed(self, s1, s2, s3):
        """Checks whether a point is kinematically allowed

        Args:
            s1 (float): the invariant mass squared of the first combination of daughters
            s2 (float): the invariant mass squared of the second combination of daughters
            s3 (float): the invariant mass squared of the third combination of daughters

        Returns:
            bool: `True` if the point is kinematically allowed, `False` otherwise
        """        
        m1Sq = self.daughters_masses[0]**2
        m2Sq = self.daughters_masses[1]**2
        m3Sq = self.daughters_masses[2]**2
        return (s1 + s2 + s3 == self.mother_mass**2 - m1Sq - m2Sq - m3Sq)
    
    
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

    def Plot(self, m1, m2, weights=None, ax=None, **kwargs):
        if len(m1) != len(m2):
            raise ValueError('The input arrays must have the same length')
        if type(m1) not in [ np.ndarray, list ] or \
           type(m2) not in [ np.ndarray, list ]:
            raise ValueError('The input arrays must be numpy arrays or lists')
        if weights is not None and len(weights) != len(m1):
            raise ValueError('The weights array must have the same length as the input arrays') 
        # prepare the arrays
        nan_indices = np.isnan(m1) | np.isnan(m2)
        m1a = m1[~nan_indices]
        m2a = m2[~nan_indices]
        weights = weights[~nan_indices] if weights is not None else None
        # plot range
        xwid = (m1a.max()-m1a.min())
        xmin = m1a.min() - 0.05*xwid
        xmax = m1a.max() + 0.05*xwid
        ywid = (m2a.max()-m2a.min())
        ymin = m2a.min() - 0.05*ywid
        ymax = m2a.max() + 0.05*ywid
        # plot
        # create figure if not existing
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        kwargs['norm'] = mpl.colors.Normalize(vmin=0)
        h = ax.hist2d(m1a, m2a, cmin=1, range=[[xmin, xmax], [ymin, ymax]], weights=weights, **kwargs) # ,norm=mpl.colors.Normalize(vmin=0), kwargs=kwargs)
        plt.colorbar(h[3], ax=ax, extend='max')
        return ax if fig is None else fig, ax

    def PlotBoundary(self, ax, daughters_indices, num_points=1000):
        # needs to draw 2 lines: 1 above the diagonal and the other one below with the phase space limits
        # x needs to run between m12min=m1+m2 and m12max=M-m3
        mSq, Ymin, Ymax = self.Contour(daughters_indices, num_points)
        ax.plot(mSq, Ymin, color='r')
        ax.plot(mSq, Ymax, color='r')
        return
