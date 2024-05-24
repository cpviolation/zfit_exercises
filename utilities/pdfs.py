import numpy as np
from dalitz import DalitzKinematics
import zfit
from zfit import z
import tensorflow as tf

class JohnsonSU(zfit.pdf.BasePDF):

    def __init__(self, mean, sigma, nu, tau, obs, extended=None, norm=None, name=None):
        params = {'mean' : mean,  # 'mean' is the name as it will be named in the PDF, mean is just the parameter to create the PDF
                  'sigma': sigma,
                  'nu'   : nu,
                  'tau'  : tau,
                  }
        super().__init__(obs=obs, params=params, extended=extended, norm=norm,
                         name=name)

    def _unnormalized_pdf(self, x):
        x = z.unstack_x(x)
        mean = self.params['mean']
        sigma= self.params['sigma']
        nu   = self.params['nu']
        tau  = self.params['tau']
        omega = -nu * tau
        w = np.exp( tau * tau )
        c = 0.5 * ( w - 1 ) / np.sqrt( w * np.cosh( 2*w ) + 1 )
        y = ( x - (mean + c * sigma * np.sqrt(w) * np.sinh(omega)) ) / c / sigma 
        r = - nu + np.arcsinh( y ) / tau
        return z.exp( 0.5 * np.exp( -0.5 * r * r) / ( np.pi * c * sigma * tau * np.sqrt( y*y + 1 ) ) )
    

class DalitzUniform(zfit.pdf.ZPDF):
    """Uniform Dalitz plot PDF

    Args:
        mMother (`zfit.Parameter`): the mass of the mother particle
        m1 (`zfit.Parameter`): the mass of the first daughter particle
        m2 (`zfit.Parameter`): the mass of the second daughter particle
        m3 (`zfit.Parameter`): the mass of the third daughter particle
        obs (`zfit.Space`):
        name (str):
        dtype (tf.DType):
    """

    _PARAMS = ['mMother', 'm1', 'm2', 'm3']
    _N_OBS = 3

    @zfit.supports()
    def _unnormalized_pdf(self, x, params):
        mMother = self.params['mMother']
        m1= self.params['m1']
        m2= self.params['m2']
        m3= self.params['m3']
        mSq12 = x[0]
        mSq23 = x[1]
        mSq13 = x[2]
        dalitzKin = DalitzKinematics(mMother, [m1, m2, m3], True)
        pdf = 1.0 * tf.cast(dalitzKin.Inside(mSq12, mSq23, [0, 1]), tf.float32)# * \
                            #dalitzKin.KinematicallyAllowed(mSq23,mSq13,mSq23), tf.float32)
        return pdf