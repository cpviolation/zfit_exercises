import numpy as np
import zfit
from zfit import z

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