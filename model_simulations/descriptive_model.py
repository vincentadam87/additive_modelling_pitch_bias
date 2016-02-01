import numpy as np
from scipy.stats import norm
from abc import abstractmethod
# from utils import phi


class AbstractModel(object):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def bern(self,f1,f2,n_samp=100000):
        raise NotImplementedError()

    def llh(self,F1,F2,Y,n_samp=5000,skip=None):
        """probability f1 higher"""
        assert len(F1.shape)==1, 'input must be 1d ndarray'
        assert len(F1.shape)==1, 'input must be 1d ndarray'
        assert len(Y.shape)==1, 'input must be 1d ndarray'
        assert (F1.shape == F2.shape)&(F1.shape == Y.shape)
        eps = 1.e-5
        B = self.bern(F1,F2,n_samp=n_samp)
        B[B<eps]=eps
        B[B>1-eps]=1-eps
        if skip==None:
            skip = np.ones(Y.shape).astype(bool)
        else:
            assert isinstance(skip,np.ndarray)
        llh = np.sum( (Y*np.log(B) + (1-Y)*np.log(1-B))[skip]  )
        return llh

class DescriptiveModel(AbstractModel):
    """
    A descriptive model of the bias
    Bias is deterministic, decision are noisy (probit model)
    p(y=1|df,d1,...,dtau) = phi( df/s - a(d1) - ... - a(dtau) )
    - df = f2 - f1
    - di = distance from f1 to mean of previous trial ( i behind )
    """

    def __init__(self,a,s):
        """
        a: bias function
        s: sensory std
        p(y=1|df,d1,...,dtau) = phi( df/s - a(d1) - ... - a(dtau) )
        - df = f2 - f1
        - di = distance from f1 to mean of previous trial ( i behind )
        """
        assert isinstance(a,list)
        self.a = a
        self.s = s

    def bern(self,f1,f2,n_samp=None):
        """
        Response probability for a sequence of pure tones
        """
        df = f2-f1 # the tone interval in current trial
        alpha = np.zeros(f1.shape)
        for i_a,a_ in enumerate(self.a):
            d = np.zeros(f1.shape)
            d[i_a+1:] = f1[i_a+1:] - 0.5*(f1+f2)[:-(i_a+1)] # distance of f1 to previous trials
            alpha += a_(d)
        return phi(df/self.s-alpha)

    def bernD(self,df,d):
        """
        Response probability for a sequence implicitely described by df, d1,..,dtau
        """
        assert isinstance(d,list)
        alpha = np.zeros(df.shape)
        for a_,d_ in zip(self.a,d):
            alpha += a_(d_)
        return phi(df/self.s-alpha)