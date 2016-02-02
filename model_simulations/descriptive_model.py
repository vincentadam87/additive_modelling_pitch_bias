import numpy as np
from abc import abstractmethod
from utils import phi
from numpy import exp
from matplotlib import pyplot as plt

class AbstractModel(object):
    '''
    Abstract class for models with bernoulli likelihood and a predictor
    p(y=1|x) = g(f(x))
    '''

    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def bern(self,X):
        ''' bern(x)= g(f(x)) '''
        raise NotImplementedError()

    def llh(self,X,Y,skip=None):
        ''' p(y|x) = g(f(x))^y*(1-g(f(x)))^(1-y) '''
        assert len(Y.shape)==1, 'input must be 1d ndarray'
        assert (X.shape[0] == Y.shape[0])
        eps = 1.e-5
        B = self.bern(X)
        B[B<eps]=eps
        B[B>1-eps]=1-eps
        if skip==None:
            skip = np.ones(Y.shape).astype(bool)
        else:
            assert isinstance(skip,np.ndarray)
        llh = np.sum( (Y*np.log(B) + (1-Y)*np.log(1-B))[skip]  )
        return llh

class AdditiveModel(AbstractModel):
    """
    A descriptive model of the bias
    Bias is deterministic, decision are noisy (probit model)
    p(y=1|x0,x1,...,xtau) = lik( x0/p_lin - a(x1) - ... - a(xtau) )
    """

    def __init__(self,a,p_lin,lik):
        """
        a: bias functions
        p_lin: linear parameter
        lik: likelihood function
        p(y=1|x0,x1,...,xtau) = lik( x0/p_lin - a1(x1) - ... - atau(xtau) )
        """
        assert isinstance(a,list)
        self.a = a
        self.p_lin = p_lin
        self.lik = lik

    def bern(self,x):
        """
        Response probability p(y=1|x)
        """
        alpha = np.zeros(x.shape[0])
        for i_a,a_ in enumerate(self.a): # iterate over lag functions
            alpha += a_(x[:,i_a+1])
        return self.lik(x[:,0]*self.p_lin-alpha)

class LinExpAdditiveModel(AdditiveModel):
    """
    A descriptive model of the bias
    Bias is deterministic, decision are noisy (probit model)
    p(y=1|x0,x1,...,xtau) = phi( x0/p_lin - a(x1) - ... - a(xtau) )
    where a :x = p_add[0]*x*exp(-|x|/p_add[1])
    """

    def __init__(self,prm_add,prm_lin,prm_lik):
        """ x : p[0]*x*exp(-|x|/p[1]) """
        self.bias = lambda w,d : lambda x : w*x*exp(-abs(x)*d)
        self.lik = lambda l : lambda x : l/2+(1-l)*phi(x)
        a = [self.bias(p[0],p[1]) for p in prm_add]
        super(LinExpAdditiveModel,self).__init__(a,prm_lin,self.lik(prm_lik))



if __name__ == '__main__':

    # Declare model parameters
    prm_add = [[1,1],[0.5,2],[0.1,0.1]] # parameters of additive functions
    prm_lin = .1 # linear parameter
    prm_lik = .1 # parameter of the likelihood

    # Create model instance
    model = LinExpAdditiveModel(prm_add,prm_lin,prm_lik)

    # plotting functions
    xp = np.linspace(-5,5,100)
    for a in model.a:
        plt.plot(xp,a(xp))
    plt.show()

    # sampling
    n = 1000
    d = len(prm_add)
    x = (2*np.random.rand(n,d+1)-1)*4
    b = model.bern(x)
    y = np.random.rand(n)<b

    # compute likelihood
    print model.llh(x,y)



