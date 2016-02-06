import numpy as np
from abc import abstractmethod
from utils import phi
from numpy import exp, log
from matplotlib import pyplot as plt
from scipy.optimize import minimize

sig = lambda x : 1./(1+np.exp(-x))
isig = lambda x : -np.log(1./x-1)

map_r_ab = lambda x,a,b : a + (b-a)*sig(x)
map_ab_r = lambda x,a,b : isig( (x-a)/(b-a) )

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
    p(y=1|x0,x1,...,xtau) = lik( x0*p_lin - a(x1) - ... - a(xtau) )
    """

    def __init__(self,a,p_lin,lik):
        """
        a: bias functions
        p_lin: linear parameter
        lik: likelihood function
        p(y=1|x0,x1,...,xtau) = lik( x0*p_lin - a1(x1) - ... - atau(xtau) )
        """
        assert isinstance(a,list)
        self.a = a
        self.p_lin = p_lin
        self.lik = lik

    def bern(self,x):
        """
        Response probability p(y=1|x)
        """
        T = x.shape[1]-1
        alpha = np.zeros(x.shape[0])
        for i_a in range(T): # iterate over lag functions
            alpha += self.a[i_a](x[:,i_a+1])

        return self.lik(x[:,0]*self.p_lin-alpha)

    def plot_functions(self,ax):
        xp = np.linspace(-1,1,200)
        for i_a,a_ in enumerate(self.a):
            ax.plot(xp,a_(xp),label='lag:'+str(i_a))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        return ax


class LinAdditiveModel(AdditiveModel):
    """
    A descriptive model of the bias
    Bias is deterministic, decision are noisy (probit model)
    p(y=1|x0,x1,...,xtau) = phi( x0*p_lin - a(x1) - ... - a(xtau) )
    where a :x = p_add[0]*x*exp(-|x|*p_add[1])
    """

    def __init__(self,prm_add=[],prm_lin=[],prm_lik=[]):
        """ x : p[0]*x
        Constructor builds and store the biasing functions from input parameters
        """
        self.prm_add, self.prm_lin, self.prm_lik = prm_add,prm_lin,prm_lik
        self.bias = lambda w : lambda x : w*x
        self.lik = lambda l : lambda x : l/2+(1-l)*phi(x)
        a = [self.bias(p) for p in prm_add]
        super(LinAdditiveModel,self).__init__(a,prm_lin,self.lik(prm_lik))

    def param_map(self,prm_add,prm_lin,prm_lik):
        prm_add = [ [exp(p[0])] for p in prm_add ]
        prm_lik = 0.5*sig(prm_lik)
        return prm_add,prm_lin,prm_lik

    def param_imap(self,prm_add,prm_lin,prm_lik):
        prm_add = [ [log(p[0])] for p in prm_add ]
        prm_lik = isig(2*prm_lik)
        return prm_add,prm_lin,prm_lik

    def flat_prm(self,p_add,p_lin,p_lik):
        return np.concatenate([np.array(p_add).flatten(),[p_lin],[p_lik]])

    def unflat_param(self,w):
        t = len(w)-2 # number of lags
        p_add = [ [w[l]]  for l in range(t)]
        p_lin = w[-2]
        p_lik = w[-1]
        return p_add,p_lin,p_lik

    def smart_init(self,T):
        prm_add = [[1] for i in range(T)]
        prm_lin = 1/.03
        prm_lik = 0.01
        prm_add,prm_lin,prm_lik = self.param_imap(prm_add,prm_lin,prm_lik)
        return self.flat_prm(prm_add,prm_lin,prm_lik)

    def set_params_uc(self,prm_add=[],prm_lin=[],prm_lik=[]):
        prm_add,prm_lin,prm_lik = self.param_map(prm_add,prm_lin,prm_lik)
        self.__init__(prm_add,prm_lin,prm_lik)


class LinExpAdditiveModel(AdditiveModel):
    """
    A descriptive model of the bias
    Bias is deterministic, decision are noisy (probit model)
    p(y=1|x0,x1,...,xtau) = phi( x0*p_lin - a(x1) - ... - a(xtau) )
    where a :x = p_add[0]*x*exp(-|x|*p_add[1])
    """

    def __init__(self,prm_add=[],prm_lin=[],prm_lik=[]):
        """ x : p[0]*x*exp(-|x|/p[1])
        Constructor builds and store the biasing functions from input parameters
        """
        self.prm_add, self.prm_lin, self.prm_lik = prm_add,prm_lin,prm_lik
        self.bias = lambda w : lambda x : w[0]*x*exp(-abs(x)*w[1])
        self.lik = lambda l : lambda x : l/2+(1-l)*phi(x)
        a = [self.bias(p) for p in prm_add]
        super(LinExpAdditiveModel,self).__init__(a,prm_lin,self.lik(prm_lik))

    def param_map(self,prm_add,prm_lin,prm_lik):
        a,b = 50,.1
        prm_add = [[exp(p[0]),map_r_ab(p[1],a,b)] for p in prm_add ]
        prm_lik = 0.5*sig(prm_lik)
        return prm_add,prm_lin,prm_lik

    def param_imap(self,prm_add,prm_lin,prm_lik):
        a,b = 50,.1
        prm_add = [ [log(p[0]),map_ab_r(p[1],a,b)] for p in prm_add ]
        prm_lik = isig(2*prm_lik)
        return prm_add,prm_lin,prm_lik


    def set_params_uc(self,prm_add=[],prm_lin=[],prm_lik=[]):
        prm_add,prm_lin,prm_lik = self.param_map(prm_add,prm_lin,prm_lik)
        self.__init__(prm_add,prm_lin,prm_lik)

    def flat_prm(self,p_add,p_lin,p_lik):
        return np.concatenate([np.array(p_add).flatten(),[p_lin],[p_lik]])

    def unflat_param(self,w):
        t = len(w)/2-1 # number of lags
        p_add = [[w[2*l],w[2*l+1]] for l in range(t)]
        p_lin = w[-2]
        p_lik = w[-1]
        return p_add,p_lin,p_lik

    def smart_init(self,T):
        """
        Setting of parameter giving realistic biasing functions for the observed bias so far in the wide range discrimination task
        :param T:
        :return:
        """
        prm_add = [[0.2,1/0.5] for i in range(T)]
        prm_lin = 1/.03
        prm_lik = 0.05
        prm_add,prm_lin,prm_lik = self.param_imap(prm_add,prm_lin,prm_lik)
        return self.flat_prm(prm_add,prm_lin,prm_lik)



def ml_fit(x,y,w0,mod,prop_bstrap=0.9,n_rep=10,n_restart=5,s_noise=0.1,tol=1e-6):
    '''
    Optimizing additive model parameters for classification
    :param x: covariates
    :param y: binary decisions
    :param w0: initialization
    :param mod: the model instance (AdditiveModel)
    :param prop_bstrap: proportion of trials on which fit is performed
    :param n_rep: number of bootstrap results
    :param n_restart: number of restart of the initialization
    :param s_noise: std of noise added on initialization parameters
    :param tol: tolerance on optimization
    :return: array of the argmax of the optimization
    '''

    def bern(w,x,mod):
        p_add,p_lin,p_lik = mod.unflat_param(w)
        mod.set_params_uc(p_add,p_lin,p_lik)
        b = mod.bern(x)
        eps = 1e-3
        b[np.where(b>1-eps)]=1-eps
        b[np.where(b<eps)]=eps
        return b

    def llh(w,x,y,mod):
        b = bern(w,x,mod)
        return np.sum( (y*np.log(b) + (1-y)*np.log(1-b))  )

    res = []
    n = x.shape[0]
    n_bstrap = int(prop_bstrap*n)
    for i in range(n_rep):
        if i%10==0: print i,
        I = np.random.permutation(n)[:n_bstrap]
        x_ = x[I,:]
        y_ = y[I]
        err = lambda w : -llh(w,x_,y_,mod)
        #==================
        temp_r = []
        lik = []
        for k in range(n_restart):
            w0_ = w0 + np.random.randn(w0.shape[0])*s_noise
            r = minimize(err,w0_,tol=tol)
            temp_r.append(r)
            lik.append(r.fun)
        i_min = np.argmin(lik)
        print lik
        r_best = temp_r[i_min]
        #==================
        res.append(r_best.x)
    print 'done'
    res = np.vstack(res)
    return res



if __name__ == '__main__':

    mod = LinExpAdditiveModel()
    #mod = LinAdditiveModel()
    w = mod.smart_init(3)
    p1,p2,p3 = mod.unflat_param(w)
    mod.set_params_uc(p1,p2,p3)

    print mod.a
    print p1,p2,p3
    fig,ax = plt.subplots()
    mod.plot_functions(ax)
    plt.show()



