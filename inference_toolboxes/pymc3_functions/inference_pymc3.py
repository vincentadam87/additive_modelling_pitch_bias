import theano.tensor as tsr
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt

'================ Inference ===================='
def phi(x,mu=0,sd=1):
    return 0.5 * (1 + tsr.erf((x - mu) / (sd * tsr.sqrt(2))))

class additive_model_lin_exp(object):
    """
    Additive function: w*t*exp(-|t|*d)
    """

    def __init__(self):
        self.bias =  lambda w,dec : lambda d : w*(d)*tsr.exp(-np.abs(d)/dec)
        self.bias_np = lambda w,dec : lambda d : w*(d)*np.exp(-np.abs(d)/dec)

    def fit(self,x, y, mcmc_samples=1000):
        t = x.shape[0]-1  # number of additive components
        varnames = ['xc','w','decay','sigma','b','lam']

        with pm.Model() as model:
            # Priors for additive predictor
            w = pm.Normal('w', mu=0, sd=2000, shape=t)
            decay = pm.HalfNormal('decay', sd=200, shape=t)
            # Prior for likelihood
            sigma = pm.Uniform('sigma', 0, 0.3)
            b = pm.Normal('b', mu=0, sd=20)
            lam = pm.Uniform('lam', 0,0.3)

            # Building linear predictor
            lin_pred=0
            for ii in range(1,t+1):
                lin_pred += self.bias(w[ii-1],decay[ii-1])(x[ii,:])

            phi2 = pm.Deterministic('phi2', 0.5*lam + (1-lam)*phi(b + lin_pred + x[0,:]/sigma))
            y = pm.Bernoulli('y', p=phi2, observed=y)

        with model:
            # Inference
            start = pm.find_MAP() # Find starting value by optimization
            print("MAP found:")
            # step = pm.NUTS(scaling = start)
            step = pm.Slice()
            trace = pm.sample(mcmc_samples,
                          step,
                          start=start,
                          progressbar=True) # draw posterior samples

        return trace, model

    def posterior_summary(self,trace):

        w = trace['w']
        decay = trace['decay']
        samples,t = w.shape

        # constructing functions from posterior
        nplot = 100
        xp = np.linspace(-1.5,1.5,nplot)

        m = np.zeros((t,nplot)) # mean
        s = np.zeros((t,nplot)) # std

        for lag in range(t):
            f_post = np.empty((samples,nplot))
            for i_s,w_,d_,xc_ in zip(range(samples),w[:,lag],decay[:,lag]):
                f_post[i_s,:] = self.bias_np(w_,d_,xc_)(xp)
            m[lag,:] = f_post.mean(axis=0)
            s[lag,:] = f_post.std(axis=0)

        return xp,m,s



