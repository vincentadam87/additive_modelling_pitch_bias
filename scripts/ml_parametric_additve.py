from scipy.optimize import minimize
from model_simulations.descriptive_model import *
from utils import *
from matplotlib import pyplot as plt
# ---------------------- Additive Linear Exp ---------------------

bias = lambda w,d : lambda x : w*x*exp(-abs(x)/d)
lik = lambda l : lambda x : l/2+(1-l)*phi(x)


# Parameters:  lambda, sigma, additive pairs


def flat_prm(prm_add,prm_lin,prm_lik):
    return np.concatenate([np.array(prm_add).flatten(),[prm_lin],[prm_lik]])

def unflat_param(w):
    t = len(w)/2-1 # number of lags
    prm_add = [[w[2*l],w[2*l+1]] for l in range(t)]
    prm_lin = w[-2]
    prm_lik = w[-1]
    return prm_add,prm_lin,prm_lik

def bern(w,x):
    prm_add,prm_lin,prm_lik = unflat_param(w)
    b = LinExpAdditiveModel(prm_add,prm_lin,prm_lik).bern(x)
    eps = 1e-5
    b[b>1-eps]=1-eps
    b[b<eps]=eps
    return b

def llh(w,x,y):
    b = bern(w,x)
    return np.sum( (y*np.log(b) + (1-y)*np.log(1-b))  )


if __name__ == '__main__':


    n = 5000
    d = 2

    # declare model
    prm_add = [[1,1],[1,1]] # parameters of additive functions
    prm_lin = 1 # linear parameter
    prm_lik = .05 # parameter of the likelihood

    w = flat_prm(prm_add,prm_lin,prm_lik)
    mod = LinExpAdditiveModel(prm_add,prm_lin,prm_lik)

    # sampling
    x= np.random.randn(n,d+1)
    b = mod.bern(x)
    y = (np.random.rand(n)<b).astype(int)
    plt.hist(b)
    plt.show()
    #plt.close()

    # learning
    res = []
    n_rep = 100
    prop_bstrap = .9
    n_bstrap = int(prop_bstrap*n)
    for i in range(n_rep):
        if i%10==0: print i,
        I = np.random.permutation(n)[:n_bstrap]
        x_ = x[I,:]
        y_ = y[I]
        w0 = w
        err = lambda w : -llh(w,x_,y_)
        r = minimize(err,w0)
        res.append(r.x)

    print 'done'
    res = np.vstack(res)

    plt.scatter(res[:,-2],res[:,-1])
    plt.plot(w[:,-2],w[:,-1],'xr')
    plt.xlim([0,2])
    plt.ylim([0,.3])
    plt.xlabel('lin')
    plt.ylabel('lik')
    plt.show()

    for lag in range(d):
        plt.scatter(res[:,2*lag],res[:,2*lag+1])
        plt.plot(w[:,2*lag],w[:,2*lag+1],'xr')
        plt.xlim([0,2])
        plt.ylim([0,2])
        plt.xlabel('w')
        plt.ylabel('decay')
        plt.title('lag '+str(d+1) )
        plt.show()

