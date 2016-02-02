from scipy.optimize import minimize
from model_simulations.descriptive_model import *
from utils import *
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
# ---------------------- Additive Linear Exp ---------------------

bias = lambda w,d : lambda x : w*x*exp(-abs(x)/d)
lik = lambda l : lambda x : l/2+(1-l)*phi(x)


# Parameters:  lambda, sigma, additive pairs


def flat_prm(p_add,p_lin,p_lik):
    return np.concatenate([np.array(p_add).flatten(),[p_lin],[p_lik]])

def unflat_param(w):
    t = len(w)/2-1 # number of lags
    p_add = [[w[2*l],w[2*l+1]] for l in range(t)]
    p_lin = w[-2]
    p_lik = w[-1]
    return p_add,p_lin,p_lik

def bern(w,x):
    p_add,p_lin,p_lik = unflat_param(w)
    b = LinExpAdditiveModel(p_add,p_lin,p_lik).bern(x)
    eps = 1e-5
    b[b>1-eps]=1-eps
    b[b<eps]=eps
    return b

def llh(w,x,y):
    b = bern(w,x)
    return np.sum( (y*np.log(b) + (1-y)*np.log(1-b))  )


if __name__ == '__main__':


    n =5000
    d = 2

    # declare model

    peaks = [0.1,0.5]
    maxs = np.ones((d,))*1
    prm_add = [[m*1./peak*np.exp(1), 1./peak] for peak, m in zip(peaks,maxs)]

    #prm_add = [[0.5,1],[0.6,2],[0.5,3]] # parameters of additive functions
    prm_lin = 1./0.02 # linear parameter
    prm_lik = .2 # parameter of the likelihood

    w = flat_prm(prm_add,prm_lin,prm_lik)
    mod = LinExpAdditiveModel(prm_add,prm_lin,prm_lik)

    # sampling
    x= 2*np.random.rand(n,d+1)-1

    x[:,0]*=.1
    x[:,1:-1]/=2.
    b = mod.bern(x)
    y = (np.random.rand(n)<b).astype(int)

    plt.figure()
    plt.hist(b)
    plt.savefig('ml-fit0.svg')
    plt.close()

    # learning
    res = []
    n_rep = 40
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

    #-----------------


    fig,axarr = plt.subplots(1,d+1,figsize=[10,5])
    for lag,ax in zip(range(d),axarr.flat):
        ax.scatter(res[:,2*lag],res[:,2*lag+1])
        ax.plot(w[2*lag],w[2*lag+1],'xr')
        #ax.set_xlim([0,2])
        #ax.set_ylim([0,2])
        ax.set_xlabel('w')
        ax.set_ylabel('decay')
        ax.set_title('lag '+str(lag+1) )
    axarr[-1].scatter(res[:,-2],res[:,-1])
    axarr[-1].plot(w[-2],w[-1],'xr')
    #axarr[-1].set_xlim([0,2])
    #axarr[-1].set_ylim([0,.3])
    axarr[-1].set_xlabel('lin')
    axarr[-1].set_ylabel('lik')
    plt.tight_layout()
    plt.savefig('ml-fit1.svg')
    plt.close()




    xmin, xmax = x.min(),x.max()
    xp = np.linspace(xmin,xmax,100)
    col = 'rgb'
    fig, ax = plt.subplots()
    plt.hist(x[:,1:-1].flatten(),bins=100,alpha=0.2,normed=True)


    for i_r,r in enumerate(res):
        p_add,_,_ = unflat_param(r)
        for lag in range(d):
            kwargs = {'label':'lag'+str(lag+1)} if i_r==0 else {}
            ax.plot(xp,mod.bias(p_add[lag][0],prm_add[lag][1])(xp),col[lag],**kwargs)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    for lag,p in zip(range(d),prm_add):
        ax.plot(xp,mod.bias(p[0],p[1])(xp),col[lag],\
                linestyle='--',\
               linewidth=2,path_effects=[path_effects.Normal(),\
                                         path_effects.SimpleLineShadow()])

    plt.savefig('ml-fit2.svg')
    plt.close()


    plt.close('all')