

import GPy
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(1)
from numpy.linalg import inv,solve
from numpy import dot
from GPy.kern._src.add import Add

k1 = GPy.kern.Linear(input_dim=1, active_dims=[0]) # works on the first column of X, index=0
k2 = GPy.kern.ExpQuad(input_dim=1, lengthscale=3, active_dims=[1]) # works on the second column of X, index=1
k3 = GPy.kern.ExpQuad(input_dim=1, lengthscale=3, active_dims=[2]) # works on the second column of X, index=1
k_list = [k1,k2,k3]


def gp_probit_ep(X,Y,k,lik,n_rep=3,max_iters=100):
    """
    EP algorithm for GP-Probit classification
    :param X: covariates N*D
    :param Y: responses N
    :param k: GPy kernel instance
    :return: posterior GP instance (GPy)
    """

    # initialization
    m = GPy.core.GP(X=X,
                Y=Y,
                kernel=k,
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=lik)

    # optimizing  EP
    for i in range(n_rep):
        m.optimize('bfgs', max_iters=max_iters)  #first runs EP and then optimizes the kernel parameters
        print 'iteration:', i,
        print m
        print ""

    return m

def sample_additive_gp(X,k_list, n_samp):
    D = len(k_list)
    f = np.zeros((n_samp,D))
    for d,k in enumerate(k_list):
        f[:,d] = np.random.multivariate_normal(np.zeros(n_samp), k.K(X))
        f[:,d]-=f[:,d].mean()
    return f, f.sum(axis=1)

def additive_gp_probit_ep(X,Y,k_list,lik,n_rep=3,max_iters=100):
    """
    EP algorithm for Additive-GP-Probit classification
    :param X: covariates N*D
    :param Y: responses N
    :param k_list: list of GPy kernel instance
    :return: ??
    """

    N = len(Y)
    D = len(k_list)
    Ks = np.zeros((N,N,D))
    for d in range(D):
        Ks[:,:,d]=k_list[d].K(X)
    ksum = Add(k_list)

    gp_post = gp_probit_ep(X,Y,ksum,lik,n_rep=n_rep,max_iters=max_iters)
    mu_p_sum,K_p_sum = gp_post._raw_predict(X,full_cov=True)
    mu_p,K_p = joint_from_sum(mu_p_sum,K_p_sum,Ks)

    return mu_p,K_p


def joint_from_sum(mu_sum,S_sum,Ks):
    '''
    p(f1,...,fd|Y) from p(f1+...+fd|Y) and p(f1)...p(fd)
    :param mu_sum: posterior mean (N)
    :param S_sum: posterior covariance (N*N)
    :param Ks: prior Covariances (N*N*D)
    :return: mus, Ss (N*D), (N*N*D)
    '''
    N = len(mu_sum)
    D = Ks.shape[2]
    R = S_sum + np.sum(Ks,axis=2)
    V = np.zeros((N,N,D))
    v = np.zeros((N,D))
    for d in range(D):
        V[:,:,d] = Ks[:,:,d] - dot(Ks[:,:,d],solve(R,Ks[:,:,d]))
        v[:,d] = np.squeeze( dot(Ks[:,:,d] - dot(Ks[:,:,d], solve(R,np.sum(Ks,axis=2))) , solve(S_sum,mu_sum)) )
    return v,V




if __name__ == '__main__':

    N = 200
    D = 3
    # covariates
    X = np.random.randn(N,D)

    # linear predictor
    lik = GPy.likelihoods.Bernoulli()
    k1 = GPy.kern.Linear(input_dim=1,variances=1, active_dims=[0]) # works on the first column of X, index=0
    k2 = GPy.kern.ExpQuad(input_dim=1, lengthscale=3, active_dims=[1]) # works on the second column of X, index=1
    k3 = GPy.kern.ExpQuad(input_dim=1, lengthscale=3, active_dims=[2]) # works on the second column of X, index=1
    k_list = [k1,k2,k3]
    ksum = Add(k_list)

    fs,f = sample_additive_gp(X,k_list, N)
    p = lik.gp_link.transf(f) # squash the latent function

    plt.hist(p)
    plt.show()

    Y = lik.samples(f).reshape(-1,1)
    print Y.shape

    for d,fd in enumerate(fs.T):
        print d,fd.shape
        o = np.argsort(X[:,d])
        plt.plot(X[o,d],fd[o])
    plt.show()

    # inference
    mu_p,K_p = additive_gp_probit_ep(X,Y,k_list,lik,n_rep=30,max_iters=100)

    print mu_p.shape, K_p.shape

    for d in range(D):
        o = np.argsort(X[:,d])
        yerr=np.squeeze(K_p[o,o,d])
        print X[o,d].shape, yerr.shape
        plt.fill_between(X[o,d],y1=mu_p[o,d]-yerr,y2=mu_p[o,d]+yerr,alpha=0.3)
        plt.plot(X[o,d],fs[o,d])

    plt.show()