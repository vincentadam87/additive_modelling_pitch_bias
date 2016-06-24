import numpy as np
from scipy.special import erf
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import GPflow.kernels
from GPflow.likelihoods import Bernoulli, Gaussian
from GPflow.svgp import SVGP
from GPflow.svgp_additive import SVGP_additive2
from GPflow.kernels import Kern
import tensorflow as tf
import time

import numpy as np
from sklearn import metrics



from scipy.special import erfinv
iphi = lambda x : erfinv(2.*x-1)*np.sqrt(2)



def gaussian_additive_2d(X,Y,D):
    # ============================= 2D ================================
    #Inducing point locations
    Z = [np.array([[1]])] # one pseudo input for linear term
    Z+= [np.random.rand(30, 2)-.5] # (M,1) and (M,2) array


    # Setting likelihood
    likelihood = Bernoulli()

    # Setting kernels
    ks = [GPflow.kernels.Linear(1)]
    ks += [ Antisymmetric(GPflow.kernels.RBF(2)) ]

    f_indices=[[0],[1,2]] # covariate indices used by each function in additive decomposition
    n_func = len(f_indices)

    # Declaring model
    m = SVGP_additive2(X, Y, ks, likelihood, Z,f_indices=f_indices)
    m.Z[0].fixed = True # no need to optimize location for linear parameter

    for k in range(2):
        m.optimize()

    # computing predicted sum (mean and variance)
    Yp, Vp = m.predict_f(X)

    # Generating predictions for individual functions
    Ys=[]
    Vs=[]
    for c in range(n_func):
        m.set_prediction_subset_ds([c])
        Yd, Vd = m.predict_f(X)
        Ys.append(Yd)
        Vs.append(Vd)

    return m, n_func, f_indices, Ys, Vs


def gaussian_additive_1d(X,Y):
    # ============================ 1D ================================
    D = 3

    Nz = 30
    Z = [np.array([[1]])] # one pseudo input for linear term
    for d in range(D-1):
        Z+= [np.random.rand(Nz, 1)-.5] # list of (M,1) array

    # Setting likelihood
    likelihood = Bernoulli()

    # Setting kernels
    ks = [GPflow.kernels.Linear(1)]
    for k in range(1,D):
        ks += [ Antisymmetric(GPflow.kernels.RBF(1))]


    # Declaring model
    m = SVGP_additive2(X, Y, ks, likelihood, Z)
    m.Z[0].fixed = True # no need to optimize location for linear parameter

    # --- Kernel parameters
    for k in m.kerns.parameterized_list:
        if k.name == 'linear':
           k.variance.fixed = True
        if k.name == 'rbf':
           k.variance.fixed = True
           # k.lengthscales.fixed = True
        pass

    for k in range(2):
        m.optimize()

    # Generating predictions for individual functions
    Ys=[]
    Vs=[]
    for d in range(D):
        m.set_prediction_subset_ds([d])
        Yd, Vd = m.predict_f(X)
        Ys.append(Yd)
        Vs.append(Vd)

    return m, Ys, Vs


OUTPUT_PATH = '/home/dell/Python_outputs/'

def plot_1d(X,Ys,Vs,D,labels):
    col=cm.rainbow(np.linspace(0,1,D))
    fig1,ax1 = plt.subplots()
    w = 5
    fig2,ax2 = plt.subplots()

    # plotting infered functions against true functions
    for d in range(1):
        Yd = Ys[d]
        Vd = Vs[d]
        o = np.argsort(X[:,d])
    #     ax1.plot(X[o,d],Fs[o,d],'--',linewidth=4,c=col[d])
        ax1.plot(X[o,d],Yd[o],'-',c=col[d])
        ax1.fill_between(X[o,d],
                         y1=np.squeeze(Yd[o]+np.sqrt(Vd[o])),
                         y2=np.squeeze(Yd[o]-np.sqrt(Vd[o])),facecolor=col[d],alpha=.5)

    for d in range(1,D):
        Yd = Ys[d]
        Vd = Vs[d]
        o = np.argsort(X[:,d])
    #     ax1.plot(X[o,d],Fs[o,d],'--',linewidth=4,c=col[d])
        ax2.plot(X[o,d],Yd[o],'-',c=col[d])
        ax2.fill_between(X[o,d],
                         y1=np.squeeze(Yd[o]+np.sqrt(Vd[o])),
                         y2=np.squeeze(Yd[o]-np.sqrt(Vd[o])),facecolor=col[d],alpha=.5)

        ax1.set_xlabel(labels[0][1],fontsize=20)
        ax1.set_ylabel(labels[0][2],fontsize=20)
        ax1.set_title(labels[0][0],fontsize=20)

        ax2.set_xlabel(labels[1][1],fontsize=20)
        ax2.set_ylabel(labels[1][2],fontsize=20)
        ax2.set_title(labels[1][0],fontsize=20)

    # plt.show()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(OUTPUT_PATH+timestr+'.svg')
    plt.close()

def plot_2d(X, n_func, f_indices, Ys,Vs,D,labels):
    col=cm.rainbow(np.linspace(0,1,3))
    w = 5

    # plotting infered functions against true functions
    for c in range(n_func):
        Yd = Ys[c]
        Vd = Vs[c]

        if len(f_indices[c])==1:
            fig1,ax1 = plt.subplots()
            d = f_indices[c][0]
            o = np.argsort(X[:,d])
            ax1.plot(X[o,d],Yd[o],'-',c=col[d])
            ax1.fill_between(X[o,d],
                             y1=np.squeeze(Yd[o]+np.sqrt(Vd[o])),
                             y2=np.squeeze(Yd[o]-np.sqrt(Vd[o])),facecolor=col[d],alpha=.5)
            ax1.set_xlabel('$x$ ',fontsize=20)
            ax1.set_ylabel('$\\alpha x$ ',fontsize=20)
            # plt.show()

            timestr = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(OUTPUT_PATH+timestr+'.svg')
            plt.close()
        elif len(f_indices[c])==2:
            fig1,ax1 = plt.subplots()
            ax1.scatter(X[o,f_indices[c][1]],
                        X[o,f_indices[c][0]],
                        c=Yd[o])
            print Yd.shape
            ax1.set_xlabel(labels[1],fontsize=20)
            ax1.set_ylabel(labels[2],fontsize=20)
            ax1.set_title(labels[0],fontsize=20)
            # plt.show()

            timestr = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(OUTPUT_PATH+timestr+'.svg')
            plt.close()


def plot_prediction_accuracy(mu_y,Y,bins=5, ax=None,mod=None):
    '''
    Plotting prediction accuracy
    :param mu_y: for all trials, predicted bernoulli parameter
    :param Y: response for all trials
    :return:
    '''

    # Binning predictions
    nbin,edges =np.histogram(mu_y,bins=bins)

    I = np.argsort(mu_y.flat)
    i_bins = [l for l in np.array_split(I, bins)]
    Y_b = [Y[i] for i in i_bins]
    mu_y_b = [mu_y[i] for i in i_bins]

    xs = np.array([m.mean() for m in mu_y_b])
    ys = np.array([y.mean() for y in Y_b])

    if mod == 'probit':
        print 'probit!'
        xs,ys = iphi(xs),iphi(ys)
        xlabel = '$\sum f_i$'
        ylabel = '$\phi^{-1}(<Y>)$'
    else:
        xlabel = '$\phi(\sum f_i)$'
        ylabel = '$<Y>$'

    ax.plot(xs,ys,'x')
    err = np.array(ys)*(1-np.array(ys))/np.sqrt(np.array(nbin))
    ax.errorbar(xs,ys,yerr=err)
    ax.plot(xs,xs,'-')
    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)

    for i in range(bins):
        ax.annotate(str(len(i_bins[i])), xy=(xs[i], xs[i]),fontsize=5)


def plot_model_comparison(LB,names = None):
    """
    :param LB: list or array of marginal log likelihood
    :param names: name of models
    :return:
    """
    n_mods = len(LB) # number of models
    if isinstance(LB,list):
        LB = np.asarray(LB)
    if names == None:
        names = ['m'+str(i) for i in range(n_mods)]

    assert len(names)==n_mods

    LB -= np.max(LB)
    LB = np.exp(LB)
    LB /= LB.sum()

    ind = np.arange(n_mods)
    fig,ax = plt.subplots()
    w = 1.
    ax.bar(ind,LB,w)
    ax.set_xticks(ind)
    ax.set_xticklabels(names,fontdict=20)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    ax.set_xticks(ind+w*.5)
    ax.set_xticklabels(names, rotation=45)
    ax.set_title('model comparison')
    fig.tight_layout()

    plt.savefig('model_comparison_'+timestr+'.svg')
    plt.close()


def structure_from_indices(f_indices):
    '''
    Returns a string
    :param f_indices: list of list
    :return:
    '''
    assert isinstance(f_indices,list)
    for c in f_indices:
        assert isinstance(c,list)
    s = ''
    for c in f_indices:
        ss = ''
        for i in c:
            ss+='x_'+str(i)+','
        s+= 'f('+ss[:-1]+')+'
    return s[:-1]

def xvalidate(model,X,Y):

    XVAL = []
    n = len(Y)
    n_train = int(n*.7)

    while len(XVAL)<10:

        try:
            I = np.random.permutation(n)
            I_train, I_test = I[:n_train],I[n_train:]
            X_train,Y_train = X[I_train,:],Y[I_train,:]
            X_test,Y_test = X[I_test,:],Y[I_test,:]
            m = model()
            m.X = X_train
            m.Y = Y_train
            res = m.optimize()
            mu_test,_ = m.predict_y(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(Y_test.flatten(),
                                                 mu_test.flatten(), pos_label=1)


            XVAL.append([Y_test,mu_test])
        except:
            pass

    return XVAL
