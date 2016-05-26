import numpy as np
from scipy.special import erf
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import GPflow.kernels
from GPflow.likelihoods import Bernoulli, Gaussian
from GPflow.svgp import SVGP
from GPflow.svgp_additive import SVGP_additive2
import scipy.io as sio
from gpflow_functions import *
np.random.seed(0)
import matplotlib.cm as cm
from matplotlib.pyplot import Rectangle
from GPflow.kernels import Antisymmetric, RBF, Linear
import time

#==== load data ====

DATA_PATH = '../../data_files/'
OUTPUT_PATH = '../../output/'
experiment = 'bimodal'

timestr = time.strftime("%Y%m%d-%H%M%S")


mat = sio.loadmat(DATA_PATH + experiment)
s1_all = np.log(np.asarray((mat['s1']))).astype(float)
s2_all = np.log(np.asarray((mat['s2']))).astype(float)
resp_all = np.asarray(mat['resp']) + 0.
acc_all = np.asarray(mat['acc']) + 0.


nsubs,ntrials =acc_all.shape
print 'nsub:', nsubs
print 'ntrials:', ntrials
print 'total trials:', nsubs*ntrials

# filter data
ranges = {
    'poor':[0.55,0.8],
    'good':[0.9,1]
}


typ = 'poor'
#typ = 'good'

filt = ranges[typ]
print 'filt:',filt
print 'subject acc:',acc_all.mean(1)
Iacc = np.where((acc_all.mean(1)>filt[0]) & (acc_all.mean(1)<filt[1]))[0]
print 'Iacc:',Iacc


s1 = s1_all[Iacc,:]
s2 = s2_all[Iacc,:]
resp = resp_all[Iacc,:]
acc = acc_all[Iacc,:]

print acc.shape

lag = 2
sm = (s1 + s2)*.5
diff = (s1 - s2)[:,lag:] # f1 - f2 at trial time
prev1 = s1[:,lag:]-sm[:,1:-lag+1] # f1(t) - (f1+f2)(t-1)
prev2 = s1[:,lag:]-sm[:,:-lag] # f1(t) - (f1+f2)(t-2)
inf = s1[:,lag:] - s1.mean() # f1(t) - overallmean(f1)
Y = resp[:,lag:].astype(int)
resp1 = resp[:,:-lag]*2.-1. # -1,+1




a,b = Y.shape
diff = np.reshape(diff,[a*b,1])
prev1 = np.reshape(prev1,[a*b,1])
prev2 = np.reshape(prev2,[a*b,1])
resp1 = np.reshape(resp1,[a*b,1])
inf = np.reshape(inf,[a*b,1])
Y = np.reshape(Y,[a*b,1])
X = np.hstack([diff,prev1,inf,prev2,resp1])

Itrial = np.where((np.abs(prev1)<0.8))[0]# subselecting

X,Y = X[Itrial,:],Y[Itrial,:]
print X.shape,Y.shape


cov_names = [
    '$df$',
    '$d_1$',
    '$d_{\infty}$',
    '$d_2$',
    '$Y_{t-1}$'
]

assert X.shape[1] == len(cov_names)

print Y.shape,X.shape

#-------------------------------------------------------------
# Declaring model structure

#------------subsampling
print 'WARNING: subsampling for speed up!'

N,D = X.shape
print X.shape,Y.shape
N_eff = -1
if N_eff >-1:
    X = X[:N_eff,:]
    Y = Y[:N_eff,:]

N,D = X.shape
print X.shape,Y.shape


#-------------------------------------------------------------
# Declaring models parameters

#-------------------------------------------------------------
#-------- MODEL 1 --------------------------------------------

def model1():
    # additive structure
    f_indices = [[0]]
    # Inducing point locations
    Z = [np.array([[1]])] # one pseudo input for linear term
    # Setting kernels
    ks = [Linear(1)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'no bias')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    return m

#-------------------------------------------------------------
#-------- MODEL 2 --------------------------------------------


def model2():
    # additive structure
    f_indices = [[0],[1]]
    # Inducing point locations
    Nz =50
    Z = [np.array([[1]]) ,
         np.expand_dims( X[np.random.permutation(N)[:Nz],1],1) ]
    # Setting kernels
    ks = [Linear(1),
          RBF(1,lengthscales=.1,variance=1.)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[1].variance.fixed = True
    return m


#-------------------------------------------------------------
#-------- MODEL 3 --------------------------------------------

def model3():
    # additive structure
    f_indices = [[0],[1],[2]]
    # Inducing point locations
    Nz =20
    Z = [np.array([[1]]) ,
         np.expand_dims( X[np.random.permutation(N)[:Nz],1],1),
         np.expand_dims( X[np.random.permutation(N)[:Nz],2],1)]
    # Setting kernels
    ks = [Linear(1),
          RBF(1,lengthscales=.1,variance=.5),
          RBF(1,lengthscales=.1,variance=.5)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1)+f(dinf)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[1].variance.fixed = True
    m.kerns.parameterized_list[2].variance.fixed = True
    return m

#-------------------------------------------------------------
#-------- MODEL 4 --------------------------------------------

def model4():
    # additive structure
    f_indices = [[0],[1,2]]
    # Inducing point locations
    Nz =20
    Z = [np.array([[1]]) ,
         X[np.random.permutation(N)[:Nz],1:]]
    print 'Z:',Z[1].shape
    # Setting kernels
    ks = [Linear(1),
          RBF(2,lengthscales=.1,variance=.5)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1,dinf)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[1].variance.fixed = True
    return m

#-------------------------------------------------------------
#-------- MODEL 4 --------------------------------------------

def model5():
    # additive structure
    f_indices = [[0],[4]]
    # Inducing point locations
    Z = [np.array([[1]]) ,
         np.array([[1]])]
    print 'Z:',Z[1].shape
    # Setting kernels
    ks = [Linear(1),
          Linear(1)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(resp1)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[1].variance.fixed = True
    return m

def model6():
    # additive structure
    f_indices = [[0],[1],[4]]
    # Inducing point locations
    Nz =20
    Z = [np.array([[1]]) ,
         X[np.random.permutation(N)[:Nz],1:],
         np.array([[1]])]
    print 'Z:',Z[1].shape
    # Setting kernels
    ks = [Linear(1),
          RBF(1,lengthscales=.1,variance=.5),
          Linear(1)]
    # Declaring model

    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1)+f(resp1)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[1].variance.fixed = True
    return m

def model(i):
    return [model1,model2,model3,model4,model5,model6][i-1]


#-------------------------------------------------------------
# Running optimization
ms = [model(i)() for i in [1,2,5,6]]

for m_ in ms:
    m_.optimize()

#=================================================================

m_aucs,std_aucs = [],[]
for i,_ in enumerate(ms):
    m_auc, std_auc = xvalidate(model(i+1),X,Y)
    m_aucs.append(m_auc)
    std_aucs.append(std_auc)

fig,ax = plt.subplots()
ax.bar(np.arange(len(ms)),m_aucs,1.,yerr=std_aucs)
ax.set_xlabel('model')
ax.set_ylabel('AUC')
fig.tight_layout()
plt.savefig('AUC_'+timestr+'.svg')


#-------------------------------------------------------------

# plotting diagnosis predicted output binned vs actual output
w = 4.
l = np.max([len(ms),2])
for mod in ['probit','']:
    fig, axarr = plt.subplots(1,l,figsize=(l*w,w))
    for i,m in enumerate(ms):
        mu_y,v_y = m.predict_y(X)
        ax = axarr[i]
        plot_prediction_accuracy(mu_y,Y,bins=20,ax=ax,mod=mod)
    fig.tight_layout()
    plt.savefig(mod+'pred_vs_y'+timestr+'.svg')


#=================================================================

# comparing models based on model evidence
llhs = []
for m in ms:
    llh = m.lower_bound_likelihood()
    llhs.append(llh)
plot_model_comparison(llhs)
plt.close()

#====================================

# computing the individual function values
M_fs,V_fs = [],[]
for m in ms:
    mu_fs,v_fs = m.predict_fs(X)
    M_fs.append(mu_fs)
    V_fs.append(v_fs)
    print mu_fs.shape
#====================================

# extracting linear parameter (the alpha)
mu_alpha = []
s_alpha = []
for i_m,m in enumerate(ms):
    with m.tf_mode():
        ma_tf = m.q_mu[0]
        sa_tf = m.q_sqrt[0]
        ma,sa = m._session.run([ma_tf,sa_tf],feed_dict={m._free_vars:m.get_free_state()})
        print ma,sa
        mu_alpha.append(np.squeeze(ma))
        s_alpha.append(np.squeeze(sa))
fig,ax = plt.subplots()
ax.bar(np.arange(len(ms)),mu_alpha,1.,yerr=s_alpha)
ax.set_xlabel('model')
ax.set_ylabel('alpha')
fig.tight_layout()
fig.savefig('alpha_'+timestr+'.svg')
plt.close()
#================================

# inferred functions per model

print '-----------------------------------'
for i_m,m in enumerate(ms):
    print 'i_m:',i_m

    n_func = len(m.f_indices)
    col = cm.rainbow(np.linspace(0, 1, n_func))

    fig,axarr = plt.subplots(1,2,figsize=(10,5))
    ax1d = axarr[0]
    ax2d = axarr[1]

    po1d= []
    labels1d = []

    #  ----------- plotting 1d functions -------------

    for c in range(1,n_func):

        print 'c:',c
        mu_fs = M_fs[i_m][:,c]
        s_fs = np.sqrt(V_fs[i_m][:,c])

        if len(m.f_indices[c])==1:

            d = m.f_indices[c][0]
            o = np.argsort(X[:,d])
            ax1d.plot(X[o,d],mu_fs[o],c=col[c],label=cov_names[d])
            ax1d.plot(X[o,d],(mu_fs-s_fs)[o],ls='--',c=col[c])
            ax1d.plot(X[o,d],(mu_fs+s_fs)[o],ls='--',c=col[c])
            #ax1d.fill_between(X[o,d],(mu_fs-s_fs)[o],(mu_fs+s_fs)[o],alpha=.3,
            #                facecolor=col[c],
            #                edgecolor=col[c])


        elif len(m.f_indices[c])==2:
            # this will work for a single 2d function
            print X[:,m.f_indices[c][0]].shape
            print mu_fs.shape
            covs = '/'.join([cov_names[u] for u in m.f_indices[c]])
            ax2d.scatter(X[:,m.f_indices[c][0]],X[:,m.f_indices[c][1]],c=mu_fs)
            ax2d.set_title(covs)
            ax2d.set_xlabel(cov_names[m.f_indices[c][0]])
            ax2d.set_ylabel(cov_names[m.f_indices[c][1]])

    handles1d, labels1d = ax1d.get_legend_handles_labels()
    ax1d.legend(handles1d, labels1d)
    ax1d.set_title(typ+'/'+m.name)

    print labels1d

    plt.savefig(m.name+'-'+typ+'_'+timestr+'.svg')
    plt.close()
