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
from GPflow.kernels import Antisymmetric_RBF, RBF, Linear
import time
import os
from scipy.special import erfinv,erf
import pickle
import glob


iphi = lambda x : erfinv(2.*x-1)*np.sqrt(2)
phi = lambda x : .5*(1+erf(x/np.sqrt(2)))




# -------------- Experiment settings
DATA_PATH = '../../data_files/'
OUTPUT_PATH = '../../output/'
timestr = time.strftime("%Y%m%d-%H%M%S")


foldername = OUTPUT_PATH+'TEST_'+timestr
os.makedirs(foldername) # one experiment = one folder


# --------------- Scientific details

experiment = 'continuous'

# SUMMARY of EXPERIMENT

description = 'Summary of Experiment:\n' \
'dataset:'+experiment+'\n'\
'Cropping on s1,s2 close to edges\n' \
'gp prior variance (stationary) fixed to 1\n' \
'antisymmetric kernel\n' \


with open(foldername+"/Readme.txt", "w") as text_file:
    text_file.write(description)

#========================================================================================


mat = sio.loadmat(DATA_PATH + experiment)
s1_all = np.log(np.asarray((mat['s1']))).astype(float)
s2_all = np.log(np.asarray((mat['s2']))).astype(float)
resp_all = np.asarray(mat['resp']) + 0.
acc_all = np.asarray(mat['acc']) + 0.

nsubs,ntrials =acc_all.shape
print 'nsub:', nsubs, '/ntrials:', ntrials, '/total trials:', nsubs*ntrials
# filter data
ranges = {'poor':[0.55,0.8],'good':[0.75,.9]}
typ = 'poor' #typ = 'good'
filt = ranges[typ]
Iacc = np.where((acc_all.mean(1)>filt[0]) & (acc_all.mean(1)<filt[1]))[0]
s1,s2 = s1_all[Iacc,:], s2_all[Iacc,:]
resp,acc = resp_all[Iacc,:], acc_all[Iacc,:]

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


X = np.hstack([diff,prev1,prev2,inf,resp1])
cov_names = ['$df$', '$d_1$','$d_2$', '$d_{\infty}$', '$Y_{t-1}$']
cov_to_index = {s:i for i,s in enumerate(cov_names)}

# further filtering

th = np.sort(np.abs(diff))[len(diff)/1.1]
Itrial = np.where((np.abs(prev1)<0.8)&(np.abs(diff)<th))[0]# subselecting

X,Y = X[Itrial,:],Y[Itrial,:]
print X.shape,Y.shape



assert X.shape[1] == len(cov_names)

print Y.shape,X.shape

#-------------------------------------------------------------
# Declaring model structure

#------------subsampling
print 'WARNING: subsampling for speed up!'
N_eff = -1
if N_eff >-1:
    X = X[:N_eff,:]
    Y = Y[:N_eff,:]
N,D = X.shape
print X.shape,Y.shape

#-------------------------------------------------------------
# Declaring models parameters


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
    m.kerns.parameterized_list[0].variance.fixed = True
    return m

def model2():
    # additive structure
    f_indices = [[0],[1]]
    # Inducing point locations
    Nz =30
    Z = [np.array([[1]]) ,
         np.expand_dims( X[np.random.permutation(N)[:Nz],1],1) ]
    # Setting kernels
    ks = [Linear(1),
          Antisymmetric_RBF(1,lengthscales=.1,variance=1.)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1)',q_diag=True)
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[1].variance.fixed = True
    m.kerns.parameterized_list[0].variance.fixed = True
    return m

def model3():
    # additive structure
    f_indices = [[0],[3]]
    # Inducing point locations
    Nz =30
    Z = [np.array([[1]]) ,
         np.expand_dims( X[np.random.permutation(N)[:Nz],3],1) ]
    # Setting kernels
    ks = [Linear(1),
          Antisymmetric_RBF(1,lengthscales=.1,variance=1.)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(dinf)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[1].variance.fixed = True
    m.kerns.parameterized_list[0].variance.fixed = True
    return m

def model4():
    # additive structure
    f_indices = [[0],[1],[3]]
    # Inducing point locations
    Nz =20
    Z = [np.array([[1]]) ,
         np.expand_dims( X[np.random.permutation(N)[:Nz],1],1),
         np.expand_dims( X[np.random.permutation(N)[:Nz],3],1)]
    # Setting kernels
    ks = [Linear(1),
          Antisymmetric_RBF(1,lengthscales=.1,variance=1.),
          Antisymmetric_RBF(1,lengthscales=.1,variance=1.)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1)+f(dinf)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[0].variance.fixed = True
    m.kerns.parameterized_list[1].variance.fixed = True
    m.kerns.parameterized_list[2].variance.fixed = True
    return m

def model5():
    # additive structure
    f_indices = [[0],[1,2]]
    # Inducing point locations
    Nz =20
    Z = [np.array([[1]]) ,
         X[np.random.permutation(N)[:Nz],[1,3]] ]
    print 'Z:',Z[1].shape
    # Setting kernels
    ks = [Linear(1),
          RBF(2,lengthscales=.1,variance=1.)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1,dinf)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[0].variance.fixed = True
    m.kerns.parameterized_list[1].variance.fixed = True
    return m

def model6():
    # additive structure
    f_indices = [[0],[1],[2],[3]]
    # Inducing point locations
    Nz = 20
    Z = [np.array([[1]]) ,
         np.expand_dims( X[np.random.permutation(N)[:Nz],1],1),
         np.expand_dims( X[np.random.permutation(N)[:Nz],2],1),
         np.expand_dims( X[np.random.permutation(N)[:Nz],3],1),
]
    # Setting kernels
    ks = [Linear(1),
          Antisymmetric_RBF(1,lengthscales=.1,variance=1.),
          Antisymmetric_RBF(1,lengthscales=.1,variance=1.),
          Antisymmetric_RBF(1,lengthscales=.1,variance=1.) ]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1)+f(d2)+f(dinf)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[0].variance.fixed = True
    m.kerns.parameterized_list[1].variance.fixed = True
    m.kerns.parameterized_list[2].variance.fixed = True
    m.kerns.parameterized_list[3].variance.fixed = True

    return m

def model(i):
    return [model1,model2,model3,model4,model5,model6][i-1]

#-------------------------------------------------------------
# Selecting  models
ms = [model(i)() for i in [1]]
model_names = [m.name for m in ms]
n_models = len(ms)
i_models = np.arange(n_models)

#-------------------------------------------------------------
# Running optimization

# optimize until convergence
for m_ in ms:
    success = False
    while success == False:
        res = m_.optimize()
        success = res['success']

# saving output of model fit:
out = {}
for m in ms:
    prms = m.extract_params()
    out[m.name]= prms
    out[m.name]['f_indices'] = m.f_indices

# saving fitted models
pickle.dump(out,open(foldername+'/models.p','wb'))


#=================================================================


# computing the individual function values
M_fs,V_fs = [],[]
for m in ms:
    try:
        mu_fs,v_fs = m.predict_fs(X)
    except Exception:
        mu_fs,v_fs=np.zeros(Y.shape),np.zeros(Y.shape)
    M_fs.append(mu_fs)
    V_fs.append(v_fs)
    print mu_fs.shape

predictions = {
    'X':X,
    'f_indices':[m.f_indices for m in ms],
    'mean':M_fs,
    'variance':V_fs
}
pickle.dump(predictions,open(foldername+'/predictions.p','wb'))


#=====================  Runnning crossvalidation ============================================

m_aucs,std_aucs = [],[]
aucs={}
for i,m in enumerate(ms):
    AUCS = xvalidate(model(i+1),X,Y)
    m_aucs.append(np.mean(AUCS))
    std_aucs.append(np.std(AUCS))
    aucs[m.name] = AUCS
pickle.dump(aucs,open(foldername+'/aucs.p','wb'))
assert False
#=======================================


ind = np.arange(n_models)
fig,ax = plt.subplots()
ax.bar(i_models,m_aucs,1.,yerr=std_aucs)
ax.set_xlabel('model')
ax.set_ylabel('AUC')
ax.set_xticks(i_models)
ax.set_xticklabels(model_names,fontdict=20,rotation=45.)
fig.tight_layout()
plt.savefig('AUC_'+timestr+'.svg')

#-------------------------------------------------------------

# plotting diagnosis predicted output binned vs actual output
w = 4.
l = np.max([len(ms),2])
nc = np.min([4,n_models])
nr =int( np.ceil(float(n_models)/4))
for mod in ['probit','']:
    fig, axarr = plt.subplots(nr,nc,figsize=(nc*w,nr*w))
    axs = [axarr] if len(ms)==1 else axarr.ravel()

    try:
        for i,m in enumerate(ms):
            mu_y,v_y = m.predict_y(X)
            ax = axs[i]
            plot_prediction_accuracy(mu_y,Y,bins=20,ax=ax,mod=mod)
            ax.set_title(model_names[i])
        fig.tight_layout()
        plt.savefig(mod+'pred_vs_y'+timestr+'.svg')
    except Exception:
        pass


#=================================================================

# comparing models based on model evidence
llhs = []
for m in ms:
    try:
        llh = m.lower_bound_likelihood()
    except Exception:
        llh = 0
    llhs.append(llh)
plot_model_comparison(llhs, model_names)
plt.close()



y_min , ymax = -1,1
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
ax.bar(i_models,mu_alpha,1.,yerr=s_alpha)
ax.set_xlabel('model')
ax.set_ylabel('alpha')
ax.set_xticks(i_models)
ax.set_xticklabels(model_names,fontdict=20,rotation=45.)
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
            ax2d.colorbar()

    # plotting the decisions isolines
    ma = mu_alpha[i_m]
    phiv = np.array([.1,.2,.4,.6,.8,.9])
    xv = iphi(phiv)
#    ax1d.hlines(xv,-1,1,linestyles='--',alpha=.5)
#    for phiv_,xv_ in zip(phiv,xv):
#        ax1d.annotate(str(phiv_), xy=(.9,xv_),fontsize=20)
    # overlay of subject performance
    ax1d.set_ylim([y_min,y_max])


    ax1dbis = ax1d.twinx()
    ax1d.grid()
    ax1d.set_ylabel('bias magnitude',fontsize=20)
    ax1dbis.set_ylabel('$d_f $',fontsize=20)
    ax1dbis.plot(diff/mu_alpha,diff,alpha=0)

    handles1d, labels1d = ax1d.get_legend_handles_labels()
    ax1d.legend(handles1d, labels1d)
    ax1d.set_title(typ+'/'+m.name)

    print labels1d

    fig.tight_layout()
    plt.savefig(m.name+'-'+typ+'_'+timestr+'.svg')
    plt.close()
