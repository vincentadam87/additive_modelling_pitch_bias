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
from GPflow.kernels import Antisymmetric, RBF, Linear
import time



#==== load data ====

DATA_PATH = '../../data_files/'
OUTPUT_PATH = '../../output/'
experiment = 'continuous'

timestr = time.strftime("%Y%m%d-%H%M%S")


mat = sio.loadmat(DATA_PATH + experiment)
s1_all = np.log(np.asarray((mat['s1']))).astype(float)
s2_all = np.log(np.asarray((mat['s2']))).astype(float)
resp_all = np.asarray(mat['resp']) + 0.
acc_all = np.asarray(mat['acc']) + 0.


# filter data
ranges = {
    'poor':[0.55,0.8],
    'good':[0.9,1]
}


typ = 'poor'
#typ = 'good'

filt = ranges[typ]
I = np.where((acc_all.mean(1)>filt[0]) & (acc_all.mean(1)<filt[1]))[0]
s1 = s1_all[I,:]
s2 = s2_all[I,:]
resp = resp_all[I,:]
acc = acc_all[I,:]

sm = (s1 + s2)*.5
diff = (s1 - s2)[:,1:]
prev = s1[:,1:]-sm[:,:-1]
inf = s1[:,1:] - s1.mean()
Y = resp[:,1:].astype(int)

print Y.shape, inf.shape
a,b = Y.shape
diff = np.reshape(diff,[a*b,1])
prev = np.reshape(prev,[a*b,1])
inf = np.reshape(inf,[a*b,1])
X = np.hstack([diff,prev,inf])
Y = np.reshape(Y,[a*b,1])

print X.shape,Y.shape


#fig,axarr = plt.subplots(3,1)
#axarr[0].hist(X[:,0])
#axarr[1].hist(X[:,1])
#axarr[2].hist(X[:,2])
#plt.show()


#-------------------------------------------------------------
# Declaring model structure

#------------subsampling
print 'WARNING: subsampling for speed up!'

N_eff = -1
if N_eff >-1:
    X = X[:N_eff,:]
    Y = Y[:N_eff,:]

N,D = X.shape

#print 'number of covariates: ',D
#print 'number of trials: ',N
#print 'number of functions:', n_func
#print 'function structure: ' + structure_from_indices(f_indices)

#-------------------------------------------------------------
# Declaring models parameters

#-------------------------------------------------------------
#-------- MODEL 1 --------------------------------------------

# additive structure
f_indices = [[0]]
# Inducing point locations
Z = [np.array([[1]])] # one pseudo input for linear term
# Setting kernels
ks = [Linear(1)]
# Declaring model
m1 = SVGP_additive2(X, Y, ks, Bernoulli(), Z,f_indices=f_indices)
m1.Z[0].fixed = True # no need to optimize location for linear parameter

#-------------------------------------------------------------
#-------- MODEL 2 --------------------------------------------

# additive structure
f_indices = [[0],[1]]
# Inducing point locations
Nz =20
Z = [np.array([[1]]) ,
     np.expand_dims( X[np.random.permutation(N)[:Nz],1],1) ]
# Setting kernels
ks = [Linear(1),
      RBF(1,lengthscales=.1,variance=.5)]
# Declaring model
m2 = SVGP_additive2(X, Y, ks, Bernoulli(), Z,f_indices=f_indices)
m2.Z[0].fixed = True # no need to optimize location for linear parameter
m2.kerns.parameterized_list[1].variance.fixed = True


#-------------------------------------------------------------
#-------- MODEL 3 --------------------------------------------

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
m3 = SVGP_additive2(X, Y, ks, Bernoulli(), Z,f_indices=f_indices)
m3.Z[0].fixed = True # no need to optimize location for linear parameter
m3.kerns.parameterized_list[1].variance.fixed = True
m3.kerns.parameterized_list[2].variance.fixed = True

#-------------------------------------------------------------
#-------- MODEL 4 --------------------------------------------

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
m4 = SVGP_additive2(X, Y, ks, Bernoulli(), Z,f_indices=f_indices)
m4.Z[0].fixed = True # no need to optimize location for linear parameter
m4.kerns.parameterized_list[1].variance.fixed = True


#-------------------------------------------------------------
# Running optimization
ms = [m1,m2,m3,m4]
#ms = [m1,m4]

for m in ms:
    m.optimize()
#-------------------------------------------------------------

# plotting diagnosis predicted output binned vs actual output
w = 4.
fig, axarr = plt.subplots(1,len(ms),figsize=(len(ms)*w,w))
for i,m in enumerate(ms):
    mu_y,v_y = m.predict_y(X)
    ax = axarr[i]
    plot_prediction_accuracy(mu_y,Y,bins=10,ax=ax)
fig.tight_layout()
plt.savefig('pred_vs_y'+timestr+'.svg')


#=================================================================

# comparing models based on model evidence
llhs = []
for m in ms:
    llh = m.lower_bound_likelihood()
    llhs.append(llh)
plot_model_comparison(llhs)

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

#================================

print '-----------------------------------'
for i_m,m in enumerate(ms):
    print 'i_m:',i_m


    n_func = len(m.f_indices)
    col = cm.rainbow(np.linspace(0, 1, n_func))

    fig, ax = plt.subplots()

    fig,ax = plt.subplots()
    for c in range(1,n_func):
        print 'c:',c
        mu_fs = M_fs[i_m][:,c]
        s_fs = np.sqrt(V_fs[i_m][:,c])


        if len(m.f_indices[c])==1:

            d = m.f_indices[c][0]
            o = np.argsort(X[:,d])
            ax.plot(X[o,d],mu_fs[o],c=col[c])
            ax.fill_between(X[o,d],(mu_fs-s_fs)[o],(mu_fs+s_fs)[o],alpha=.3,
                            facecolor=col[c],
                            edgecolor=col[c])

        elif len(m.f_indices[c])==2:
            print X[:,m.f_indices[c][0]].shape
            print mu_fs.shape
            ax.scatter(X[:,m.f_indices[c][0]],X[:,m.f_indices[c][1]],c=mu_fs)
    ax.set_title(str(i_m))
    plt.savefig('functions_nl_model'+str(i_m)+'_'+timestr+'.svg')

