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

#==== load data ====

DATA_PATH = '../../data_files/'
OUTPUT_PATH = '../../output/'
experiment = 'continuous'

mat = sio.loadmat(DATA_PATH + experiment)
s1_all = np.log(np.asarray((mat['s1']))).astype(float)
s2_all = np.log(np.asarray((mat['s2']))).astype(float)
resp_all = np.asarray(mat['resp']) + 0.
acc_all = np.asarray(mat['acc']) + 0.


# filter data
acc_filter = [[0.55,0.8], [0.8,1]]
for filt in acc_filter:
    poor = np.where((acc_all.mean(1)>filt[0]) & (acc_all.mean(1)<filt[1]))
    s1 = s1_all[poor[0],:]
    s2 = s2_all[poor[0],:]
    resp = resp_all[poor[0],:]
    acc = acc_all[poor[0],:]


    [nSub,nTrials] = s1.shape

    s1_ = s1[:,1:] - s1[:,:-1]
    s2_ = s2[:,1:] - s2[:,:-1]


    s1_inf = s1[:,1:] - np.log(1000)
    s2_inf = s2[:,1:] - np.log(1000)

    resp_ = resp[:,1:].astype(float)

    diff = s1_.flatten() - s2_.flatten()
    prev = 0.5*(s1_.flatten() + s2_.flatten())
    inf = 0.5*(s1_inf.flatten() + s2_inf.flatten())
    resp_ = resp_.flatten()

    select = np.where((np.abs(prev)<0.85))
    diff = diff[select[0]]
    inf = inf[select[0]]
    prev = prev[select[0]]
    resp_ = resp_[select[0]]

    X = np.vstack([diff,prev,inf]).transpose()
    Y = np.random.rand(len(prev.flatten()),1)
    Y[:,0] = resp_.transpose()


#-------------------------------------------------------------
# Declaring model structure

#------------subsampling
print 'WARNING: subsampling for speed up!'
N_eff = 200
X = X[:N_eff,:]
Y = Y[:N_eff,:]

N,D = X.shape
f_indices = [[0],[1],[2]]
n_func = len(f_indices)

print 'number of covariates: ',D
print 'number of trials: ',N
print 'number of functions:', n_func
print 'function structure: ' + structure_from_indices(f_indices)

#-------------------------------------------------------------
# Declaring model parameters

# Inducing point locations
Nz =30
Z = [np.array([[1]])] # one pseudo input for linear term
Z +=  [ np.random.rand(30, 1)-.5 for c in range(1,n_func)]
# Setting kernels
ks = [GPflow.kernels.Linear(1),
      Antisymmetric(GPflow.kernels.RBF(1)),
      Antisymmetric(GPflow.kernels.RBF(1))]
# Declaring model
m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,f_indices=f_indices)
m.Z[0].fixed = True # no need to optimize location for linear parameter
m.kerns.parameterized_list[1].k.variance.fixed = False
m.kerns.parameterized_list[2].k.variance.fixed = True

#-------------------------------------------------------------
# Running optimization
m.optimize()

#-------------------------------------------------------------
# plotting diagnosis
mu_y,v_y = m.predict_y(X)
plot_prediction_accuracy(mu_y,Y)

#-------------------------------------------------------------
# plotting individual functions
mu_fs,v_fs = m.predict_fs(X)

fig,ax = plt.subplots()
for d in range(n_func):
    o = np.argsort(X[:,d])
    ax.plot(X[o,d],mu_fs[o,d])
plt.savefig('functions.svg')