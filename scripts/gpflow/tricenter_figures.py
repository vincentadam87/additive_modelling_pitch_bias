
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

'''
Figure1 for the tricenter meeting

DATASET: Broad range
'''


DATA_PATH = '../../data_files/'
OUTPUT_PATH = '../../output/'
timestr = time.strftime("%Y%m%d-%H%M%S")
foldername = OUTPUT_PATH+'TRICENTER_BIMODAL_'+timestr
os.makedirs(foldername) # one experiment = one folder

# --------------- Scientific details

# what dataset
experiment = 'bimodal'
# what accuracy ranges
acc_ranges = [[0.6,0.75],
          [0.75,0.83],
          [0.83,.95]]


# ====================================================================================
# ====================================================================================


# iterating over accuracy ranges
for i_range,acc_range in enumerate(acc_ranges):


    mat = sio.loadmat(DATA_PATH + experiment)
    s1_all = np.log(np.asarray((mat['s1']))).astype(float)
    s2_all = np.log(np.asarray((mat['s2']))).astype(float)
    resp_all = np.asarray(mat['resp']) + 0.
    acc_all = np.asarray(mat['acc']) + 0.


    nsubs,ntrials =acc_all.shape
    print 'nsub:', nsubs, '/ntrials:', ntrials, '/total trials:', nsubs*ntrials

    Iacc = np.where((acc_all.mean(1)>acc_range[0]) & (acc_all.mean(1)<acc_range[1]))[0]
    s1,s2 ,resp,acc= s1_all[Iacc,:], s2_all[Iacc,:],resp_all[Iacc,:], acc_all[Iacc,:]

    lag = 2
    sm = (s1 + s2)*.5
    diff = (s1 - s2)[:,lag:] # f1 - f2 at trial time
    prev1 = s1[:,lag:]-sm[:,1:-lag+1] # f1(t) - (f1+f2)(t-1)
    prev2 = s1[:,lag:]-sm[:,:-lag] # f1(t) - (f1+f2)(t-2)
    inf = s1[:,lag:] - s1.mean() # f1(t) - overallmean(f1)
    Y = resp[:,lag:].astype(int)

    a,b = Y.shape
    diff = np.reshape(diff,[a*b,1])
    prev1 = np.reshape(prev1,[a*b,1])
    prev2 = np.reshape(prev2,[a*b,1])
    inf = np.reshape(inf,[a*b,1])
    Y = np.reshape(Y,[a*b,1])

    X = np.hstack([diff,prev1,inf])
    cov_names = ['$df$', '$d_1$', '$d_{\infty}$']
    cov_to_index = {s:i for i,s in enumerate(cov_names)}

    # cropping of trials for large d1 values
    th = np.sort(np.abs(diff))[len(diff)/1.1]
    Itrial = np.where((np.abs(prev1)<.9))[0]# subselecting
    X,Y = X[Itrial,:],Y[Itrial,:]


    print X.shape,Y.shape


    #TODO : add more info about filtering
    exp_details = {
        'X':X,
        'Y':Y,
        'cov_names':cov_names,
        'subjects':Iacc,
        'accuracies':acc.mean(1)
    }
    pickle.dump(exp_details,open(foldername+'/details_'+str(i_range)+'.p','wb'))

# ====================================================================================
# ====================================================================================

    lengthscales = .833


    def model_add():
        ''' additive model for bias of d1,dinf '''
            # additive structure
        f_indices = [[0],[1],[2]]
        # Inducing point locations
        N,Nz =len(X),40

        Z = [np.array([[1]]) ,
             np.expand_dims( X[np.random.permutation(N)[:Nz],1],1),
             np.expand_dims( X[np.random.permutation(N)[:Nz],2],1)]
        # Setting kernels
        ks = [Linear(1),
              Antisymmetric_RBF(1,lengthscales=lengthscales,variance=1.),
              Antisymmetric_RBF(1,lengthscales=lengthscales,variance=1.)]
        # Declaring model
        m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                            f_indices=f_indices,name = 'bias f(d1)+f(dinf)')
        m.Z[0].fixed = True # no need to optimize location for linear parameter
        m.kerns.parameterized_list[0].variance.fixed = True
        m.kerns.parameterized_list[1].variance.fixed = True
        m.kerns.parameterized_list[2].variance.fixed = True
        m.kerns.parameterized_list[1].lengthscales.fixed = True
        #m.kerns.parameterized_list[2].lengthscales.fixed = True # no fix for dinf
        return m

    def model_joint():
        ''' joint model for bias of d1,dinf  '''
        # additive structure
        f_indices = [[0],[1,2]]
        # Inducing point locations
        N,Nz =len(X),40
        Nz =40
        Z = [np.array([[1]]),
             np.vstack( [ X[np.random.permutation(N)[:Nz],1] ,\
                       X[np.random.permutation(N)[:Nz],2]] ).T ]

        print 'Z:',Z[1].shape
        # Setting kernels
        ks = [Linear(1),
              RBF(2,lengthscales=1.,variance=1.,ARD=True)]
        # Declaring model
        m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                            f_indices=f_indices,name = 'bias f(d1,dinf)')
        m.Z[0].fixed = True # no need to optimize location for linear parameter
        m.kerns.parameterized_list[0].variance.fixed = True
        m.kerns.parameterized_list[1].variance.fixed = True
#       m.kerns.parameterized_list[1].lengthscales.fixed = True

        return m


    ms = [model_add(),model_joint()]
    #assert False
# ====================================================================================
# ====================================================================================


    # running optimization

    for m_ in ms:
        success = False
        rep = 0
        while (success == False)&(rep<10):
            try:
                res = m_.optimize()
                success = res['success']
            except Exception:
                pass
            rep+=1

    # extracting output of fit

    out = {}
    for m in ms:
        prms = m.extract_params()
        out[m.name]= prms
        out[m.name]['f_indices'] = m.f_indices
        try:
            out[m.name]['lb'] = m.lower_bound_likelihood()
        except Exception:
            pass

    pickle.dump(out,open(foldername+'/models_'+str(i_range)+'.p','wb'))
    # performing prediction
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
        'Y':Y,
        'f_indices':[m.f_indices for m in ms],
        'mean':M_fs,
        'variance':V_fs
    }
    pickle.dump(predictions,open(foldername+'/predictions_'+str(i_range)+'.p','wb'))

