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

DATA_PATH = '../../data_files/'
OUTPUT_PATH = '../../output/'
timestr = time.strftime("%Y%m%d-%H%M%S")
foldername = OUTPUT_PATH+'TRICENTRE-DYS-D1Dinf-LS-'+timestr
os.makedirs(foldername) # one experiment = one folder


# ITERATING OVER LENGTHSCALES

LScales_d1 = np.linspace(.5, 2,5)
LScales_dinf = np.linspace(.5,2,5)
print LScales_d1
print LScales_dinf


LB_LS = []

experiment = 'neuro_sagi_FD'


acc_range = [0.6,0.9]

populations = ['control','dyslexics']

#========================================================================================


def get_covariates(filt,experiment,population):

    if population == 'control':
        mat = sio.loadmat(DATA_PATH + experiment)
        s1_all = np.asarray((mat['s1_con'])).astype(float)
        s2_all = np.asarray((mat['s2_con'])).astype(float)
        resp_all = np.asarray(mat['resp_con']) + 0.
        acc_all = np.asarray(mat['acc_con']) + 0.
    elif population == 'dyslexics':
        mat = sio.loadmat(DATA_PATH + experiment)
        s1_all = np.asarray((mat['s1_dys'])).astype(float)
        s2_all = np.asarray((mat['s2_dys'])).astype(float)
        resp_all = np.asarray(mat['resp_dys']) + 0.
        acc_all = np.asarray(mat['acc_dys']) + 0.


    nsubs,ntrials =acc_all.shape
    print 'nsub:', nsubs, '/ntrials:', ntrials, '/total trials:', nsubs*ntrials

    Iacc =  np.where((acc_all.mean(1)>filt[0]) & (acc_all.mean(1)<filt[1]))[0]
    s1,s2 = s1_all[Iacc,:], s2_all[Iacc,:]
    resp,acc = resp_all[Iacc,:], acc_all[Iacc,:]

    lag = 2
    sm = (s1 + s2)*.5
    diff = (s1 - s2)[:,lag:] # f1 - f2 at trial time
    prev1 = s1[:,lag:]-sm[:,1:-lag+1] # f1(t) - (f1+f2)(t-1)
    inf = s1[:,lag:] - s1.mean() # f1(t) - overallmean(f1)
    Y = resp[:,lag:].astype(int)

    a,b = Y.shape
    diff = np.reshape(diff,[a*b,1])
    prev1 = np.reshape(prev1,[a*b,1])
    inf = np.reshape(inf,[a*b,1])
    Y = np.reshape(Y,[a*b,1])

    X = np.hstack([diff,prev1,inf])
    cov_names = ['$df$', '$d_1$', '$d_{\infty}$']
    cov_to_index = {s:i for i,s in enumerate(cov_names)}

    # further filtering
    Itrial = np.where((np.abs(prev1)<.9))[0]# subselecting
    X,Y = X[Itrial,:],Y[Itrial,:]

    return X,Y, len(Iacc)


def model_add(X,Y,l_d1,l_dinf):
    '''
    :param l_d1: lengthscale for d1
    :param l_dinf: lengthscale for dinf
    :return:
    '''
    # additive structure
    f_indices = [[0],[1],[2]]
    # Inducing point locations
    Nz =40
    N = len(Y)
    Z = [np.array([[1]]) ,
         np.expand_dims( X[np.random.permutation(N)[:Nz],1],1),
         np.expand_dims( X[np.random.permutation(N)[:Nz],2],1)]
    # Setting kernels
    ks = [Linear(1),
          Antisymmetric_RBF(1,lengthscales=l_d1,variance=1.),
          Antisymmetric_RBF(1,lengthscales=l_dinf,variance=1.)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1)+f(dinf)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.kerns.parameterized_list[0].variance.fixed = True
    m.kerns.parameterized_list[1].variance.fixed = True
    m.kerns.parameterized_list[2].variance.fixed = True
    m.kerns.parameterized_list[1].lengthscales.fixed = True
    m.kerns.parameterized_list[2].lengthscales.fixed = True
    return m


#========================================================================================




#TODO : add more info about filtering
exp_details = {
    'experiment':experiment,
    'acc_range':acc_range,
    'LS_d1':LScales_d1,
    'LS_dinf':LScales_dinf
}

pickle.dump(exp_details,open(foldername+'/details.p','wb'))


# -------------------------

# iterate over accuracy range
for i_pop,population in enumerate(populations):

    X,Y,nsub = get_covariates(acc_range,experiment,population)

    # iterate over lengthscale
    for i_lsd1, l_d1 in enumerate(LScales_d1):

        # iterate over lengthscale
        for i_lsdinf,l_dinf in enumerate(LScales_dinf):

            postfix = '_'+str(i_pop)+'_'+str(i_lsd1)+'_'+str(i_lsdinf)+'_'


            m = model_add(X,Y,l_d1,l_dinf)
            model_name = m.name
            # optimize until convergence
            success = False
            rep = 0
            while (success == False)&(rep<5):
                res = m.optimize()
                success = res['success']
                rep+=1
            # saving output of model fit:
            prms = m.extract_params()
            prms['f_indices'] = m.f_indices
            b= m.lower_bound_likelihood()
            prms['lb'] = b
            prms['population']=population
            prms['nsubjects']=nsub
            LB_LS.append(b)

            # saving fitted models
            pickle.dump(prms,open(foldername+'/models'+postfix+'.p','wb'))

            # computing the individual function values
            try:
                mu_fs,v_fs = m.predict_fs(X)
            except Exception:
                mu_fs,v_fs=np.zeros(Y.shape),np.zeros(Y.shape)
            print mu_fs.shape

            predictions = {
                'X':X,
                'Y':Y,
                'f_indices':m.f_indices,
                'mean':mu_fs,
                'nsubject':nsub,
                'variance':v_fs,
                'population':population
            }
            pickle.dump(predictions,open(foldername+'/predictions+'+postfix+'.p','wb'))




pickle.dump({'LS_1':LScales_d1,
             'LS_inf':LScales_dinf,
             'LB':LB_LS},open(foldername+'/evidence_LS.p','wb'))