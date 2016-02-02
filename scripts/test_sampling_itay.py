import sys
import os
import Globals
import numpy as np
from matplotlib import pyplot as plt

OUTPUT_PATH = os.path.expanduser('~')+'/'+'Python_outputs'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_PATH = "/home/ubuntu-lieder-pc/Python_outputs/"
OUTPUT_PATH = "/home/dell/Python_outputs/"

CODE_PATH = Globals.path_dic('code')
DATA_PATH = Globals.path_dic('data')

# Make sure the data files are named: 'continuous.mat'
# and 'bimodal.mat' for each respective experiment
sys.path.append(CODE_PATH)

from data_loader.load_data import *
from inference_toolboxes.pymc3_functions.inference_pymc3 import *
from stimulus_generation.generate_stimuli import *
from model_simulations.descriptive_model import *

samples = 1000
frange = (500,2000)
dfrange = (1.005,1.1)
threshold = 0.4
condition = 'inner_minus'

s1,s2 = sample_s1_s2_threshold(samples, frange, dfrange, threshold, condition)

f, ax = plt.subplots(2,2)
ax[0,0].plot(s1,s2,'.')
ax[0,1].plot(s1[1:]-s2[1:],s1[1:]-0.5*(s1[:-1]+s2[:-1]),'.')
ax[1,0].hist(s1)
ax[1,1].hist(s2)

plt.savefig(OUTPUT_PATH  + "samples.svg")


plt.figure()
samples = 1000
frange = (500,2000)
dfrange = (1.005,1.07)
condition = 'inner_minus'

mrange_close = [0, np.log(800) - np.log(450)]
mrange_far = [np.log(1250) - np.log(800), np.log(2300) - np.log(450)]


mrange_close = [0, np.log(1100) - np.log(950)]
mrange_far = [np.log(1250) - np.log(800), np.log(2300) - np.log(450)]




s1,s2 = sample_s1_s2_markov(samples, frange, dfrange, mrange_close, mrange_far, condition)

f, ax = plt.subplots(2,2)
ax[0,0].plot(s1,s2,'.')
ax[0,1].plot(s1[1:]-s2[1:],s1[1:]-0.5*(s1[:-1]+s2[:-1]),'.')
ax[1,0].hist(s1, bins=30)
ax[1,1].hist(s2, bins=30)

plt.savefig(OUTPUT_PATH  + "samples_markov.svg")




p_a_to_w_d = lambda p,a : [a*np.exp(1)/p, 1./p]
peaks = [0.1]
amps = [1]
prm_add = [p_a_to_w_d(p,a) for p,a in zip(peaks,amps)]
# 0.014
# Declare model parameters GOOD
prm_add = [[-25.64,0.05]] # parameters of additive functions
prm_lin = .02 # linear parameter
prm_lik = .1 # parameter of the likelihood

# # Declare model parameters POOR
# prm_add = [[-1.92,0.68]] # parameters of additive functions
# prm_lin = .052 # linear parameter
# prm_lik = .1 # parameter of the likelihood

# Create model instance
model = LinExpAdditiveModel(prm_add,prm_lin,prm_lik)


samples = 100000
frange = (500,2000)
dfrange = (1.005,1.05)
threshold = 0.35




plt.figure()

for ii in range(2):
    condition = ['inner_plus','inner_minus']

    s1,s2 = sample_s1_s2_threshold(samples, frange, dfrange, threshold, condition[ii])
    # s1,s2 = sample_s1_s2_markov(samples, frange, dfrange, mrange_close, mrange_far, condition[ii])

    # plotting functions
    xp = np.linspace(-1.5,1.5,100)
    for a in model.a:
        plt.plot(xp,a(xp))

    plt.savefig(OUTPUT_PATH  + "sim.svg")
    plt.close()



    T = 1
    inf = 0

    x,y = get_trial_covariates_single(s1,s2,s1,T=T,inf=inf)
    b = model.bern(x.T)
    y = np.random.rand(samples-1)<b
    acc = y == (x[0,:]>0)

    # compute likelihood
    # print model.llh(x.T,y)
    # print acc.mean()

    plt.figure()
    f, ax = plt.subplots(2)
    ax[0].hist(b)


    # ax[1].bar([acc[x[1,:]<threshold].mean(),acc[x[1,:]>threshold].mean()])

    nbins = 10
    acc = acc+np.random.normal(0,0.001,size=samples-1)
    n, _ = np.histogram(x[1,:], bins=nbins)
    sy, _ = np.histogram(x[1,:], bins=nbins, weights=acc)
    sy2, _ = np.histogram(x[1,:], bins=nbins, weights=acc*acc)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    ax[1].errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='r-')
    # ax[1,1].plot(x[1,:],acc+np.random.normal(0,0.1,size=samples-1),'.',alpha=0.5)

    plt.savefig(OUTPUT_PATH  + "acc.svg")

    print acc[np.abs(x[1,:])<threshold].mean()
    print acc[np.abs(x[1,:])>threshold].mean()

