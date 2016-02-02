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

def plot_distribution(s1,s2):
    f, ax = plt.subplots(2,2)
    ax[0,0].plot(s1,s2,'.')
    ax[0,1].plot(s1[1:]-s2[1:],s1[1:]-0.5*(s1[:-1]+s2[:-1]),'.')
    ax[1,0].hist(s1, bins=30)
    ax[1,1].hist(s2, bins=30)


dist = "threshold"
dist = "markov_bimodal"


p_a_to_w_d = lambda p,a : [a*np.exp(1)/p, 1./p]
peaks = [0.15]
amps = [0.4]
prm_add = [p_a_to_w_d(p,a) for p,a in zip(peaks,amps)]
prm_lin = 1/0.02 # linear parameter
prm_lik = .1 # parameter of the likelihood


peaks = [1.3]
amps = [1.3]
prm_add = [p_a_to_w_d(p,a) for p,a in zip(peaks,amps)]
prm_lin = 1/0.052 # linear parameter
prm_lik = .1 # parameter of the likelihood

# threshold1 =
# threshold2 =

samples = 10000
frange = (500,2000)

dfrange = (1.005,1.05)
threshold = 0.2
frange = (450,2300)
mrange_close = [np.log(1100) - np.log(950), np.log(1250) - np.log(800)]
mrange_far = [np.log(1250) - np.log(800), np.log(2300) - np.log(450)]




# # Create model instance
acc_vec = np.empty([2,2])
s1_pm,s2_pm,s1_mp,s2_mp = sample_s1_s2_markov(samples, frange, dfrange, mrange_close, mrange_far)
# s1_pm, s2_pm, s1_mp, s2_mp = sample_s1_s2_threshold(samples, frange, dfrange, threshold)

for jj,ii in enumerate(['mp','pm']):
    if ii == 'mp':
        s1 = s1_mp
        s2 = s2_mp
    elif ii == 'pm':
        s1 = s1_pm
        s2 = s2_pm

    plot_distribution(s1,s2)
    plt.savefig(OUTPUT_PATH  + "samples_" + ii + ".svg")

    model = LinExpAdditiveModel(prm_add,prm_lin,prm_lik)
    T = 1
    inf = 0
    x,y = get_trial_covariates_single(s1,s2,s1,T=T,inf=inf)
    b = model.bern(x.T)
    y = np.random.rand(samples-1)<b
    acc = y == (x[0,:]>0)

    # plotting functions
    plt.figure()
    xp = np.linspace(-1.5,1.5,100)
    for a in model.a:
        plt.plot(xp,a(xp))

    plt.savefig(OUTPUT_PATH  + "sim.svg")
    plt.close()

    acc_vec[jj,:] = [acc[np.abs(x[1,:])<threshold].mean(),acc[np.abs(x[1,:])>threshold].mean()]

acc_diff = acc_vec[0,:] - acc_vec[1,:]
print acc_diff


