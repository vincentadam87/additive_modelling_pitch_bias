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

def test_distribution(model,samples=10000):
    threshold = 0.5
    dfrange = (1.005,1.04)
    frange = (450,2300)
    thresh1 = np.log(1070) - np.log(970)
    thresh2 = np.log(1180) - np.log(970)
    thresh3 = np.log(1250) - np.log(800)

    mrange_close = [thresh1, thresh2]
    mrange_far = [thresh3, np.log(2300) - np.log(450)]
    proportions = 0.35


    # # Create model instance
    acc_vec = np.empty([3,2])
    acc_all = np.empty([3])

    s1_pm,s2_pm,s1_mp,s2_mp = sample_s1_s2_markov(samples, frange, dfrange, mrange_close, mrange_far,proportions)
    # s1_pm, s2_pm, s1_mp, s2_mp = sample_s1_s2_threshold(samples, frange, dfrange, threshold)
    s1_reg, s2_reg, s1_reg, s2_reg = sample_s1_s2_unimodal(samples,frange,dfrange)

    for jj,ii in enumerate(['reg','mp','pm']):
        if ii == 'reg':
            s1 = s1_reg
            s2 = s2_reg
        elif ii == 'mp':
            s1 = s1_mp
            s2 = s2_mp
        elif ii == 'pm':
            s1 = s1_pm
            s2 = s2_pm

        plot_distribution(s1,s2)
        plt.savefig(OUTPUT_PATH  + "samples_" + ii + ".svg")

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

        # plt.savefig(OUTPUT_PATH  + "sim.svg")
        # plt.close()

        acc_vec[jj,:] = [acc[np.abs(x[1,:])<=thresh2].mean(),acc[np.abs(x[1,:])>thresh2].mean()]
        acc_all[jj] = acc.mean()

    return acc_vec, acc_all


def get_means_errors(acc_vec, acc_all):
    # calculate means
    acc_vec_out = 0
    for ii in range(nSub):
        acc_vec_out = acc_vec_out + acc_vec[ii]
    acc_vec_out = acc_vec_out/nSub
    # calculate error bars
    acc_vec_out_err = 0
    for ii in range(nSub):
        acc_vec_out_err = acc_vec_out_err + (acc_vec[ii]-acc_vec_out)**2
    acc_vec_err_out = np.sqrt(acc_vec_out_err/nSub)/np.sqrt(nSub)

    acc_all_out = acc_all.mean(axis=0)
    acc_all_err_out = (acc_all_out.std(axis=0))/np.sqrt(nSub)

    return acc_vec_out, acc_all_out, acc_vec_err_out, acc_all_err_out


def autolabel(rects,ii):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax[ii].text(rect.get_x() + rect.get_width()/2., 0.01+1.05*height,
                '%d' % int(height*100) + '%',
                ha='center', va='bottom',fontsize=8,rotation=90)

def acc_bars(ii,acc_vec, acc_all, acc_vec_err, acc_all_err,title):
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4
    rects1 = ax[ii].bar(ind, acc_all, width, color='b',yerr=acc_all_err)
    rects2 = ax[ii].bar(ind + width , acc_vec[:,0], width/2.5, color='g',yerr=acc_vec_err[:,0])
    rects3 = ax[ii].bar(ind + width + width/2.5 , acc_vec[:,1], width/2.5, color='r',yerr=acc_vec_err[:,1])

    ax[ii].set_ylim([0.2,1])
    ax[ii].set_xlim([0,2.7])
    ax[ii].set_ylabel('Accuracy')

    ax[1].legend((rects1[0], rects2[0], rects3[0]), ('Overall (100%)', 'Close (65%)', 'Far (35%)'),loc=0, bbox_to_anchor=(0, 0, 1, 1),prop={'size':9})
    ax[ii].set_title(title)

    x = [0.25,1.25,2.25]
    ax[ii].set_xticks(x)
    ax[ii].set_xticklabels(['Control', 'close(+)/far(-)', 'close(-)/far(+)'],rotation=45)

    autolabel(rects1,ii)
    autolabel(rects2,ii)
    autolabel(rects3,ii)





if __name__ == '__main__':
    p_a_to_w_d = lambda p,a : [a*np.exp(1)/p, 1./p]
    peaks = [0.15, 1.3]
    amps = [0.4, 1.3]

    peaks = [1.3, 1.3]
    amps = [1.3, 1.3]
    prm_add = [p_a_to_w_d(p, a) for p,a in zip(peaks, amps)]
    prm_lin = [1/0.02, 1/0.052] # linear parameter
    prm_lik = [.1, .1] # parameter of the likelihood

    nSub = 15
    samples = 150

    acc_all_good = np.empty([nSub,3])
    acc_all_poor = np.empty([nSub,3])
    acc_vec_good = []
    acc_vec_poor = []

    model_good = LinExpAdditiveModel([prm_add[0]],prm_lin[0],prm_lik[0])
    model_poor = LinExpAdditiveModel([prm_add[1]],prm_lin[1],prm_lik[1])

    for ii in range(nSub):
        print ii
        a, acc_all_good[ii,:] = test_distribution(model_good,samples)
        acc_vec_good.append(a)
        a, acc_all_poor[ii,:] = test_distribution(model_poor,samples)
        acc_vec_poor.append(a)

    plt.figure()
    f, ax = plt.subplots(1,2)
    acc_vec, acc_all, acc_vec_err, acc_all_err = get_means_errors(acc_vec_good, acc_all_good)
    acc_bars(0,acc_vec, acc_all, acc_vec_err, acc_all_err,'Good (n=15)')

    acc_vec, acc_all, acc_vec_err, acc_all_err = get_means_errors(acc_vec_poor, acc_all_poor)
    acc_bars(1,acc_vec, acc_all, acc_vec_err, acc_all_err,'Poor (n=15)')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + "bar.png")


