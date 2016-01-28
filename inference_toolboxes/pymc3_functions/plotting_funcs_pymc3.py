# -*- coding: utf-8 -*-
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy.stats import norm
# import pickle
import numpy.matlib
# from analyze_data import *
from inference import *


def plot_mcmc_nd(trace, t, axarr, input_color, alpha_input,bias=None, samples=1000):

    w = trace['w']
    decay = trace['decay']
    xc = trace['xc']

    x = np.linspace(-1.5,1.5,1000)
    R = np.empty((samples,1000))
    for jj in range(t):
        ii = 0
        for w_,d,xc_ in zip(w[:,jj],decay[:,jj],xc[:,jj]):
            # R[ii,:] = (w_*x*np.exp(-np.abs(x)*d))
            # print w_,d
            # print x
            # print bias(w_,d)(x)
            R[ii,:] = bias(w_,d,xc_)(x)
            # R = (w*x*np.exp(-np.abs(x)**d))/s
            # axarr[jj].plot(x, R[ii,:], color=input_color, lw=1, alpha=1)
            ii = ii + 1
        R_err1 = R.mean(axis=0) - R.std(axis=0)
        R_err2 = R.mean(axis=0) + R.std(axis=0)
        axarr[jj].plot(x, R.mean(axis=0), color=input_color, lw=1, alpha=1)
        axarr[jj].fill_between(x,R_err1, R_err2, color=input_color, lw=1, alpha=alpha_input)
        axarr[jj].set_ylim([-0.8, 0.8])
        if jj==(t-1):
            axarr[jj].set_title(r' Time lag: inf ', fontsize=20)
            axarr[jj].set_xlabel(r'$\langle f \rangle _{(t)} - \langle f \rangle _{\infty}$', fontsize=20)

        elif jj==(t-2):
            axarr[jj].set_xlabel(r'$\langle f \rangle _{(t)} - \langle f \rangle _{(t-2)}$', fontsize=20)
            axarr[jj].set_title(r' Time lag: t-' + str(jj+1), fontsize=20)
        else:
            axarr[jj].set_xlabel(r'$\langle f \rangle _{(t)} - \langle f \rangle _{(t-1)}$', fontsize=20)
            axarr[jj].set_title(r' Time lag: t-' + str(jj+1), fontsize=20)

        axarr[jj].set_ylabel(r'$ \alpha $', fontsize=30)

def plot_mcmc_nd_poly(trace, t, axarr, input_color, alpha_input,bias=None, samples=1000):

    w1 = trace['w1']
    w3 = trace['w3']
    w5 = trace['w5']
    w7 = trace['w7']
    w9 = trace['w9']
    x = np.linspace(-1,1,1000)
    R = np.empty((samples,1000))
    for jj in range(t):
        ii = 0
        for w1_,w3_,w5_,w7_,w9_ in zip(w1[:,jj],w3[:,jj],w5[:,jj],w7[:,jj],w9[:,jj]):
            # R[ii,:] = (w_*x*np.exp(-np.abs(x)*d))
            R[ii,:] = bias(w1_,w3_,w5_,w7_,w9_)(x)
            # R = (w*x*np.exp(-np.abs(x)**d))/s
            axarr[jj].plot(x, R[ii,:], color=input_color, lw=1, alpha=1)
            ii = ii + 1

        # R_err1 = R.mean(axis=0) - R.std(axis=0)
        # R_err2 = R.mean(axis=0) + R.std(axis=0)
        # axarr[jj].plot(x, R.mean(axis=0), color=input_color, lw=1, alpha=1)
        # axarr[jj].fill_between(x,R_err1, R_err2, color=input_color, lw=1, alpha=alpha_input)
        # axarr[jj].set_ylim([-0.6, 0.6])



    # plt.ylabel(r'$ \alpha $ / $ \sigma $', fontsize=40)
    # plt.xlabel(r'$\langle f \rangle _{(t)} - \langle f \rangle _{(t-1)}$', fontsize=40)
    # label_size = 15
    # plt.rcParams['ytick.labelsize'] = label_size
    # plt.title('Bias as a function of distance', fontsize=30)

