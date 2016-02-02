'''
Functions to sample pairs of frequencies for frequency discrimination tasks
'''
import numpy as np
#import scipy.io as sio




def unimodal_grid(a=900.,b=1100,n_steps=5):
    """
    Creates a homogenous 2d grid of [a,b]*[a,b], with n_step steps on each dimension
    TODO: permutation and subselect f1,f2 as in experiment
    :param a: Lower Bound (Hz)
    :param b: Upper Bound (Hz)
    :param n_step: number of samples (log)
    :return: log frequencies arrays
    """
    f = np.linspace(np.log(a),np.log(b),n_steps)
    f1,f2 = np.meshgrid(f,f)
    return np.reshape(f1,n_steps*n_steps),\
           np.reshape(f2,n_steps*n_steps)
           
def sample_s1_s2_unimodal(samples,frange,dfrange):
    '''
    Sampling frequency pairs, first one according to uniform
    Second one conditional on first setting a difficulty sampled from  U[-df,df]
    with resampling if out of range bounds
    :param samples: number of trials
    :param frange: frequency range (list)
    :param dfrange: difficulty range, in Hz
    :return: s1, s2
    '''
    dfrange = np.log(dfrange)
    s1 = np.random.rand(samples)*(np.log(frange[1])-np.log(frange[0]))+np.log(frange[0])
    s2 = np.empty(samples)
    for ii in range(samples):
        s2[ii] = s1[ii] + (np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
        if np.random.rand()>0.5:
            s2[ii] = s1[ii] - (np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
        if np.random.rand()>0.5:
            s1[ii],s2[ii] = s2[ii],s1[ii]
    return s1,s2

def sample_s1_s2_bimodal(samples,franges,dfrange):
    '''
    Sampling from bimodal distribution
    :param samples: number of trials
    :param franges: frequency range (list of list)
    :param dfrange: difficulty range, in Hz
    :return:
    '''
    dfrange = np.log(dfrange)
    dice = np.random.rand(samples)
    s1 = (dice>0.5)*np.random.rand(samples)*(np.log(franges[1])-np.log(franges[0]))+np.log(franges[0]) +\
        (dice<0.5)*np.random.rand(samples)*(np.log(franges[3])-np.log(franges[2]))+np.log(franges[2])
    s2 = np.empty(samples)
    for ii in range(samples):
        s2[ii] = s1[ii] + (np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
        if np.random.rand()>0.5:
            s2[ii] = s1[ii] - (np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
        if np.random.rand()>0.5:
            s1[ii],s2[ii] = s2[ii],s1[ii]
    return s1,s2

def sample_s1_s2_threshold(samples,frange,dfrange,threshold,condition):
    '''
    Sampling frequency pairs, first one according to uniform
    Second one conditional on first setting a difficulty sampled from  U[-df,df]
    with resampling if out of range bounds
    :param samples: number of trials
    :param frange: frequency range (list)
    :param dfrange: difficulty range, in Hz
    :return: s1, s2
    '''
    dfrange = np.log(dfrange)
    s1 = np.random.rand(samples) * (np.log(frange[1]) - np.log(frange[0])) + np.log(frange[0])
    s2 = np.empty(samples)

    s2[0] = s1[0] + (np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
    if np.random.rand()>0.5:
        s2[0] = s1[0] - (np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
    if np.random.rand()>0.5:
        s1[0],s2[0] = s2[0],s1[0]

    for ii in range(1,samples):
        dist = s1[ii] - 0.5*(s1[ii-1]+s2[ii-1])
        direction = np.sign(dist)

        if condition == 'inner_minus':
            if (np.abs(dist) < threshold):
                s2[ii] = s1[ii] + direction*(np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
            else:
                s2[ii] = s1[ii] - direction*(np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
        if condition == 'inner_plus':
            if (np.abs(dist) < threshold):
                s2[ii] = s1[ii] - direction*(np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
            else:
                s2[ii] = s1[ii] + direction*(np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
    return s1,s2

def sample_s1_s2_markov(samples, frange, dfrange, mrange_close, mrange_far, condition):
    '''
    Sampling frequency pairs, first one according to uniform
    Second one conditional on first setting a difficulty sampled from  U[-df,df]
    with resampling if out of range bounds
    :param samples: number of trials
    :param frange: frequency range (list)
    :param dfrange: difficulty range, in Hz
    :return: s1, s2
    '''

    if condition == 'inner_plus':
        inner = -1
    elif condition == 'inner_minus':
        inner = 1

    dfrange = np.log(dfrange)
    s1 = np.empty(samples)
    s2 = np.empty(samples)

    s1[0] = np.random.rand() * (np.log(frange[1]) - np.log(frange[0])) + np.log(frange[0])
    s2[0] = s1[0] + (np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
    if np.random.rand()>0.5:
        s2[0] = s1[0] - (np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])
    if np.random.rand()>0.5:
        s1[0],s2[0] = s2[0],s1[0]

    for ii in range(1,samples):

        if np.random.rand()>0.5:
            # sample close
            bias = 1
            add_sub = np.random.randint(2)*2-1
            s1[ii] = 0.5*(s1[ii-1]+s2[ii-1]) + add_sub*(np.random.rand()*(mrange_close[1] - mrange_close[0])+ mrange_close[0])
            while (s1[ii] < np.log(frange[0])) or (s1[ii] > np.log(frange[1])):
                add_sub = np.random.randint(2)*2-1
                s1[ii] = 0.5*(s1[ii-1]+s2[ii-1]) + add_sub*(np.random.rand()*(mrange_close[1] - mrange_close[0])+mrange_close[0])


        else:
            # sample far
            bias = -1
            add_sub = np.random.randint(2)*2-1
            s1[ii] = 0.5*(s1[ii-1]+s2[ii-1]) + add_sub*(np.random.rand()*(mrange_far[1] - mrange_far[0]) + mrange_far[0])
            while (s1[ii] < np.log(frange[0])) or (s1[ii] > np.log(frange[1])):
                add_sub = np.random.randint(2)*2-1
                s1[ii] = 0.5*(s1[ii-1]+s2[ii-1]) + add_sub*(np.random.rand()*(mrange_far[1] - mrange_far[0]) + mrange_far[0])

        dist = s1[ii] - 0.5*(s1[ii-1]+s2[ii-1])
        s2[ii] = s1[ii] + inner*bias*np.sign(dist)*(np.random.rand()*(dfrange[1]-dfrange[0]) + dfrange[0])

    return s1,s2