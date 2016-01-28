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


#================================================================
# OLD BELOW
#================================================================

'''

def real_participant(p=11,width=1):
    if width=='narrow':
        mat= sio.loadmat('hillaTrainNarrow_1block.mat')
        a,b = 940, 1060
    elif width=='mid':
        mat= sio.loadmat('hillaTrainMedium_1block.mat')
        a,b = 800,1250
    elif width=='broad':
        mat= sio.loadmat('hillaTrainWide_1block.mat')
        a,b = 670, 1500
        
    s1=mat['s1']
    s2=mat['s2']
    acc=mat['acc']
    resp=mat['resp']
    
    return np.concatenate(s1[0][p]), \
           np.concatenate(s2[0][p]), \
           np.concatenate(resp[0][p]), \
           a ,\
           b, \
#    acc=np.concatenate(acc[0][ii])
#    resp=np.concatenate(resp[0][ii])
          
def load_bimodal_single_subject(i_s):
    """
    Loading (tone1,tone2, response) of Mturk bimodal experiment
    :param i_s: index of subject
    :return:
    """
    assert i_s in range(70), 'subject index must lie between 0 and 69'
    data_path = './data/'
    fname = 'BiModalTurk2.mat'
    m = sio.loadmat(data_path+fname)
    return m['s1'][:,i_s],m['s2'][:,i_s],m['acc'][:,i_s]


'''
