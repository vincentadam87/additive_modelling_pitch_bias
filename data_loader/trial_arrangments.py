
import numpy as np
import scipy as sc

def arrange_trials_t_back(data,t,inf=0):
    '''
    This functions calculates the distance from previous trials, going back according to the argument 't'
    :param data: contains the data loaded so far (s1,s2,resp etc...)
    :param t: how far back to go. 1: t-1, 2: t-2 etc...
    :return: returns a list of list. The outer list corresponds to trials-back, the inner lists to participants.
    for example:
    list[0][4] will correspond to list(t-1)(subject 5)
    list[1][2] will correspond to list(t-2)(subject 3)
    When pooled, always use '0' for the second argument
    '''
    s1 = data['s1']; s2 = data['s2']; acc = data['acc']; resp = data['resp']
    nSub, nTrials = np.shape(s1)
    s1l = []; s2l = []
    t_0 = t
    # t = t + inf
    for jj in range(t+inf): # +1 is for computing global
        s1ltemp = []; s2ltemp = []
        respl = []; accl = []
        for ii in range(nSub):
            f1 = s1[ii,t:]
            f2 = s2[ii,t:]
            accr = acc[ii,t:]
            res = resp[ii,t:]
            if jj == t_0:
                prev = np.mean(0.5*(s1[ii,:] + s2[ii,:]))
            else:
                prev = 0.5*(s1[ii,(t-jj-1):(-jj-1)] + s2[ii,(t-jj-1):(-jj-1)])

            s1ltemp.append(f1 - prev)
            s2ltemp.append(f2 - prev)
            respl.append(res)
            accl.append(accr)
        s1l.append(s1ltemp)
        s2l.append(s2ltemp)

    data['s1l'] = s1l; data['s2l'] = s2l; data['accl'] = accl; data['respl'] = respl
    return data


def arrange_trials_encoding_s1_s2(data):
      s1 = data['s1']; s2 = data['s2']; acc = data['acc']; resp = data['resp']

      nSub, nTrials = np.shape(s1)

      ps = np.abs(s1[:,0:-1] - s2[:,0:-1]) > 0.2
      s11l = s1[:,1:] - s1[:,0:-1]
      s21l = s2[:,1:] - s1[:,0:-1]
      s12l = s1[:,1:] - s2[:,0:-1]
      s22l = s2[:,1:] - s2[:,0:-1]
      accl = acc[:,1:]
      respl = resp[:,1:]
      ps = ps + 0.


      s11l = s11l[np.where(ps == 1)]
      s21l = s21l[np.where(ps == 1)]
      s12l = s12l[np.where(ps == 1)]
      s22l = s22l[np.where(ps == 1)]
      respl = respl[np.where(ps == 1)]
      accl = accl[np.where(ps == 1)]

      data['s11l'] = s11l; data['s21l'] = s21l; data['s12l'] = s12l; data['s22l'] = s22l
      data['accl'] = accl; data['respl'] = respl

      return data


def get_bias(data):
    '''

    :param s1:
    :param s2:
    :param resp:
    :param acc:
    :param nSub:
    :param nTrials:
    :return:
    '''

    s1 = data['s1']; s2 = data['s2']; acc = data['acc']; resp = data['resp']

    nSub, nTrials = np.shape(s1)

    acc_bias_g = np.empty([nSub,2])
    acc_bias_l = np.empty([nSub,2])
    resp_bias_g = np.empty([nSub,2])
    resp_bias_l = np.empty([nSub,2])

    # ============= Global bias ====================
    M = ((s1 + s2)/2).mean(1)
    M = np.matlib.repmat(M,nTrials,1).transpose()
    # ============= Accuracy ====================
    bias_p = np.sign(s2-s1) == np.sign(s1-M)
    bias_m = (np.sign(s2-s1) != np.sign(s1-M)) * (np.sign(s1-M) == np.sign(s2-M))

    acc_bias_g[:,0] = (acc*bias_p).mean(1)/(bias_p.mean(1))
    acc_bias_g[:,1] = (acc*bias_m).mean(1)/(bias_m.mean(1))
    # ============= Response ====================
    bias_high = (s2 > M) * (s1 > M)
    bias_low = (s2 < M) * (s1 < M)

    resp_bias_g[:,0] = (resp*bias_high).mean(1)/(bias_high.mean(1))
    resp_bias_g[:,1] = (resp*bias_low).mean(1)/(bias_low.mean(1))


    # ============= Local bias ====================
    M = ((s1[:,:-1] + s2[:,:-1])/2)
    # ============= Accuracy ====================
    bias_p = np.sign(s2[:,1:]-s1[:,1:]) == np.sign(s1[:,1:]-M)
    bias_m = (np.sign(s2[:,1:]-s1[:,1:]) != np.sign(s1[:,1:]-M)) * (np.sign(s1[:,1:]-M) == np.sign(s2[:,1:]-M))

    acc_bias_l[:,0] = (acc[:,1:]*bias_p).mean(1)/(bias_p.mean(1))
    acc_bias_l[:,1] = (acc[:,1:]*bias_m).mean(1)/(bias_m.mean(1))

    # ============= Response ====================
    bias_high = (s2[:,1:] > M) * (s1[:,1:] > M)
    bias_low = (s2[:,1:] < M) * (s1[:,1:] < M)

    resp_bias_l[:,0] = (resp[:,1:]*bias_high).mean(1)/(bias_high.mean(1))
    resp_bias_l[:,1] = (resp[:,1:]*bias_low).mean(1)/(bias_low.mean(1))


    biases = {'acc_global':acc_bias_g, 'acc_local':acc_bias_l, 'resp_global':resp_bias_g, 'resp_local':resp_bias_l}

    return biases