import numpy as np
import scipy.io as sio



#TODO
def get_data(path,experiment,flatten,acc_limits,subgroup,subtrials,width=1):
    '''
    Loads psychophysical data, frequency discrimination task
    :param path: Location of files
    :param experiment: name of experiment
    :param flatten: put all subjects together if 1
    :param acc_limits: bounds of accuracy to subselect subjects
    :param subgroup: subselect subjects by index
    :param subtrials: subselect by trials
    :param width: relevant only for "three_widths" experiment: 1 for narrow, 2 for medium, 3 for broad
    :return:
    '''
               
    mat = sio.loadmat(path + experiment)
    s1 = np.log(np.asarray((mat['s1'])))
    s2 = np.log(np.asarray((mat['s2'])))
    acc = np.asarray(mat['acc'])
    resp = np.asarray(mat['resp']) + 0.         
    
    ' %%%%%%%%%%% special case of 3-widths %%%%%%%%%%%%%'
    if experiment == 'three_widths':
        
        ' ========== Choose the width ==========='          
        s1 = s1[width-1,:]
        s2 = s2[width-1,:]
        acc = acc[width-1,:]
        resp = resp[width-1,:]
                
        ' ========== Choose a subset of participants ==========='          
        if subgroup == []:
            subgroup = range(len(s1))
        f1 = s1; f2 = s2; accr = acc; res = resp 
        s1 = []; s2 = []; acc = []; resp = []
    
        for ii in subgroup:
            if flatten == 1:
                s1.extend(f1[ii])
                s2.extend(f2[ii])
                acc.extend(accr[ii])
                resp.extend(res[ii])
            else:
                s1.append(np.asarray(f1[ii]))
                s2.append(np.asarray(f2[ii]))
                acc.append(np.asarray(accr[ii]))
                resp.append(np.asarray(res[ii]))
       
        if flatten == 1:
            s1 = np.asarray(s1)
            s2 = np.asarray(s2)
            acc = np.asarray(acc)
            resp = np.asarray(resp)
           
    else:
        ' %%%%%%%%%%% Rest of the experiments %%%%%%%%%%%%%'
    
        ' ========== Filter according to accuracy ==========='          
        lower,upper = acc_limits   
        filt = np.where((acc.mean(1) > lower) & (acc.mean(1) <= upper) )[0]
        s1 = s1[filt,:]
        s2 = s2[filt,:]
        acc = acc[filt,:]
        resp = resp[filt,:]  
            
            
        ' ========== Choose a subset of participants ==========='          
        if subgroup == []:
            pass
        else:        
            
            s1 = s1[subgroup,:]
            s2 = s2[subgroup,:]
            acc = acc[subgroup,:]
            resp = resp[subgroup,:] 
            
        ' ========== Choose a subset of trials ==========='             
        if subtrials == []:
            pass
        else:
            s1 = s1[:,subtrials]
            s2 = s2[:,subtrials]
            acc = acc[:,subtrials]
            resp = resp[:,subtrials] 
            
        ' ========== Flatten ==========='          
        if flatten == 0:
            pass
        else:
            s1 = s1.reshape(1,np.size(s1))
            s2 = s2.reshape(1,np.size(s2))
            acc = acc.reshape(1,np.size(acc))
            resp = resp.reshape(1,np.size(resp))
        data = {}
        data['s1'] = s1; data['s2'] = s2; data['acc'] = acc; data['resp'] = resp; data['filt'] = filt;

        return data


#=====================================================
# Load data Vincent
#=====================================================

class Dataloader(object):
    """
    Class to load subject data, assuming it contains s1,s2,acc,resp
    Loader is tied to a mat file
    """
    def __init__(self, fname):
        self.fname = fname

    def subject_data(self,i_sub,acc_filter = (0,1) , flat=False):
        """
        Loading subject data
        :param i_sub:
        :return: stimuli and response
        """
        data = sio.loadmat(self.fname)
        accr = data['acc'].mean(axis=1)
        filter = np.where((accr > acc_filter[0]) & (accr <= acc_filter[1]) )
        i_sub = filter[0]
        F1 = np.log(np.array(data['s1'][i_sub,:]))
        F2 = np.log(np.array(data['s2'][i_sub,:]))
        Y = data['resp'][i_sub,:]
        if flat:
            return F1.flatten(),F2.flatten(),Y.flatten()
        else:
            return F1,F2,Y

    def group_by_acc(self,th):
        """
        Group subjects by accuracy
        :param th:
        :return: indices of subjects whose accuracy is below (resp above) the threshold
        """
        data = loadmat(self.fname)
        acc = np.mean(data['acc'],axis=1)
        order = np.argsort(acc)
        acc= acc[order]
        I_good = order[acc>th]
        I_poor = order[acc<=th]
        return I_poor, I_good

    def group_by_acc_interval(self, th_down=0.,th_up=1.):
        """
        Select subjects by accuracy range
        :param th_down: accuracy threshold below
        :param th_up: accuracy threshold above
        :return: indices of subjects
        """
        data = sio.loadmat(self.fname)
        acc = np.mean(data['acc'],axis=1)
        order = np.argsort(acc)
        acc= acc[order]
        return order[(th_down<acc)&(acc<th_up)]



def get_trial_covariates_single(f1,f2,y,T=1):
    '''
    Build column matrix: [df. d1, d2, ... dT]
    where di is the distance from f1 to the mean of the trial T back
    :param F1: 1d Array of log frequencies
    :param F2: 1d Array of log frequencies
    :param T: lag, integer
    '''
    assert f1.ndim == 1
    assert f1.shape == f2.shape
    n_trials= len(f1)
    x_ = np.zeros((T+1,n_trials-T))
    y_= y[T:]
    # Trial difficulty
    x_[0,:]=(f1-f2)[T:]
    # Distance F1(t) - 1/2(F1+F2)(t-T)
    for t in range(1,T+1):
        x_[t,:]= f1[T:]-0.5*(f1+f2)[T-t:-t]
    return x_,y_

def get_trial_covariates(F1,F2,Y,T=1):
    '''
    Build column matrix: [df. d1, d2, ... dT], for multiple subjects.
    where di is the distance from f1 to the mean of the trial T back
    :param F1: 2d Array of log frequencies
    :param F2: 2d Array of log frequencies
    :param T: lag, integer
    '''
    assert F1.ndim==2
    assert F2.shape == F1.shape
    n_subs, n_trials = F1.shape
    X_ = []
    Y_ = []
    for f1,f2,y in zip(F1,F2,Y):
        x, y = get_trial_covariates_single(f1,f2,y,T)
        X_.append(x)
        Y_.append(y)
    return np.hstack(X_),np.hstack(Y_)






if __name__ == '__main__':
    fname = '/home/vincent/data/Itay_data/wideRange.mat'
    loader = Dataloader(fname)
    F1,F2,Y = loader.subject_data([0,1])
    print F1.shape
    F1 = F1[:,:10]
    F2 = F2[:,:10]
    print F1.shape
    f1 = F1.flatten()
    f2 = F2.flatten()

    T = 0
    M = get_trial_covariates_single(f1,f2,T)

    M = get_trial_covariates(F1,F2,T)

    print 'M:',M.shape
    print M