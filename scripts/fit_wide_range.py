'''
This script performs maximum likelihood learning of an additive model for classification
The data used is that of the wide range sampling experiment (2 tones discrimination)
'''

from model_simulations.descriptive_model import *
from matplotlib import pyplot as plt
from data_loader.load_data import *
import Globals
import os


# Data processing options (accuracy ranges)
poor = (0.65,0.85)
good = (0.85,1)

# Optimization options
n_rep = 10
n_restart = 10
thresh  = .1 # threshold on distance to t-1
mod_typ ='lin' #

# saving options
save_prefix = 'lags_non_lin_ml/'
if not os.path.exists('./'+save_prefix):
    os.makedirs('./'+save_prefix)


# iterating over lags
for T in [1,2]:

    # iterating over groups of subjects
    for i_typ,typ in enumerate(['good','poor']):


        save_name = mod_typ+'_'+typ+str(T)+'_tresh_'+str(thresh)+'.svg'
        print save_name

        #=========================================
        # Loading data per accuracy
        #=========================================

        data_path = Globals.path_dic('data')
        loader = Dataloader(data_path+'continuous.mat')

        r = poor if typ == 'poor' else good
        I = loader.subject_indices_for_acc_range(r)
        F1,F2,Y = loader.subject_data_from_indices(list(I))

        #=========================================
        # Preprocess data for regression
        #=========================================

        x,y = get_trial_covariates(F1,F2,Y,T=T,inf=0)
        x = x.T

        #=========================================
        # Subselect close trials
        #=========================================
        if T>0:
            I = np.where(np.abs(x[:,1])<thresh)[0]
            print x.shape, x[:,1].min(),x[:,1].max()
            x =x[I,:]
            y =y[I]
            print x.shape, x[:,1].min(),x[:,1].max()
            xmin, xmax = x[:,1].min(),x[:,1].max()

        n_eff = x.shape[0]
        #=========================================
        # Declare model
        #=========================================

        if mod_typ == 'explin':
            mod_fit = LinExpAdditiveModel()

        elif mod_typ == 'lin':
            mod_fit = LinAdditiveModel()

        w0 = mod_fit.smart_init(T)
        print 'w0:', w0
        #=========================================
        # Run ML fit
        #=========================================

        res = ml_fit(x,y,w0,mod_fit,prop_bstrap=0.90,n_rep=n_rep,n_restart=n_restart, s_noise=0.1,tol=1e-6)

        #========================================


        '''
        fig,axarr = plt.subplots(1,T+1,figsize=[10,5])
        for lag,ax in zip(range(T),axarr.flat):
            ax.scatter(res[:,2*lag],res[:,2*lag+1])
            ax.set_xlabel('w')
            ax.set_ylabel('decay')
            ax.set_title('lag '+str(lag+1)+'/ n='+str(n_eff) )
        axarr[-1].scatter(res[:,-2],res[:,-1])
        axarr[-1].set_xlabel('lin')
        axarr[-1].set_ylabel('lik')
        plt.tight_layout()
        plt.savefig(save_prefix+'scatter_'+save_name)
        plt.close()
        '''



        xp = np.linspace(-1,1,500)
        col = 'rgbkcy'
        fig,ax = plt.subplots(2,1,sharex=True)
        for i_r,r in enumerate(res):
            p_add,p_lin,p_lik = mod_fit.unflat_param(r)
            for lag in range(T):
                kwargs = {'label':'lag'+str(lag+1)+'/ n='+str(n_eff)} if i_r==0 else {}
                mod_fit.set_params_uc(p_add,p_lin,p_lik)
                ax[0].plot(xp,mod_fit.a[lag](xp),col[lag],**kwargs)

        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(handles, labels)
        ax[1].hist(x[:,1:].flatten(),bins=100,alpha=0.2,normed=True)

        plt.savefig(save_prefix+'curves_'+save_name)
        plt.close()


plt.close('all')