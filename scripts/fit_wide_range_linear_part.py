'''
This script is very similar to fit_wide_range
The only difference is in the final plotting to superimpose curves across subject groups
'''

from model_simulations.descriptive_model import *
from utils import *
from matplotlib import pyplot as plt
from data_loader.load_data import *
import Globals
import os



n_rep = 10
n_restart = 5
T=1


for model in ['lin','explin']:

    for thresh in [0.1,0.2,1]:

        col = 'rgb'
        fig1, ax1 = plt.subplots()


        for i_typ,typ in enumerate(['good','poor']):

            save_prefix = 'linear_part/'
            if not os.path.exists('./'+save_prefix):
                os.makedirs('./'+save_prefix)

            save_name = model+'_ml_fit_'+typ+'_'+str(thresh)+'.svg'

            print save_name

            #=========================================
            # Loading data per accuracy
            #=========================================

            data_path = Globals.path_dic('data')
            loader = Dataloader(data_path+'continuous.mat')
            poor = (0.65,0.85)
            good = (0.85,1)

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
            I = np.where(np.abs(x[:,1])<thresh)[0]
            print x.shape, x[:,1].min(),x[:,1].max()
            x =x[I,:]
            y =y[I]
            print x.shape, x[:,1].min(),x[:,1].max()
            n_eff = x.shape[0]
            #=========================================
            # Declare model
            #=========================================

            if model == 'explin':
                mod_fit = LinExpAdditiveModel()
            elif model == 'lin':
                mod_fit = LinAdditiveModel()
            w0 = mod_fit.smart_init(T)

            #=========================================
            # Run ML fit
            #=========================================


            res = ml_fit(x,y,w0,mod_fit,prop_bstrap=0.9,n_rep=n_rep,n_restart=n_restart,tol=1.e-7)

            #========================================



            fig,axarr = plt.subplots(1,T+1,figsize=[10,5])
            for lag,ax in zip(range(T),axarr.flat):
                ax.scatter(res[:,2*lag],res[:,2*lag+1])
                ax.set_xlabel('w')
                ax.set_ylabel('decay')
                ax.set_title('lag 1, '+typ+'/ n='+str(n_eff) )
            axarr[-1].scatter(res[:,-2],res[:,-1])
            axarr[-1].set_xlabel('lin')
            axarr[-1].set_ylabel('lik')
            plt.tight_layout()
            plt.savefig(save_prefix+'scatter_'+save_name)
            plt.close()




            xmin, xmax = x[:,1].min(),x[:,1].max()
            xp = np.linspace(xmin,xmax,100)
            #plt.hist(x[:,1:].flatten(),bins=100,alpha=0.2,normed=True)


            for i_r,r in enumerate(res):
                p_add,p_lin,p_lik = mod_fit.unflat_param(r)
                for lag in range(T):
                    kwargs = {'label':'lag 1, '+typ+'/ n='+str(n_eff)} if i_r==0 else {}
                    mod_fit.set_params_uc(p_add,p_lin,p_lik)
                    ax1.plot(xp,mod_fit.a[lag](xp),col[i_typ],**kwargs)
                    #axarr1[1].plot(xp*p_lin,mod.bias(p_add[lag])(xp),col[i_typ],**kwargs)
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels)


        plt.savefig(save_prefix+'curves_'+save_name)
        plt.close()


    plt.close('all')