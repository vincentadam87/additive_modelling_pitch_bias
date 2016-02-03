from model_simulations.descriptive_model import *
from utils import *
from matplotlib import pyplot as plt
from data_loader.load_data import *
import Globals
import os


n_rep = 30

for model in ['lin','explin']:
    for typ in ['good','poor']:
        for thresh in [0.1, 0.5 , 1, 2]:

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

            T=1
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

            #=========================================
            # Declare model
            #=========================================

            if model == 'explin':
                mod = LinExpAdditiveModel()
                w0 = np.ones((2*T+2,))
            elif model == 'lin':
                mod = LinAdditiveModel()
                w0 = np.ones((T+2,))

            #=========================================
            # Run ML fit
            #=========================================


            res = ml_fit(x,y,w0,mod,prop_bstrap=0.9,n_rep=n_rep)

            #========================================



            fig,axarr = plt.subplots(1,T+1,figsize=[10,5])
            for lag,ax in zip(range(T),axarr.flat):
                ax.scatter(res[:,2*lag],res[:,2*lag+1])
                ax.set_xlabel('w')
                ax.set_ylabel('decay')
                ax.set_title('lag '+str(lag+1) )
            axarr[-1].scatter(res[:,-2],res[:,-1])
            axarr[-1].set_xlabel('lin')
            axarr[-1].set_ylabel('lik')
            plt.tight_layout()
            plt.savefig(save_prefix+'scatter_'+save_name)
            plt.close()




            xmin, xmax = x[:,1].min(),x[:,1].max()
            xp = np.linspace(xmin,xmax,100)
            col = 'rgb'
            fig, ax = plt.subplots()
            #plt.hist(x[:,1:].flatten(),bins=100,alpha=0.2,normed=True)


            for i_r,r in enumerate(res):
                p_add,_,_ = mod.unflat_param(r)
                for lag in range(T):
                    kwargs = {'label':'lag'+str(lag+1)} if i_r==0 else {}
                    ax.plot(xp,mod.bias(p_add[lag])(xp),col[lag],**kwargs)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)

            plt.savefig(save_prefix+'curves_'+save_name)

            plt.close()


            plt.close('all')