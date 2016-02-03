'''
This script performs maximum likelihood estimates of the
'''


from model_simulations.descriptive_model import *
from utils import *
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
# ---------------------- Additive Linear Exp ---------------------

isig = lambda x : -np.log(1./x-1)

if __name__ == '__main__':


    # declare model parameters for simulations


    n =5000
    d = 2

    prm_lin = 1./0.02 # linear parameter
    prm_lik =  .01 # parameter of the likelihood

    for sim_type,fit_type in [['LinExp','LinExp'],['Lin','LinExp'],['LinExp','Lin'],['Lin','Lin']]:


        if sim_type == 'LinExp':
            peaks = [0.1,0.5]
            maxs = [1,2]
            prm_add = [[m*1./peak*np.exp(1), 1./peak] for peak, m in zip(peaks,maxs)]
            mod_sim = LinExpAdditiveModel()
            mod_sim.__init__(prm_add,prm_lin,prm_lik)

        elif sim_type == 'Lin':
            prm_add = [[2],[1]]
            mod_sim = LinAdditiveModel()
            mod_sim.__init__(prm_add,prm_lin,prm_lik)

        if fit_type == 'Lin':
            mod_fit = LinAdditiveModel()
            w0 = np.zeros((d+2,))-5
        elif fit_type == 'LinExp':
            mod_fit = LinExpAdditiveModel()
            w0 = np.zeros((2*d+2,))-5

        w = mod_sim.flat_prm(prm_add,prm_lin,prm_lik)


        # sampling
        x= (2*np.random.rand(n,d+1)-1)

        x[:,0]*=.1
        #x[:,1:-1]/=2.
        b = mod_sim.bern(x)
        y = (np.random.rand(n)<b).astype(int)

        # learning
        res = ml_fit(x,y,w0,mod_fit,prop_bstrap=0.9,n_rep=10,n_restart=5)

        #----------------------------------------------------------------------------


        plt.figure()
        plt.hist(b)
        plt.savefig('sim_'+sim_type+'_'+fit_type+'_ml-fit0.svg')
        plt.close()


        fig,axarr = plt.subplots(1,d+1,figsize=[10,5])
        for lag,ax in zip(range(d),axarr.flat):
            ax.scatter(res[:,2*lag],res[:,2*lag+1])
            ax.plot(w[2*lag],w[2*lag+1],'xr')
            #ax.set_xlim([0,2])
            #ax.set_ylim([0,2])
            ax.set_xlabel('w')
            ax.set_ylabel('decay')
            ax.set_title('lag '+str(lag+1) )
        axarr[-1].scatter(res[:,-2],res[:,-1])
        axarr[-1].plot(w[-2],w[-1],'xr')
        #axarr[-1].set_xlim([0,2])
        #axarr[-1].set_ylim([0,.3])
        axarr[-1].set_xlabel('lin')
        axarr[-1].set_ylabel('lik')
        plt.tight_layout()
        plt.savefig('sim_'+sim_type+'_'+fit_type+'_ml-fit1.svg')
        plt.close()




        xmin, xmax = x.min(),x.max()
        xp = np.linspace(xmin,xmax,100)
        col = 'rgb'
        fig, ax = plt.subplots()
        plt.hist(x[:,1:-1].flatten(),bins=100,alpha=0.2,normed=True)


        for i_r,r in enumerate(res):
            p_add,p_lin,p_lik = mod_fit.unflat_param(r)
            for lag in range(d):
                kwargs = {'label':'lag'+str(lag+1)} if i_r==0 else {}
                mod_fit.set_params_uc(p_add,p_lin,p_lik)
                ax.plot(xp,mod_fit.a[lag](xp),col[lag],**kwargs)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        for lag,p in zip(range(d),prm_add):
            ax.plot(xp,mod_sim.a[lag](xp),col[lag],\
                    linestyle='--',\
                   linewidth=2,path_effects=[path_effects.Normal(),\
                                             path_effects.SimpleLineShadow()])

        plt.savefig('sim_'+sim_type+'_'+fit_type+'_ml-fit2.svg')
        plt.close()


        plt.close('all')