import sys
import os
import Globals
from matplotlib import pyplot as plt

OUTPUT_PATH = os.path.expanduser('~')+'/'+'Python_outputs'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_PATH = "/home/ubuntu-lieder-pc/Python_outputs/"
CODE_PATH = Globals.path_dic('code')
DATA_PATH = Globals.path_dic('data')

CODE_PATH = "/home/ubuntu-lieder-pc/git/additive_modelling_pitch_bias/"
DATA_PATH = CODE_PATH + "data_files/"
print CODE_PATH
# Make sure the data files are named: 'continuous.mat'
# and 'bimodal.mat' for each respective experiment
sys.path.append(CODE_PATH)

from data_loader.load_data import *
from inference_toolboxes.pymc3_functions.inference_pymc3 import *


samples=10000
T = 1
inf = 0

dataloader = Dataloader(DATA_PATH +'continuous.mat')

groups = 3
acc_sample = np.array([[0.63,0.75],[0.75,0.88],[0.88,1]])
color_mat = ['b','g','r']
dic = np.empty([groups])

for ii in range(groups):
    F1,F2,Y = dataloader.subject_data(range(10),acc_filter=acc_sample[ii,:])
    x,y = get_trial_covariates(F1,F2,Y,T=T)

    print x.shape,y.shape

    m = additive_model_lin_exp()
    trace,model = m.fit(x,y,mcmc_samples=samples)
    xp,m_,s_ = m.posterior_summary(trace)

    for lag in range(T):
        plt.fill_between(xp,y1=(m_-s_)[lag,:],y2=(m_+s_)[lag,:],color = color_mat[ii],alpha=0.5)
        plt.plot(xp,m_[lag,:],color = color_mat[ii])
plt.show()
plt.savefig(OUTPUT_PATH  + "test.svg")
print dic