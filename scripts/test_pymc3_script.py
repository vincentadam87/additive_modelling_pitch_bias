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


samples=1000
T = 3
inf = 0

dataloader = Dataloader(DATA_PATH+'continuous.mat')
F1,F2,Y = dataloader.subject_data(range(10),acc_filter=(0.65,0.8))
x,y = get_trial_covariates(F1,F2,Y,T=T)

print x.shape,y.shape

m = additive_model_lin_exp()
trace, dic = m.fit(x,y,mcmc_samples=samples)
xp,m_,s_ = m.posterior_summary(trace)

for lag in range(T):
    plt.fill_between(xp,y1=(m_-s_)[lag,:],y2=(m_+s_)[lag,:], alpha=0.5)
    plt.plot(xp,m_[lag,:])
plt.show()

plt.savefig(OUTPUT_PATH  + "test.svg")
