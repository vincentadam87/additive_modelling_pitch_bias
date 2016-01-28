import sys
import os
import Globals
from matplotlib import pyplot as plt

OUTPUT_PATH = os.path.expanduser('~')+'/'+'Python_outputs'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

CODE_PATH = Globals.path_dic('code')
DATA_PATH = Globals.path_dic('data')

# Make sure the data files are named: 'continuous.mat'
# and 'bimodal.mat' for each respective experiment
sys.path.append(CODE_PATH)

from data_loader.load_data import *
from inference_toolboxes.pymc3_functions.inference_pymc3 import *


samples=50
T = 2
inf = 1


dataloader = Dataloader(DATA_PATH+'continuous.mat')
F1,F2,Y = dataloader.subject_data(range(10))
x,y = get_trial_covariates(F1,F2,Y,T=T)

print x.shape,y.shape

m = additive_model_lin_exp()
trace=m.fit(x,y,mcmc_samples=samples)
xp,m_,s_ = m.posterior_summary(trace)

for lag in range(T):
    plt.fill_between(xp,y1=(m_-s_)[lag,:],y2=(m_+s_)[lag,:])
plt.show()