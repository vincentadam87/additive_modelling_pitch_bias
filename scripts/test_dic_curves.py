import sys
import os
import Globals
from matplotlib import pyplot as plt
import pymc3 as pm

OUTPUT_PATH = os.path.expanduser('~')+'/'+'Python_outputs'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_PATH = "/home/ubuntu-lieder-pc/Python_outputs/"
CODE_PATH = Globals.path_dic('code')
DATA_PATH = Globals.path_dic('data')

# Make sure the data files are named: 'continuous.mat'
# and 'bimodal.mat' for each respective experiment
sys.path.append(CODE_PATH)

from data_loader.load_data import *
from inference_toolboxes.pymc3_functions.inference_pymc3 import *


samples=1000
T = 1
inf = 0

dataloader = Dataloader(DATA_PATH+'continuous.mat')
F1,F2,Y = dataloader.subject_data(range(10),acc_filter=(0.92,1))
x,y = get_trial_covariates(F1,F2,Y,T=T,inf=inf)
print 'x:',x.shape
print x.shape,y.shape

m = additive_model_lin_exp()

decay_0 = []

for ii in range(1):
    trace, model = m.fit(x,y,mcmc_samples=samples,decay_0=decay_0)
    decay = trace['decay']
    a, b, _ = plt.hist(decay)
    decay_0 = b[a.argmax()]
    print decay_0
    xp,m_,s_ = m.posterior_summary(trace)
    plt.figure()
    for lag in range(T+inf):
        plt.fill_between(xp,y1=(m_-s_)[lag,:],y2=(m_+s_)[lag,:], alpha=0.5)
        plt.plot(xp,m_[lag,:])
    plt.savefig(OUTPUT_PATH  + "DIC:_" + str(np.round(pm.dic(trace,model))) + ".svg")


decay = trace['decay']
w = trace['w']

plt.figure()
plt.scatter(decay[:,0], w[:,0])
# plt.imshow(H)
plt.savefig(OUTPUT_PATH  + "im.svg")