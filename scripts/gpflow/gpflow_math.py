import numpy as np
from scipy.special import erf
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import GPflow.kernels
from GPflow.likelihoods import Bernoulli, Gaussian
from GPflow.svgp import SVGP
from GPflow.svgp_additive import SVGP_additive2
import scipy.io as sio
from gpflow_functions import *


 #==== load data ====
DATA_PATH = '/home/dell/Dropbox/additive_modelling_pitch_bias/data_files/'
experiment = 'passive_encoding'

mat = sio.loadmat(DATA_PATH + experiment)
s1 = np.asarray((mat['s1']))
s2 = np.asarray((mat['s2']))
resp = np.asarray(mat['resp']) + 0.
math = np.asarray(mat['math']) + 0.
acc = np.asarray(mat['acc']) + 0.
[nSub,nTrials] = s1.shape


[nSub,nTrials] = s1.shape
s1_ = s1[:,1:] - 0.5*(s1[:,:-1] +s2[:,:-1])
s2_ = s2[:,1:] - 0.5*(s1[:,:-1] +s2[:,:-1])


s1_inf = s1[:,1:] - np.log(1000)
s2_inf = s2[:,1:] - np.log(1000)

resp_ = resp[:,1:]

diff = (s1_ - s2_).flatten()
prev = 0.5*(s1_ + s2_).flatten()
inf = 0.5*(s1_inf + s2_inf).flatten()
resp_ = resp_.flatten()

select = np.where(  ((math[:,1:].flatten()) ==55 ) &
                  ((math[:,:-1].flatten()) != 55)     )
diff = diff[select[0]]
inf = inf[select[0]]
prev = prev[select[0]]
resp_ = resp_[select[0]]


X = np.vstack([diff,prev, inf]).transpose()
Y = np.random.rand(len(prev.flatten()),1)
Y[:,0] = resp_.flatten().transpose()

print X.shape
print Y.shape
D = 3
m, n_func, f_indices, Ys, Vs = gaussian_additive_2d(X,Y,D)

plot_2d(X, n_func,f_indices, Ys,Vs,D)


m, n_func, f_indices, Ys, Vs = gaussian_additive_1d(X,Y,D)

plot_1d(X, n_func,f_indices, Ys,Vs,D)