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
np.random.seed(0)
import matplotlib.cm as cm
from GPflow.kernels import Antisymmetric, RBF, Linear
import time



#==== load data ====

DATA_PATH = '../../data_files/'
OUTPUT_PATH = '../../output/'
experiment = 'continuous'

timestr = time.strftime("%Y%m%d-%H%M%S")


mat = sio.loadmat(DATA_PATH + experiment)
s1_all = np.log(np.asarray((mat['s1']))).astype(float)
s2_all = np.log(np.asarray((mat['s2']))).astype(float)
resp_all = np.asarray(mat['resp']) + 0.
acc_all = np.asarray(mat['acc']) + 0.


# filter data
ranges = {
    'poor':[0.6,0.82],
    'good':[0.85,1]
}


typ = 'good'

filt = ranges[typ]
I = np.where((acc_all.mean(1)>filt[0]) & (acc_all.mean(1)<filt[1]))[0]

print len(I)
s1 = s1_all[I,:]
s2 = s2_all[I,:]
resp = resp_all[I,:]
acc = acc_all[I,:]

sm = (s1 + s2)*.5
diff = (s1 - s2)[:,1:]
prev = s1[:,1:]-sm[:,:-1]
inf = s1[:,1:] - s1.mean()
Y = resp[:,1:].astype(int)
acc = acc[:,1:]

s1 = s1[:,1:]
s2 = s2[:,1:]

a,b = Y.shape
diff = np.reshape(diff,[a*b,1])
prev = np.reshape(prev,[a*b,1])
inf = np.reshape(inf,[a*b,1])
Y = np.reshape(Y,[a*b,1])
acc = np.reshape(acc,[a*b,1])

s1 = np.reshape(s1,[a*b,1])
s2 = np.reshape(s2,[a*b,1])



select = np.where( (np.abs(prev)>0.25)&(np.abs(prev)<0.45)&\
                  (np.abs(diff)<.1))
diff = diff[select[0]]
inf = inf[select[0]]
prev = prev[select[0]]
Y =Y[select[0]]
acc = acc[select[0]]
s1 = s1[select[0]]
s2 = s2[select[0]]


X = np.hstack([diff,prev,inf])

print X.shape,Y.shape

bins = 20
nbin,edges =np.histogram(diff,bins=bins)

I = np.argsort(diff.flat)
i_bins = [l for l in np.array_split(I, bins)]
Y_b = [Y[i] for i in i_bins]
diff_b = [diff[i] for i in i_bins]

xs = np.array([d.mean() for d in diff_b])
ys = np.array([y.mean() for y in Y_b])


fig, axarr = plt.subplots(1,4,figsize=(20,5))
axarr[0].plot(xs,ys)
axarr[0].set_title('phi(df)')
axarr[1].hist(diff)
axarr[1].set_title('df')
axarr[2].hist(prev)
axarr[2].set_title('d1')
axarr[3].hist(inf)
axarr[3].set_title('dinf')
plt.show()


Iplus = np.where( prev*diff < 0 )[0]
Iminus = np.where( prev*diff >= 0 )[0]

print len(diff), len(Iplus),len(Iminus)

print '+:',acc[Iplus].mean(),'|-:',acc[Iminus].mean(),'|all:',acc.mean()


I1inf_plus = np.where(prev*inf>0)
I1inf_minus = np.where(prev*inf<0)

print 'd1*dinf>0:',acc[I1inf_plus].mean(),'|d1*dinf<0:',acc[I1inf_minus].mean()

Ipm = np.where( (prev*inf>0)&( prev*diff < 0 ))[0]
Ipp = np.where( (prev*inf>0)&( prev*diff > 0 ))[0]
Imm = np.where( (prev*inf<0)&( prev*diff < 0 ))[0]
Imp = np.where( (prev*inf<0)&( prev*diff > 0 ))[0]

print 'Ipm ,Ipp,Imm,Imp:',acc[Ipm].mean(),acc[Ipp].mean(),acc[Imm].mean(),acc[Imp].mean()
print 'Ipm ,Ipp,Imm,Imp:',len(Ipm),len(Ipp),len(Imm),len(Imp)

