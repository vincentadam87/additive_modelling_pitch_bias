"""
Model free analysis of adaptive dataset

"""
from gpflow_functions import *

from scipy.io import loadmat
from scipy.special import erf
from numpy import sqrt,log,exp,expand_dims
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

DATA_PATH = '../../data_files/'

fname = 'wide_range_adaptive.mat'
data = loadmat(DATA_PATH+fname)

print data.keys()

acc_all = data['acc']
s1 = data['s1']
s2 = data['s2']
resp = data['resp']
print acc_all.shape

#=========================================




ns,nt = acc_all.shape

#======================= Accuracy per block
# divide blocks
I1,I2 = np.arange(0,300), np.arange(300,600)
for I in [I1,I2]:
    print acc_all[:,I].mean(1)


#======================= simple logit fit



alphas = np.zeros((ns,2))
accs = np.zeros((ns,2))
for i_s in range(ns):
    for i_b,I in enumerate([I1,I2]):
        X = expand_dims( (s2-s1)[i_s,I] ,1)
        Y = resp[i_s,I].astype(float)
        clf = LogisticRegression(C=1e10,penalty='l2')
        clf = clf.fit(X, Y)
        alphas[i_s,i_b] = clf.coef_[0][0]
        accs[i_s,i_b] = acc_all[i_s,I].mean()


fig,axarr=plt.subplots(2,2)
ax=axarr.ravel()[0]
ax.plot(accs[:,0],accs[:,1],'x')
ax.plot(accs[:,0],accs[:,0],'k-')
ax.set_xlabel('accuracy block 1')
ax.set_ylabel('accuracy block 2')
ax.set_title('N='+str(ns))
ax = axarr.ravel()[1]
ax.plot(alphas[:,0],alphas[:,1],'x')
ax.plot(alphas[:,0],alphas[:,0],'k-')
ax.set_xlabel('alpha block 1')
ax.set_ylabel('alpha block 2')
ax = axarr.ravel()[2]
ax.plot(accs[:,0],alphas[:,0],'x')
ax.set_xlabel('accuracy block 1')
ax.set_ylabel('alpha block 1')
ax = axarr.ravel()[3]
ax.plot(accs[:,1],alphas[:,1],'x')
ax.set_xlabel('accuracy block 2')
ax.set_ylabel('alpha block 2')

fig.tight_layout()
plt.savefig('adaptive_broad.svg')

plt.close()


#================== Check response probabilities
'''
fig,axarr = plt.subplots(2,1)

Ps,Ys = [],[]
for i_b,I in enumerate([I1,I2]):
    ps,ys = [],[]
    ax = axarr.ravel()[i_b]
    for i_s in range(ns):
        X = expand_dims( (s2-s1)[i_s,I] ,1)
        Y = resp[i_s,I].astype(float)
        p = LogisticRegression(C=1e10).fit(X, Y).predict_proba(X)[:,1]

        ps.append(p)
        ys.append(Y)
    ps = np.asarray(ps).flatten()
    ys = np.asarray(ys).flatten()
    ax.hist(ps,bins=80)
    ax.set_title('block '+str(i_b+1))
    ax.set_ylabel('p(y=1)')
    Ps.append(ps)
    Ys.append(ys)

fig.tight_layout()
plt.savefig('histogram_p.svg')


#================= histogram of s1,s2

fig, axarr = plt.subplots(2,2)
for i_b, I in enumerate([I1,I2]):
    axarr[0,i_b].hist(s1[:,I].flatten(),bins=20)
    axarr[1,i_b].hist((s1-s2)[:,I].flatten(),bins=20)
    axarr[0,i_b].set_title('block '+str(i_b))
plt.savefig('stim_distribution.svg')
'''

#================ some 2d plots




