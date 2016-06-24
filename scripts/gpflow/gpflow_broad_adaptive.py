"""
Model free analysis of adaptive dataset

"""
from gpflow_functions import *
from GPflow.kernels import Linear,Antisymmetric_RBF, RBF

from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np
import time, pickle, os

DATA_PATH = '../../data_files/'
OUTPUT_PATH = '../../output/'
timestr = time.strftime("%Y%m%d-%H%M%S")
timestr = 'test6_crop'
foldername = OUTPUT_PATH+'BROAD_ADAPTIVE-'+timestr
if not os.path.exists(foldername):
    os.makedirs(foldername) # one experiment = one folder

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


bins = 3
l_grid = np.linspace(.5,.5,1)
model_types = ['add','joint']

#======================== Separate subjects per-alpha on block 2

alphas = np.zeros((ns,))
for i_s in range(ns):
    X = np.expand_dims( (s2-s1)[i_s,I2] ,1)
    Y = resp[i_s,I2].astype(float)
    clf = LogisticRegression(C=1e10,penalty='l2')
    clf = clf.fit(X, Y)
    alphas[i_s] = clf.coef_[0][0]


#======================== Filter bad subjects < .6 on block 1

def get_covariates(s1,s2,resp):

    lag = 2
    sm = (s1 + s2)*.5
    diff = (s1 - s2)[:,lag:] # f1 - f2 at trial time
    prev1 = s1[:,lag:]-sm[:,1:-lag+1] # f1(t) - (f1+f2)(t-1)
    inf = s1[:,lag:] - s1.mean() # f1(t) - overallmean(f1)
    Y = resp[:,lag:].astype(int)

    a,b = Y.shape
    diff = np.reshape(diff,[a*b,1])
    prev1 = np.reshape(prev1,[a*b,1])
    inf = np.reshape(inf,[a*b,1])
    Y = np.reshape(Y,[a*b,1])

    X = np.hstack([diff,prev1,inf])
    cov_names = ['$df$', '$d_1$', '$d_{\infty}$']
    cov_to_index = {s:i for i,s in enumerate(cov_names)}

    # further filtering
    Itrial = np.where((np.abs(prev1)<.9))[0]# subselecting
    X,Y = X[Itrial,:],Y[Itrial,:]

    return X,Y


acc = acc_all[:,I1].mean(1)
Is = np.where( (acc>.6) )[0]

I = np.argsort(alphas[Is])
i_bins = [Is[l] for l in np.array_split(I, bins)]
alphas_b = [alphas[i] for i in i_bins]


#===========================
from matplotlib import pyplot as plt
from matplotlib import cm
colors = cm.rainbow(np.linspace(0, 1, bins))

for i_bin,Ibin in enumerate(i_bins):
    for Ib,marker in zip([I1,I2],['x','o']):
        plt.plot(acc_all[Ibin,:][:,I1].mean(1),acc_all[Ibin,:][:,I2].mean(1),'x',color=colors[i_bin],label=str(i_bin))
plt.plot([0.5,1],[0.5,1])
plt.legend()
plt.savefig('acc_all.png')
assert False
#================ fit model for each group

def model_add(X,Y,l_d1,l_dinf):
    '''
    :param l_d1: lengthscale for d1
    :param l_dinf: lengthscale for dinf
    :return:
    '''
    # additive structure
    f_indices = [[0],[1],[2]]
    # Inducing point locations
    Nz =40
    N = len(Y)
    Z = [np.array([[1]]) ,
         np.expand_dims( np.linspace(X[:,1].min(),X[:,1].max(),Nz) ,1),
         np.expand_dims( np.linspace(X[:,2].min(),X[:,2].max(),Nz) ,1)]
    # Setting kernels
    ks = [Linear(1),
          Antisymmetric_RBF(1,lengthscales=l_d1,variance=1.),
          Antisymmetric_RBF(1,lengthscales=l_dinf,variance=1.)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1)+f(dinf)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.Z[1].fixed = True
    m.Z[2].fixed = True
    m.kerns.parameterized_list[0].variance.fixed = True
    m.kerns.parameterized_list[1].variance.fixed = True
    m.kerns.parameterized_list[2].variance.fixed = True
    m.kerns.parameterized_list[1].lengthscales.fixed = True
    m.kerns.parameterized_list[2].lengthscales.fixed = True
    return m

def model_joint(X,Y,l_d1,l_dinf):
    '''
    :param l_d1: lengthscale for d1
    :param l_dinf: lengthscale for dinf
    :return:
    '''
    # additive structure
    f_indices = [[0],[1,2]]
    # Inducing point locations
    Nz = 10
    N = len(Y)
    z1 = np.linspace(X[:,1].min(),X[:,1].max(),Nz)
    z2 = np.linspace(X[:,2].min(),X[:,2].max(),Nz)
    Z1,Z2 = np.meshgrid(z1,z2)
    Z12= np.vstack([Z1.flatten(),Z2.flatten()]).T
    print Z12.shape
    Z = [np.array([[1]]) ,Z12]
    # Setting kernels
    ks = [Linear(1),
          Antisymmetric_RBF(2,lengthscales=[l_d1,l_dinf], variance=1.,ARD=True)]
    # Declaring model
    m = SVGP_additive2(X, Y, ks, Bernoulli(), Z,\
                        f_indices=f_indices,name = 'bias f(d1)+f(dinf)')
    m.Z[0].fixed = True # no need to optimize location for linear parameter
    m.Z[1].fixed = True
    m.kerns.parameterized_list[0].variance.fixed = True
    m.kerns.parameterized_list[1].variance.fixed = True
    m.kerns.parameterized_list[1].lengthscales.fixed = True
    return m

L1,Linf = np.meshgrid(l_grid,l_grid)
L1 = L1.flatten()
Linf = Linf.flatten()


for i_type,m_type in enumerate(model_types):

    # iterate over block
    for i_block, Ib in enumerate([I1,I2]):

        # iterate over accuracies
        for i_bin,I in enumerate(i_bins):

            X,Y = get_covariates(s1[I,:][:,Ib],
                                 s2[I,:][:,Ib],
                                 resp[I,:][:,Ib])

            postfix = str(i_bin)+'-'+str(i_block)+'-'+m_type
            print postfix

            res = {}

            if not os.path.exists(foldername+'/results_'+postfix+'.p'):
                # iterate over lengthscales
                for i_l in range(len(L1)):

                    print '-----------------',i_l,len(L1)

                    l1,linf = L1[i_l],Linf[i_l]


                    if m_type == 'add':
                        m = model_add(X,Y,l1,linf)
                    elif m_type == 'joint':
                        m = model_joint(X,Y,l1,linf)
                    success = False
                    while success == False:
                        r =m.optimize()
                        print r['success']
                        success = r['success']


                    # saving output of model fit:
                    prms = m.extract_params()
                    prms['f_indices'] = m.f_indices
                    b= m.lower_bound_likelihood()
                    prms['lb'] = b
                    prms['subjects']=len(I)

                    # saving fitted models
                    #pickle.dump(prms,open(foldername+'/models'+postfix+'.p','wb'))

                    # computing the individual function values
                    try:
                        mu_fs,v_fs = m.predict_fs(X)
                    except Exception:
                        mu_fs,v_fs=np.zeros(Y.shape),np.zeros(Y.shape)
                    print mu_fs.shape

                    predictions = {
                        'X':X,
                        'Y':Y,
                        'Z':prms['Z'],
                        'f_indices':m.f_indices,
                        'mean':mu_fs,
                        'nsubject':len(I),
                        'variance':v_fs,
                    }

                    res[i_l]={'l1':l1,
                              'linf':linf,
                              'predictions':predictions,
                              'model':prms}
                    del m

                pickle.dump(res,open(foldername+'/results_'+postfix+'.p','wb'))
            else:
                print 'skip!'



