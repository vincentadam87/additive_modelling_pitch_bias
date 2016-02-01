import numpy as np          
from scipy.optimize import minimize
from inference import *
from generate_stimuli import *
from matplotlib import pyplot as plt
from scipy.special import erf


#TODO: Comment, I do not understand
def simulate(data, simulation_param):
    
    experiment = simulation_param[0]
    dfrange = simulation_param[1]
    frange = simulation_param[2]
    samples = simulation_param[3]

    '===== sample s1,s2 ====='
    s1 = np.empty((1, samples))
    s2 = np.empty((1, samples))
    resp = np.ones((1, samples))
    acc = np.ones((1, samples))
   
    if experiment == "unimodal":       
        s1[0],s2[0] = sample_s1_s2_unimodal(samples,frange,dfrange)
    elif experiment == "bimodal":
        s1[0],s2[0] = sample_s1_s2_bimodal(samples,frange,dfrange)

        
    sigma0_eta0 = np.array([0.03,0.1])
    '===== fit raviv ====='
    pars = minimize(loss_raviv,sigma0_eta0, (data['s1'][0], data['s2'][0], data['resp'][0],'loss'),
            method='nelder-mead', options={'xtol': 1e-8, 'disp': True})    
          
    '===== simulate raviv ====='
    resp[0] = loss_raviv(pars['x'],s1[0],s2[0],resp[0],'resp')
    resp[0]  = np.random.binomial(resp[0] > -np.inf, resp[0])
    data['s1'] = s1; data['s2'] = s2; data['acc'] = acc; data['resp'] = resp

    return data


def loss_raviv(sigma_eta,f1,f2,resp,output):
    p_resp = np.empty(len(resp))
    sigma = sigma_eta[0]
    eta = sigma_eta[1]
    I = f1[0]
    p_resp[0] = f1[0]>f2[0]
    for ii in range(1,len(f1)):
        Sigma = sigma*np.sqrt(eta**(2*(ii+1)-2) + (1-eta**(2*(ii+1)-2))*(1-eta)/(1+eta))     
        I = eta*I + (1-eta)*(f1[ii])
        p_resp[ii] = phi(I-f2[ii],0,Sigma)
    if output == 'loss':
        return -(np.sum(np.log(0.001 + 1-np.abs(resp-p_resp)))) + 10000*np.abs(eta)*(eta<0) + 10000*np.abs(eta)*(eta>1) 
    if output == 'resp':
        return p_resp
        
        
def simulate_realistic_model():
    phi = lambda x : 0.5*(1.+erf(x/np.sqrt(2.)))
    bias1d = lambda r1,r2 : lambda x : r1*x*np.exp(-np.abs(x)/(r2))

    dfmax = 1




def generate_vincent(s1,s2,sigma,Color):
    """
    For given f1, f2 simulate responses of a model
    """

    prev = 0.5*(s1[0:-1] + s2[0:-1])    # mean of previous trials
    f1 = s1[1:]; f2 = s2[1:]
    df = f1-f2 # trial difficulty
    N = len(f1)
    # bias function
    predictor = lambda r1,r2 : 1/sigma*df + r1*x*np.exp(-abs(x)/sigma*0.55)
    B = phi(predictor(20,0.08)(prev))
    Y = np.random.rand(N)<B
    Y = np.concatenate(([1],Y), axis=0)   #TODO: why add a 1?
    return np.reshape(Y,(1,len(Y)))
    
    
def phi(x,alpha=0,sigma=1,lam=0):
    '''   lam/2 + (1-lam) + Norm.cdf (  x/sigma - alpha   ) '''
    return 0.5*lam +(1-lam)*( 0.5 * (1 + sc.special.erf((x - alpha) / (sigma * np.sqrt(2)))) )    

              