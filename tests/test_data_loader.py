from Itay.load_data import *
import platform


node = platform.node()
if node == 'vincent-ThinkPad-T450s':
    PATH = '/home/vincent/Dropbox/Gatsby/Shared_Itay_Vincent/Python_vincent_itay/Itay/'
else: # itay
    PATH = 'C:/Users/User/Dropbox/Shared_Itay_Vincent/Python_vincent_itay/Itay/'


fname = PATH+'data/continuous.mat'
loader = Dataloader(fname)

F1,F2,Y = loader.subject_data([0])
print F1.shape
X = get_trial_covariates(F1,F2)