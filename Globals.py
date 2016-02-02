import platform



def path_dic(k):
    node = platform.node()
    if node == 'vincent-ThinkPad-T450s':
        return {'code':'/home/vincent/Dropbox/Gatsby/Shared_Itay_Vincent/Python_vincent_itay/Itay/',
                'data':'/home/vincent/git/additive_modelling_pitch_bias/data_files/'}[k]
    elif node == 'dell-XPS-15-9530':
        return {'code':'/home/dell/git/additive_modelling_pitch_bias/',
                'data':'/home/dell/git/additive_modelling_pitch_bias/data_files/'}[k]
    elif node == 'ubuntuliederpc-desktop':
        return {'code':'/home/ubuntu-lieder-pc/git/additive_modelling_pitch_bias/',
                'data':'/home/ubuntu-lieder-pc/git/additive_modelling_pitch_bias/data_files/'}[k]
    else:
        return None