import yaml
import types
import astropy.cosmology
from skylens.wigner_transform import *
from skylens.utils import *
from skylens.survey_utils import *
import astropy.cosmology
import importlib
from importlib import import_module
d2r=np.pi/180.


corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')


skylens_args_names=['yaml_inp_file','l','Ang_PS',
                'cov_utils','logger','tracer_utils',
                'shear_zbins','kappa_zbins','galaxy_zbins',
                'pk_params','cosmo_params','WT_kwargs','WT',
                'z_PS','nz_PS','log_z_PS',
                'do_cov','SSV_cov','tidal_SSV_cov','do_sample_variance',
                'Tri_cov','sparse_cov',
                'use_window','window_lmax','window_l','store_win','Win',
                'f_sky','wigner_step','cl_func_names','l_bins_center',
                'l_bins','bin_cl','use_binned_l','do_pseudo_cl',
                'stack_data','bin_xi','do_xi','theta_bins','theta_bins_center',
                'use_binned_theta','xi_SN_analytical', #xi_win_approx
                'corrs','corr_indxs','stack_indxs',
                'wigner_files','name','clean_tracer_window',
                'scheduler_info']

def parse_dict(dic={}):
    """
    Get skylens input arguments from a dictionary.
    """
    skylens_kwargs={}
    for k in skylens_args_names:
        skylens_kwargs[k]=dic.get(k)
        if isinstance(skylens_kwargs[k],types.ModuleType):
            skylens_kwargs[k]=None #remove objects obtained from imports
    return skylens_kwargs
        
def parse_python(file_name=None):
    """
    Run the given python file and then get the skylens
    arguments from it.
    """
    exec(open(file_name).read())
    dic=locals()
    return parse_dict(dic=dic)

def parse_yaml(file_name=''):
    """
    Get skylens arguments from a yaml file.
    """
    with open(file_name) as file:
        skylens_kwargs=yaml.load(file, Loader=yaml.FullLoader)
    
    skylens_kwargs=set_cluster(skylens_kwargs=skylens_kwargs)
    
    skylens_kwargs=set_corrs(skylens_kwargs)   
    
    skylens_kwargs=eval_funcs(skylens_kwargs)   

    skylens_kwargs=set_l_bins(skylens_kwargs=skylens_kwargs)
    
    skylens_kwargs=set_theta_bins(skylens_kwargs=skylens_kwargs)
    
    skylens_kwargs=set_WT(skylens_kwargs=skylens_kwargs)
    
    skylens_kwargs=set_cosmology(skylens_kwargs=skylens_kwargs)
    
#     skylens_kwargs=cleanup(skylens_kwargs)
    
    return parse_dict(dic=skylens_kwargs)

# def cleanup(skylens_kwargs): #remove unwanted args ... not necesasry since we use parse_dict now
#     keys=['Nl_bins','l_bin_min','l_bin_max','log_bins']
#     keys+=['theta_min','theta_max','n_theta_bins','theta']
#     keys+=['start_new_cluster','scheduler_args']
#     if skylens_kwargs['clean_keys'] is not None:
#         keys+=skylens_kwargs['clean_keys']
#     for k in keys:
#         del skylens_kwargs[k]
#     return skylens_kwargs

def eval_funcs(skylens_kwargs):
    """
    evaluate the functions passed in a yaml file.
    We require that functions be passed as eval..func(args)
    """
    for k in skylens_kwargs.keys():
        if isinstance(skylens_kwargs[k],dict):
            skylens_kwargs[k]=eval_funcs(skylens_kwargs[k])
        elif isinstance(skylens_kwargs[k],str):
            if 'eval..' in skylens_kwargs[k]:
                func_str=skylens_kwargs[k].split('..')
                print('eval str',func_str)
                func_str=func_str[1]
                skylens_kwargs[k]=eval(func_str)
    return skylens_kwargs

def test_func(skylens_kwargs):
    return None

def set_corrs(skylens_kwargs={}):
    """
    Yaml doesn't allow to pass a list of tuples cleanly.
    This function sorts it out, to give a list of
    correlation pairs.
    """
    shear='shear'
    galaxy='galaxy'
    kappa='kappa'
    corrs2=[]
    for c in skylens_kwargs['corrs']:
        corrs2+=[eval(c)]
    skylens_kwargs['corrs']=corrs2
    return skylens_kwargs 

def set_l_bins(skylens_kwargs={}):
    """
        set the ell and ell_bin values from arguments passed in yaml
    """
    keys=['l','l_bins']
    if skylens_kwargs['l'] is None:
        skylens_kwargs['l']=np.arange(skylens_kwargs['l_bin_min'],skylens_kwargs['l_bin_max'])
    if skylens_kwargs['l_bins'] is None:
        if log_bins:
            skylens_kwargs['l_bins']=np.logspace(np.log10(skylens_kwargs['l_bin_min']),
                                                 np.log10(skylens_kwargs['l_bin_max']),skylens_kwargs['Nl_bins']+1)
        else:
            skylens_kwargs['l_bins']=np.linspace(skylens_kwargs['l_bin_min'],
                                                 skylens_kwargs['l_bin_max'],skylens_kwargs['Nl_bins']+1)
        skylens_kwargs['l_bins']=np.int64(skylens_kwargs['l_bins'])
    skylens_kwargs['l_bins']=np.unique(skylens_kwargs['l_bins'])
    return skylens_kwargs
    
def set_theta_bins(skylens_kwargs={}):
    """
        set the theta and theta_bin values from arguments passed in yaml
    """
    if skylens_kwargs['theta_bins'] is None:
        if log_bins:
            skylens_kwargs['theta_bins']=np.logspace(np.log10(skylens_kwargs['theta_min']),
                        np.log10(skylens_kwargs['theta_max']),skylens_kwargs['n_theta_bins']+1)
        else:
            skylens_kwargs['theta_bins']=np.linspace(skylens_kwargs['theta_min'],
                                    skylens_kwargs['theta_max'],skylens_kwargs['n_theta_bins']+1)
    return skylens_kwargs
    
def set_WT(skylens_kwargs={}):
    """
        Set the arguments for wigner_transform
    """
    if not skylens_kwargs['do_xi']:
        return skylens_kwargs
    if skylens_kwargs['WT_kwargs'].get('theta') is None:
        if log_bins:
            skylens_kwargs['WT_kwargs']['theta']=np.logspace(np.log10(skylens_kwargs['theta_min']*.99),
                    np.log10(skylens_kwargs['theta_max']*1.01),skylens_kwargs['n_theta_bins']*10)
        else:
            skylens_kwargs['WT_kwargs']['theta']=np.linspace(skylens_kwargs['theta_min']*.99,
                                skylens_kwargs['theta_max']*1.01,skylens_kwargs['n_theta_bins']*10)
    s1_s2_new=[]
    skylens_kwargs['WT_kwargs']['theta']=skylens_kwargs['WT_kwargs']['theta']*d2r
    for i in skylens_kwargs['WT_kwargs']['s1_s2']:
        s1_s2_new+=[eval(i)]

    skylens_kwargs['WT_kwargs']['s1_s2']=s1_s2_new
        #     skylens_kwargs['WT']=wigner_transform(**skylens_kwargs['WT_kwargs'])
    return skylens_kwargs
    
def set_cosmology(skylens_kwargs={}):
    """
    Set the cosmological parameters, from astropy object, 
    depending on yaml input.
    """
    if skylens_kwargs['cosmo_params']['read_cosmo'] is not None:
        cosmo=getattr(astropy.cosmology,skylens_kwargs['cosmo_params']['read_cosmo'])
        skylens_kwargs['cosmo_params'].update({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'Ase9':2.2,'mnu':cosmo.m_nu[-1].value,'Omk':cosmo.Ok0,'tau':0.06,'ns':0.965,
                'OmR':cosmo.Ogamma0+cosmo.Onu0,'w':-1,'wa':0,'Tcmb':cosmo.Tcmb0})

    return skylens_kwargs

def set_cluster(skylens_kwargs={}):
    """
    Start a dask cluster, if instructed to do so in the yaml.
    """
    if skylens_kwargs['scheduler_info'] is not None:
            return skylens_kwargs
    if not skylens_kwargs['start_new_cluster']:
        return skylens_kwargs

    if skylens_kwargs['scheduler_args']['local_directory'] is None:
        skylens_kwargs['scheduler_args']['local_directory']='./temp_skylens/'
    skylens_kwargs['scheduler_args']['local_directory']+='/pid'+str(os.getpid())+'/'
    LC,scheduler_info=start_client(**skylens_kwargs['scheduler_args'])
    skylens_kwargs['scheduler_info']=scheduler_info
    return skylens_kwargs