import os,sys
sys.path.append('/verafs/scratch/phy200040p/sukhdeep/project/skylens/skylens/')
import dask
import numpy as np
import warnings,logging
import copy
import multiprocessing,psutil
import sparse
import gc
import dask.bag
from dask import delayed

from skylens.power_spectra import *
from skylens.angular_power_spectra import *
from skylens.hankel_transform import *
from skylens.wigner_transform import *
from skylens.binning import *
from skylens.cov_utils import *
from skylens.tracer_utils import *
from skylens.window_utils import *
from skylens.utils import *
from skylens.parse_input import *

d2r=np.pi/180.

class Skylens():
    def __init__(self,yaml_inp_file=None,python_inp_file=None,l=None,l_cl=None,Ang_PS=None,
                cov_utils=None,logger=None,tracer_utils=None,#lensing_utils=None,galaxy_utils=None,
                shear_zbins=None,kappa_zbins=None,galaxy_zbins=None,
                pk_params=None,cosmo_params=None,WT_kwargs=None,WT=None,
                z_PS=None,nz_PS=100,log_z_PS=2,
                do_cov=False,SSV_cov=False,tidal_SSV_cov=False,do_sample_variance=True,
                Tri_cov=False,sparse_cov=False,
                use_window=True,window_lmax=None,window_l=None,store_win=False,Win=None,
                f_sky=None,wigner_step=None,cl_func_names={},zkernel_func_names={},
                l_bins=None,l_bins_center=None,bin_cl=False,use_binned_l=False,do_pseudo_cl=True,
                stack_data=False,bin_xi=False,do_xi=False,theta_bins=None,theta_bins_center=None,
                use_binned_theta=False, xi_SN_analytical=False,
                corrs=None,corr_indxs=None,stack_indxs=None,
                wigner_files=None,name='',clean_tracer_window=True,
                scheduler_info=None):

        self.__dict__.update(locals()) #assign all input args to the class as properties
        if yaml_inp_file is not None:
            yaml_inp_args=parse_yaml(file_name=yaml_inp_file)
            self.__dict__.update(yaml_inp_args)
            #print('skylens init, yaml args',yaml_inp_args.keys())
            del yaml_inp_args
        elif python_inp_file is not None:
            yaml_inp_args=parse_python(file_name=python_inp_file)
            self.__dict__.update(yaml_inp_args)
            del yaml_inp_args
            
        self.l0=self.l*1.
        if self.l_cl is None:
            self.l_cl=self.l
        self.l_cl0=self.l_cl*1.
        
        self.set_WT()
        self.set_bin_params()
        self.set_binned_measure(locals())

        if logger is None:#not really being used right now
            self.logger=logging.getLogger() 
            self.logger.setLevel(level=logging.DEBUG)
            logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                                level=logging.DEBUG, datefmt='%I:%M:%S')

        if tracer_utils is None:
            self.tracer_utils=Tracer_utils(shear_zbins=self.shear_zbins,galaxy_zbins=self.galaxy_zbins,kappa_zbins=self.kappa_zbins,
                                            logger=self.logger,l=self.l_cl,scheduler_info=self.scheduler_info,
                                            zkernel_func_names=self.zkernel_func_names,do_cov=self.do_cov)

        self.set_corr_indxs(corr_indxs=self.corr_indxs,stack_indxs=self.stack_indxs)

        self.window_lmax=30 if window_lmax is None else window_lmax
        self.window_l=np.arange(self.window_lmax+1) if self.window_l is None else self.window_l

        self.set_WT_spins()

        self.z_bins=self.tracer_utils.z_bins
        
        if cov_utils is None:
            self.cov_utils=Covariance_utils(f_sky=self.f_sky,l=self.l_cl,logger=self.logger,
                                            do_xi=self.do_xi,xi_SN_analytical=self.xi_SN_analytical,
                                            do_sample_variance=self.do_sample_variance,WT=self.WT,
                                            use_window=self.use_window,use_binned_l=self.use_binned_l,
                                            window_l=self.window_l,use_binned_theta=self.use_binned_theta,
                                            do_cov=self.do_cov,SSV_cov=self.SSV_cov,tidal_SSV_cov=self.tidal_SSV_cov,
                                            Tri_cov=self.Tri_cov,sparse_cov=self.sparse_cov,bin_cl=self.bin_cl,
                                            do_pseudo_cl=self.do_pseudo_cl,
                                           cl_bin_utils=self.cl_bin_utils,xi_bin_utils=self.xi_bin_utils,)

        if Ang_PS is None:
            power_spectra_kwargs={'pk_params':self.pk_params,'cosmo_params':self.cosmo_params}
            self.Ang_PS=Angular_power_spectra(
                                SSV_cov=self.SSV_cov,l=self.l_cl,logger=self.logger,
                                power_spectra_kwargs=power_spectra_kwargs,
                                cov_utils=self.cov_utils,window_l=self.window_l,
                                z_PS=self.z_PS,nz_PS=self.nz_PS,log_z_PS=self.log_z_PS,
                                z_PS_max=self.tracer_utils.z_PS_max)
                        #FIXME: Need a dict for these args
        self.set_cl_funcs()
        
#         if self.do_xi and not self.xi_win_approx: #FIXME: Since the `aprrox' is actually the correct way, change the notation.
#             self.do_pseudo_cl=True #we will use pseudo_cl transform to get correlation functions.

        self.Win0=window_utils(window_l=self.window_l,l=self.l0,l_bins=self.l_bins,l_cl=self.l_cl0,
                               corrs=self.corrs,s1_s2s=self.s1_s2s,
                               cov_indxs=self.cov_indxs,scheduler_info=self.scheduler_info,
                               use_window=self.use_window,do_cov=self.do_cov,#cov_utils=self.cov_utils,f_sky=self.f_sky,
                               corr_indxs=self.stack_indxs,z_bins=self.tracer_utils.z_win,
                               window_lmax=self.window_lmax,Win=self.Win,WT=self.WT,do_xi=self.do_xi,
                               do_pseudo_cl=self.do_pseudo_cl,wigner_step=self.wigner_step,
                               kappa_class0=self.kappa0,kappa_class_b=self.kappa_b,kappa_b_xi=self.kappa_b_xi,
                               xi_bin_utils=self.xi_bin_utils,store_win=self.store_win,wigner_files=self.wigner_files,
                               bin_window=self.use_binned_l,bin_theta_window=self.use_binned_theta)
        self.Win0.get_Win()
        self.bin_window=self.Win0.bin_window
        win=self.Win0.Win
#         del self.Win0
        self.Win=win
        self.set_WT_binned()
        self.set_binned_measure(None,clean_up=True)
        
        if clean_tracer_window:
            self.tracer_utils.clean_z_window()
        
        print('Window done. Size:',get_size_pickle(self.Win))
        self.clean_setup()

    def clean_setup(self):
        """
            remove some objects from self that are not required after setup.
        """
        atrs=['kappa_zbins','galaxy_zbins','shear_zbins','WT_kwargs','pk_params','cosmo_params']
        for atr in atrs:
            if hasattr(self,atr):
                delattr(self,atr)

    def clean_non_essential(self):
        """
            remove some objects from that may not be required after some calculations.
        """
        atrs=['z_bins']
        for atr in atrs:
            if hasattr(self,atr):
                delattr(self,atr)
            if hasattr(self.tracer_utils,atr):
                delattr(self.tracer_utils,atr)
                
    def set_cl_funcs(self,):
        if self.cl_func_names is None:
            self.cl_func_names={} #we assume it is a dict below.
        self.cl_func={}
        for corr in self.corrs:
#            self.cl_func[corr]=self.calc_cl
            if self.cl_func_names.get(corr) is None:
                if self.cl_func_names.get(corr[::-1]) is None:
                    self.cl_func_names[corr]='calc_cl'
                    self.cl_func_names[corr[::-1]]='calc_cl'
                else:
                    self.cl_func_names[corr]=self.cl_func_names[corr[::-1]]
            if self.cl_func.get(corr) is None: 
#                 if hasattr(self,self.cl_func_names[corr]):
#                     self.cl_func[corr]=getattr(self,self.cl_func_names[corr])
#                 elif hasattr(self.Ang_PS.PS,self.cl_func_names[corr]):
#                     self.cl_func[corr]=getattr(self.Ang_PS.PS,self.cl_func_names[corr])
#                 else:
                    self.cl_func[corr]=globals()[self.cl_func_names[corr]]
            if not callable(self.cl_func[corr]):
                raise Exception(self.cl_func[corr],'is not a callable function')
            self.cl_func[corr[::-1]]=self.cl_func[corr]
            
    def set_binned_measure(self,local_args,clean_up=False):
        """
            If we only want to run computations at effective bin centers, then we 
            need to bin the windows and wigner matrices properly, for which unbinned
            quantities need to be computed once. This function sets up the unbinned
            computations, which are used later for binning window coupling and wigner
            matrices. 
            This is useful when running multiple computations for chains etc. For 
            covariance and one time calcs, may as well just do the full computation.
        """
        if clean_up:
            if self.use_binned_l or self.use_binned_theta:
                del self.kappa0,self.kappa_b
            return 
        if self.use_binned_l or self.use_binned_theta:
            inp_args={}
            for k in local_args.keys():
                if k=='self' or k=='client' or 'yaml' in k or 'python' in k:
                    continue
                inp_args[k]=copy.deepcopy(self.__dict__[k])#when passing yaml, most of input_args are updated. use updated ones
            #print('binned_meansure',inp_args.keys())
            if self.l_bins_center is None:
                self.l_bins_center=np.int32((self.l_bins[1:]+self.l_bins[:-1])*.5)
            inp_args['use_binned_l']=False
            inp_args['use_binned_theta']=False
            inp_args['use_window']=False
            inp_args['do_cov']=False 
            inp_args['bin_xi']=False
            inp_args['name']='S0'
            if self.do_cov:
                inp_args['corr_indxs']=None
                inp_args['stack_indxs']=None
#             del inp_args['self']
            inp_args2=copy.deepcopy(inp_args)
    
            self.kappa0=Skylens(**inp_args)  #to get unbinned c_ell and xi
#             self.kappa0.bin_xi=False #we want to get xi_bin_utils

            inp_args2['l']=self.l_bins_center
            inp_args2['l_cl']=self.l_bins_center
            inp_args2['name']='S_b'
            inp_args2['l_bins']=None
            inp_args2['bin_cl']=False
            inp_args2['do_xi']=False
            self.kappa_b=Skylens(**inp_args2) #to get binned c_ell
            self.kappa_b_xi=None
            if self.do_xi and self.use_binned_theta:
                theta_bins=inp_args['theta_bins']
                if self.theta_bins_center is None:
                    self.theta_bins_center=(theta_bins[1:]+theta_bins[:-1])*.5 #FIXME:this may not be effective theta of meaurements
                inp_args_xi=copy.deepcopy(inp_args)
                inp_args_xi['name']='S_b_xi'
                inp_args_xi['bin_xi']=True
                inp_args_xi['do_pseudo_cl']=False
                inp_args_xi['use_window']=self.use_window
#                 inp_args_xi['WT'].reset_theta_l(theta=self.theta_bins_center)#FIXME
                self.kappa_b_xi=Skylens(**inp_args_xi) #to get binned xi. 
                
                self.xi0=self.kappa0.xi_tomo()['xi']
                self.xi_b=self.kappa_b_xi.xi_tomo()['xi']
            self.l=self.l_bins_center*1.
            self.l_cl=self.l_bins_center*1.
            self.c_ell0=self.kappa0.cl_tomo()['cl']
            self.c_ell_b=self.kappa_b.cl_tomo()['cl']
            print('set binned measure done')
        else:
            self.kappa_b=self
            self.kappa0=self
            self.kappa_b_xi=None


    def set_corr_indxs(self,corr_indxs=None,stack_indxs=None):
        """
        set up the indexes for correlations. indexes= tracer and bin ids. 
        User can input the corr_indxs which will be the ones computed (called stack_indxs later). 
        However, when doing covariances, we may need to compute the 
        aiddtional correlations, hence those are included added to the corr_indxs.
        corr_indxs are used for constructing full compute graph but only the stack_indxs 
        is actually computed when stack_dat is called. 
        """
        self.stack_indxs=stack_indxs
        self.corr_indxs=corr_indxs
        self.cov_indxs=None
        if self.corrs is None:
            if self.stack_indxs is not None:
                self.corrs=list(corr_indxs.keys())
            else:
                nt=len(self.tracer_utils.tracers)
                self.corrs=[(self.tracer_utils.tracers[i],self.tracer_utils.tracers[j])
                            for i in np.arange(nt)
                            for j in np.arange(i,nt)
                            ]
                
        if not self.do_cov and self.corr_indxs is None:
            self.corr_indxs=self.stack_indxs #if no covariance, we only work with user input  stack_indxs
                                            #for covariance, we may need additional correlation pairs, 
                                            # which are added below.
            
        if not self.do_cov and  (not self.corr_indxs is None):
            print('not setting corr_indxs',self.do_cov , bool(self.corr_indxs))
            return
        
        else:
            self.corr_indxs={}
            
        for tracer in self.tracer_utils.tracers:
            self.corr_indxs[(tracer,tracer)]=[j for j in itertools.combinations_with_replacement(
                                                    np.arange(self.tracer_utils.n_bins[tracer]),2)]

#             if tracer=='galaxy' and not self.do_cov:
#                 self.corr_indxs[(tracer,tracer)]=[(i,i) for i in np.arange(self.tracer_utils.n_bins[tracer])] 
                #by default, assume no cross correlations between galaxy bins

        for tracer1 in self.tracer_utils.tracers:#zbin-indexs for cross correlations
            for tracer2 in self.tracer_utils.tracers:
                if tracer1==tracer2:#already set above
                    continue
                if self.corr_indxs.get((tracer1,tracer2)) is not None:
                    continue
                self.corr_indxs[(tracer1,tracer2)]=[ k for l in [[(i,j) for i in np.arange(
                                        self.tracer_utils.n_bins[tracer1])] 
                                        for j in np.arange(self.tracer_utils.n_bins[tracer2])] for k in l]
        
        if self.stack_indxs is None:# or not bool(self.stack_indxs):
            self.stack_indxs=self.corr_indxs
        self.cov_indxs={}
        if self.do_cov:
            stack_corr_indxs=self.stack_indxs
            corrs=self.corrs
            corrs_iter=[(corrs[i],corrs[j]) for i in np.arange(len(corrs)) for j in np.arange(i,len(corrs))]
            for (corr1,corr2) in corrs_iter:
                corr1_indxs=stack_corr_indxs[(corr1[0],corr1[1])]
                corr2_indxs=stack_corr_indxs[(corr2[0],corr2[1])]
                if corr1==corr2:
                    cov_indxs_iter=[ k for l in [[corr1_indxs[i]+corr2_indxs[j] for j in np.arange(i,
                                     len(corr1_indxs))] for i in np.arange(len(corr2_indxs))] for k in l]
                else:
                    cov_indxs_iter=[ k for l in [[corr1_indxs[i]+corr2_indxs[j] for i in np.arange(
                                    len(corr1_indxs))] for j in np.arange(len(corr2_indxs))] for k in l]
                self.cov_indxs[corr1+corr2]=cov_indxs_iter

    def set_WT_spins(self):
        """
        set the spin factors for tracer pairs, used for wigner transforms.
        """
        self.s1_s2s={}
        for tracer1 in self.tracer_utils.tracers:#zbin-indexs for cross correlations
            for tracer2 in self.tracer_utils.tracers:
                self.s1_s2s[(tracer1,tracer2)]=[(self.tracer_utils.spin[tracer1],self.tracer_utils.spin[tracer2])]
        if 'shear' in self.tracer_utils.tracers:
            self.s1_s2s[('shear','shear')]=[(2,2),(2,-2)]
        self.s1_s2s[('window')]=[(0,0)]
    
    def set_WT(self):
        """
        Setup wigner transform based on input args, if not transform
        class is not passed directly.
        """
        if self.WT is not None or self.WT_kwargs is None or not self.do_xi:
            return
        self.WT=wigner_transform(**self.WT_kwargs)
        
    def set_WT_binned(self):
        """
        If we only want to compute at bin centers, wigner transform matrices need to be binned.
        """
        if self.WT is None:
            return 
        client=client_get(self.scheduler_info)
#         self.WT.scatter_data(scheduler_info=self.scheduler_info)
        WT=self.WT
        self.WT_binned={corr:{} for corr in self.corrs} #intialized later.
        self.inv_WT_binned={corr:{} for corr in self.corrs} #intialized later.
        self.WT_binned_cov={corr:{} for corr in self.corrs}
        client=client_get(self.scheduler_info)
        if self.use_binned_theta or self.use_binned_l:
            WT.set_binned_theta(theta_bins=self.theta_bins)
            WT.set_binned_l(l_bins=self.l_bins)
        if self.do_xi and (self.use_binned_l or self.use_binned_theta):
            for corr in self.corrs:
                s1_s2s=self.s1_s2s[corr]
                self.WT_binned[corr]={s1_s2s[im]:{} for im in np.arange(len(s1_s2s))}
                self.inv_WT_binned[corr]={s1_s2s[im]:{} for im in np.arange(len(s1_s2s))}
                self.WT_binned_cov[corr]={s1_s2s[im]:{} for im in np.arange(len(s1_s2s))}
                for indxs in self.corr_indxs[corr]:    
                    cl0=self.c_ell0[corr][indxs].compute()
                    cl_b=self.c_ell_b[corr][indxs].compute()
                    wt0=cl0
                    wt_b=1./cl_b
                    if np.all(cl_b==0):
                        wt_b[:]=0
                    for im in np.arange(len(s1_s2s)):
                        s1_s2=s1_s2s[im]
                        win_xi=None
                        if self.use_window:# and self.xi_win_approx:
                            win_xi=client.gather(self.Win['cl'][corr][indxs])#['xi']#this will not work for covariance
                            win_xi=win_xi['xi']
                        self.WT_binned[corr][s1_s2][indxs]=delayed(self.binning.bin_2d_WT)(
                                                            wig_mat=self.WT.wig_d[s1_s2],
                                                            wig_norm=self.WT.wig_norm,
                                                            wt0=wt0,wt_b=wt_b,bin_utils_cl=self.cl_bin_utils,
                                                            bin_utils_xi=self.xi_bin_utils[s1_s2],
                                                            win_xi=win_xi,use_binned_theta=self.use_binned_theta)
                        self.WT_binned[corr][s1_s2][indxs]=client.compute(self.WT_binned[corr][s1_s2][indxs]).result()
                        self.WT_binned[corr][s1_s2][indxs]=client.scatter(self.WT_binned[corr][s1_s2][indxs])
                        
                        if self.do_xi and self.use_binned_theta:
                            xi0=self.xi0[corr][s1_s2][indxs].compute()
                            xi_b=self.xi_b[corr][s1_s2][indxs].compute()
                            wt0_inv=xi0
                            wt_b_inv=1./xi_b
                            if np.all(xi_b==0):
                                wt_b_inv[:]=0
                            self.inv_WT_binned[corr][s1_s2][indxs]=delayed(self.binning.bin_2d_inv_WT)(
                                                            wig_mat=self.WT.wig_d[s1_s2],
                                                            wig_norm=self.WT.inv_wig_norm,
                                                            wt0=wt0_inv,wt_b=wt_b_inv,bin_utils_cl=self.cl_bin_utils,
                                                            bin_utils_xi=self.xi_bin_utils[s1_s2],
                                                            win_xi=None,use_binned_l=self.use_binned_l)
                            self.inv_WT_binned[corr][s1_s2][indxs]=client.compute(self.inv_WT_binned[corr][s1_s2][indxs]).result()
                            self.inv_WT_binned[corr][s1_s2][indxs]=client.scatter(self.inv_WT_binned[corr][s1_s2][indxs])
                        
                        if self.do_cov:
                            if win_xi is None:
                                self.WT_binned_cov[corr][s1_s2][indxs]=self.WT_binned[corr][s1_s2][indxs]
                            else:
                                self.WT_binned_cov[corr][s1_s2][indxs]=delayed(self.binning.bin_2d_WT)(
                                                            wig_mat=self.WT.wig_d[s1_s2],
                                                            wig_norm=self.WT.wig_norm,
                                                            wt0=wt0,wt_b=wt_b,bin_utils_cl=self.cl_bin_utils,
                                                            bin_utils_xi=self.xi_bin_utils[s1_s2],
                                                            win_xi=None,
                                                            use_binned_theta=self.use_binned_theta)
                        
    def update_zbins(self,z_bins={},tracer='shear'):
        """
        If the tracer bins need to be updated. Ex. when running chains with varying photo-z params.
        """
        self.tracer_utils.set_zbins(z_bins,tracer=tracer)
        self.z_bins=self.tracer_utils.z_bins
        return


    def set_bin_params(self):
        """
            Setting up the binning functions to be used in binning the data
        """
        self.binning=binning()
        self.cl_bin_utils=None
        client=client_get(self.scheduler_info)
        if self.bin_cl or self.use_binned_l:
            self.cl_bin_utils=self.binning.bin_utils(r=self.l0,r_bins=self.l_bins,
                                                r_dim=2,mat_dims=[1,2])
#             self.cl_bin_utils={k:client.scatter(self.cl_bin_utils[k]) for k in self.cl_bin_utils.keys()}
            self.cl_bin_utils=scatter_dict(self.cl_bin_utils,scheduler_info=self.scheduler_info,broadcast=True)
        self.xi_bin_utils=None
        if self.do_xi and self.bin_xi:
            self.xi_bin_utils={}
            for s1_s2 in self.WT.s1_s2s:
                self.xi_bin_utils[s1_s2]=delayed(self.binning.bin_utils)(r=self.WT.theta_deg[s1_s2],
                                                    r_bins=self.theta_bins,
                                                    r_dim=2,mat_dims=[1,2])
                self.xi_bin_utils[s1_s2]=client.compute(self.xi_bin_utils[s1_s2]).result()
                self.xi_bin_utils[s1_s2]=scatter_dict(self.xi_bin_utils[s1_s2],scheduler_info=self.scheduler_info,broadcast=True)
            
    def calc_cl(self,zbin1={}, zbin2={},corr=('shear','shear'),cosmo_params=None,Ang_PS=None):#FIXME: this can be moved outside the class.thenwe don't need to serialize self.
        """
            Compute the angular power spectra, Cl between two source bins
            zs1, zs2: Source bins. Dicts containing information about the source bins
        """
        clz=Ang_PS.clz
        cls=clz['cls']
        f=Ang_PS.cl_f
        sc=zbin1['kernel_int']*zbin1['kernel_int']
        dchi=clz['dchi']
        cl=np.dot(cls.T*sc,dchi)
                # cl*=2./np.pi #FIXME: needed to match camb... but not CCL
        return cl

    def bin_cl_func(self,cl=None,cov=None):#moved out of class. This is no longer used
        """
            bins the tomographic power spectra
            results: Either cl or covariance
            bin_cl: if true, then results has cl to be binned
            bin_cov: if true, then results has cov to be binned
            Both bin_cl and bin_cov can be true simulatenously.
        """
        cl_b=None
        if not cl is None:
            if self.use_binned_l or not self.bin_cl:
                cl_b=cl*1.
            else:
                cl_b=self.binning.bin_1d(xi=cl,bin_utils=self.cl_bin_utils)
            return cl_b

    def calc_pseudo_cl(self,cl,Win):# moved outside the class. No used now.
        pcl=cl@Win['M']
        return  pcl
        
    def cl_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,
                corrs=None,bias_kwargs={},bias_func=None,stack_corr_indxs=None,
                z_bins=None,Ang_PS=None):
        """
         Computes full tomographic power spectra and covariance, including shape noise. output is
         binned also if needed.
         Arguments are for the power spectra  and sigma_crit computation,
         if it needs to be called from here.
         source bins are already set. This function does set the sigma crit for sources.
        """

        l=self.l
        if corrs is None:
            corrs=self.corrs
        if stack_corr_indxs is None:
            stack_corr_indxs=self.stack_indxs
        if Ang_PS is None:
            Ang_PS=self.Ang_PS
        if z_bins is None:
            z_bins=self.z_bins
#         if z_params is None:
#             z_params=z_bins
        client=client_get(self.scheduler_info)
        tracers=np.unique([j for i in corrs for j in i])
        
        corrs2=corrs.copy()
        if self.do_cov:
            for i in np.arange(len(tracers)):
                for j in np.arange(i,len(tracers)):
                    if (tracers[i],tracers[j]) not in corrs2 and (tracers[j],tracers[i]) in corrs2:
                        corrs2+=[(tracers[i],tracers[j])]
                        print('added extra corr calc for covariance',corrs2)

        Ang_PS.angular_power_z(cosmo_h=cosmo_h,pk_params=pk_params,
                                    cosmo_params=cosmo_params)
#         clz=Ang_PS.clz
            
        if cosmo_h is None:
            cosmo_h=Ang_PS.PS#.cosmo_h

        zkernel={}
        self.SN={}
        AP=client.scatter(Ang_PS,broadcast=True)
        for tracer in tracers:
            zkernel[tracer]=self.tracer_utils.set_kernels(Ang_PS=AP,tracer=tracer,z_bins=z_bins[tracer],delayed_compute=True)
            self.SN[(tracer,tracer)]=self.tracer_utils.SN[tracer]
            if 'galaxy' in tracers:
                if bias_func is None:
                    bias_func='constant_bias'
                    bias_kwargs={'b1':1,'b2':1}
#         clz=scatter_dict(clz,scheduler_info=self.scheduler_info)
        
        cosmo_params=scatter_dict(cosmo_params,scheduler_info=self.scheduler_info,broadcast=True)
        self.SN=scatter_dict(self.SN,scheduler_info=self.scheduler_info,broadcast=True)

        out={}
        cl={corr:{} for corr in corrs2}
        cl.update({corr[::-1]:{} for corr in corrs2})
        pcl={corr:{} for corr in corrs2}
        pcl.update({corr[::-1]:{} for corr in corrs2}) #pseudo_cl
        cl_b={corr:{} for corr in corrs2}
        cl_b.update({corr[::-1]:{} for corr in corrs2})
        pcl_b={corr:{} for corr in corrs2}
        pcl_b.update({corr[::-1]:{} for corr in corrs2})
        cov={}
        for corr in corrs2:
            corr2=corr[::-1]
            corr_indxs=self.corr_indxs[(corr[0],corr[1])]#+self.cov_indxs
            for (i,j) in corr_indxs:#FIXME: we might want to move to map, like covariance. will be useful to define the tuples in forzenset then.
                cl[corr][(i,j)]=delayed(self.cl_func[corr])(zbin1=zkernel[corr[0]][i],zbin2=zkernel[corr[1]][j],
                                                             corr=corr,cosmo_params=cosmo_params,Ang_PS=AP) 
                cl_b[corr][(i,j)]=delayed(bin_cl_func)(cl=cl[corr][(i,j)],use_binned_l=self.use_binned_l,bin_cl=self.bin_cl,cl_bin_utils=self.cl_bin_utils)
                if self.use_window and self.do_pseudo_cl and (i,j) in self.stack_indxs[corr]:
                    if not self.bin_window:
                        pcl[corr][(i,j)]=delayed(calc_pseudo_cl)(cl[corr][(i,j)],Win=self.Win['cl'][corr][(i,j)])
                        pcl_b[corr][(i,j)]=delayed(bin_cl_func)(cl=pcl[corr][(i,j)],use_binned_l=self.use_binned_l,bin_cl=self.bin_cl,cl_bin_utils=self.cl_bin_utils)
                    else:
                        pcl[corr][(i,j)]=None
                        pcl_b[corr][(i,j)]=delayed(calc_cl_pseudo_cl)(zbin1=zkernel[corr[0]][i],zbin2=zkernel[corr[1]][j],
                                                             corr=corr,cosmo_params=cosmo_params,Ang_PS=AP,Win=self.Win['cl'][corr][(i,j)])
#                         pcl_b[corr][(i,j)]=delayed(calc_pseudo_cl)(None,cl_b[corr][(i,j)],Win=self.Win.Win['cl'][corr][(i,j)])
                else:
                    pcl[corr][(i,j)]=cl[corr][(i,j)]
                    pcl_b[corr][(i,j)]=cl_b[corr][(i,j)]
                cl[corr2][(j,i)]=cl[corr][(i,j)]#useful in gaussian covariance calculation.
                pcl[corr2][(j,i)]=pcl[corr][(i,j)]#useful in gaussian covariance calculation.
                cl_b[corr2][(j,i)]=cl_b[corr][(i,j)]#useful in gaussian covariance calculation.
                pcl_b[corr2][(j,i)]=pcl_b[corr][(i,j)]#useful in gaussian covariance calculation.
    
        print('cl graph done')
        cosmo_params=gather_dict(cosmo_params,scheduler_info=self.scheduler_info)
        if self.do_cov:# and (self.do_pseudo_cl or not self.do_xi):
            Win_cov=None
            Win_cl=None
            corrs_iter=[(corrs[i],corrs[j]) for i in np.arange(len(corrs)) for j in np.arange(i,len(corrs))]
            cov_indxs={}
            CU=client.scatter(self.cov_utils,broadcast=True)
            for (corr1,corr2) in corrs_iter:
                cov[corr1+corr2]={}
                cov[corr2+corr1]={}

                corr1_indxs=stack_corr_indxs[(corr1[0],corr1[1])]
                corr2_indxs=stack_corr_indxs[(corr2[0],corr2[1])]
                if corr1==corr2:
                    cov_indxs_iter=[ k for l in [[corr1_indxs[i]+corr2_indxs[j] for j in np.arange(i,
                                     len(corr1_indxs))] for i in np.arange(len(corr2_indxs))] for k in l]
                else:
                    cov_indxs_iter=[ k for l in [[corr1_indxs[i]+corr2_indxs[j] for i in np.arange(
                                    len(corr1_indxs))] for j in np.arange(len(corr2_indxs))] for k in l]
                cov_indxs[corr1+corr2]=cov_indxs_iter #because in principle we allow stack_indxs to be different than self.stack_indxs
                
                if self.use_window:# and not self.store_win:
                    Win_cov=self.Win['cov'][corr1+corr2] # we only want to pass this if it is a graph. Otherwise, read within function
                    Win_cl1=self.Win['cl'][corr1]
                    Win_cl2=self.Win['cl'][corr2]
                #cov[corr1+corr2]=dask.bag.from_sequence(cov_indxs_iter).map(self.cl_cov,cls=cl,Win_cov=Win_cov,tracers=corr1+corr2,Win_cl=Win_cl)
                for indxs in cov_indxs_iter:
                    if self.use_window:
                        Win_cl1i=Win_cl1[(indxs[0],indxs[1])]
                        Win_cl2i=Win_cl2[(indxs[2],indxs[3])]
                        Win_covi=Win_cov[indxs]
                    else:
                        Win_cl1i,Win_cl2i,Win_covi=None,None,None
                    sig_cL=delayed(self.cov_utils.cov_four_kernels)(z_bins={0:zkernel[corr1[0]][indxs[0]],
                                                                            1:zkernel[corr1[1]][indxs[1]],
                                                                            2:zkernel[corr2[0]][indxs[2]],
                                                                            3:zkernel[corr2[1]][indxs[3]]},
                                                                             Ang_PS=AP)#don't want to copy z_bins,
                    cov[corr1+corr2][indxs]=delayed(cl_cov)(indxs,CU,cls=self.get_CV_cl(cl,corr1+corr2,indxs),
                                                                        SN=self.SN,cl_bin_utils=self.cl_bin_utils,
                                                                        Win_cov=Win_covi,tracers=corr1+corr2,
                                                                          Win_cl1=Win_cl1i,Ang_PS=AP,
                                                                          Win_cl2=Win_cl2i,sig_cL=sig_cL)
            cov['cov_indxs']=cov_indxs

            print('cl cov graph done')

        out_stack=delayed(self.stack_dat)({'cov':cov,'pcl_b':pcl_b,'est':'pcl_b'},corrs=corrs,
                                          corr_indxs=stack_corr_indxs)
#         if not self.do_xi:
#             return {'stack':out_stack,'cl_b':cl_b,'cov':cov,'cl':cl,'pseudo_cl':pcl,'pseudo_cl_b':pcl_b,'zkernel':zkernel,'clz':clz}
#         else:
        return {'stack':out_stack,'cl_b':cl_b,'cov':cov,'cl':cl,'pseudo_cl':pcl,'pseudo_cl_b':pcl_b,'zkernel':zkernel}#,'clz':clz}

    def gather_data(self):
        client=client_get(self.scheduler_info)
        keys=['xi_bin_utils','cl_bin_utils','Win','WT_binned','z_bins','SN']
        for k in keys:
            if hasattr(self,k):
                self.__dict__[k]=gather_dict(self.__dict__[k],scheduler_info=self.scheduler_info)
        self.Ang_PS.clz=client.gather(self.Ang_PS.clz)
        self.tracer_utils.gather_z_bins()
        if self.WT is not None:
            self.WT.gather_data()
        
    def scatter_data(self):
        client=client_get(self.scheduler_info)
        keys=['xi_bin_utils','cl_bin_utils','Win','WT_binned']
        for k in keys:
            if hasattr(self,k):
                self.__dict__[k]=scatter_dict(self.__dict__[k],scheduler_info=self.scheduler_info,depth=1)
        self.WT.scatter_data()
        
        
    def tomo_short(self,cosmo_h=None,cosmo_params=None,pk_lock=None,WT_binned=None,WT=None,
                corrs=None,bias_kwargs={},bias_func=None,stack_corr_indxs=None,
                z_bins=None,Ang_PS=None,zkernel=None,Win=None,cl_bin_utils=None,
                  xi_bin_utils=None,pk_params=None):
        """
            
        """
        if corrs is None:
            corrs=self.corrs
        if stack_corr_indxs is None:
            stack_corr_indxs=self.stack_indxs

        tracers=np.unique([j for i in corrs for j in i])
        
        Ang_PS.angular_power_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_lock=pk_lock,
                                    cosmo_params=cosmo_params)
        if cosmo_h is None:
            cosmo_h=Ang_PS.PS#.cosmo_h

        if zkernel is None:
            zkernel={}
            for tracer in tracers:
                zkernel[tracer]=self.tracer_utils.set_kernels(Ang_PS=Ang_PS,tracer=tracer,z_bins=z_bins[tracer],delayed_compute=False)
                if 'galaxy' in tracers:
                    if bias_func is None:
                        bias_func='constant_bias'
                        bias_kwargs={'b1':1,'b2':1}
        if self.do_xi:
            return self.xi_tomo_short(corrs=corrs,stack_corr_indxs=stack_corr_indxs,zkernel=zkernel,
                                      Ang_PS=Ang_PS,Win=Win,WT_binned=WT_binned,WT=WT,xi_bin_utils=xi_bin_utils)
        else:
            return self.cl_tomo_short(corrs=corrs,stack_corr_indxs=stack_corr_indxs,
                                      zkernel=zkernel,Ang_PS=Ang_PS,Win=Win,cl_bin_utils=cl_bin_utils)
    
    def cl_tomo_short(self,corrs=None,stack_corr_indxs=None,Ang_PS=None,zkernel=None,cosmo_params=None,Win=None,cl_bin_utils=None):
        """
         Same as cl_tomo, except no delayed is used and it only returns a stacked vector of binned pseudo-cl.
         This function is useful for mcmc where we only need to compute pseudo-cl, and want to reduce the
         dask overheard. You should run a parallel mcmc, where each call to this function is placed inside 
         delayed.
        """
        if Win is None:
            Win=self.Win
        l=self.l
        out={}
        pcl_b=[] 
        for corr in corrs:
            corr_indxs=stack_corr_indxs[corr]#+self.cov_indxs
            for (i,j) in corr_indxs:#FIXME: we might want to move to map, like covariance. will be useful to define the tuples in forzenset then.
                if self.use_window and self.do_pseudo_cl:
                    if not self.bin_window:
                        cl=self.cl_func[corr](zbin1=zkernel[corr[0]][i],zbin2=zkernel[corr[1]][j],
                                                             corr=corr,cosmo_params=cosmo_params,clz=Ang_PS.clz)#Ang_PS=Ang_PS) 

                        pcl=calc_pseudo_cl(cl,Win=Win['cl'][corr][(i,j)])
                        pcl_b+=[bin_cl_func(cl=pcl,use_binned_l=self.use_binned_l,bin_cl=self.bin_cl,cl_bin_utils=cl_bin_utils)]
                    else:
                        pcl=None
#                         try:
#                             tt=Win['cl'][corr][(i,j)]['M']
#                         except:
#                             print('cl tomo short: ',corr,(i,j),Win['cl'][corr][(i,j)])
#                             client=client_get(scheduler_info=self.scheduler_info)
#                             print('cl tomo short2 : ',gather_dict(Win['cl'][corr][(i,j)],scheduler_info=self.scheduler_info))
#                             print('cl tomo short3 : ',client.gather(Win['cl'][corr][(i,j)]))
                        pcl_b+=[calc_cl_pseudo_cl(zbin1=zkernel[corr[0]][i],zbin2=zkernel[corr[1]][j],
                                                             corr=corr,cosmo_params=cosmo_params,Ang_PS=Ang_PS,Win=Win['cl'][corr][(i,j)])]
                else:
                    pcl=self.cl_func[corr](zbin1=zkernel[corr[0]][i],zbin2=zkernel[corr[1]][j],
                                                             corr=corr,cosmo_params=cosmo_params,Ang_PS=Ang_PS)
                    pcl_b+=[bin_cl_func(cl=pcl,use_binned_l=self.use_binned_l,bin_cl=self.bin_cl,cl_bin_utils=self.cl_bin_utils)]
        pcl_b=np.concatenate(pcl_b).ravel()
        return pcl_b

    def get_xi(self,cls={},s1_s2=[],corr=None,indxs=None,Win=None):
        cl=cls[corr][indxs] #this should be pseudo-cl when using window
        wig_m=None
        if self.use_binned_l or self.use_binned_theta:
            wig_m=self.WT_binned[corr][s1_s2][indxs]
        th,xi=self.WT.projected_correlation(l_cl=self.l,s1_s2=s1_s2,cl=cl,wig_d=wig_m)
        xi_b=xi
        
        if self.bin_xi and not self.use_binned_theta: #wig_d is binned when use_binned_l
            if self.use_window:
                xi=xi*Win['xi']
            xi_b=self.binning.bin_1d(xi=xi,bin_utils=self.xi_bin_utils[s1_s2])
        
        if self.use_window:# and self.xi_win_approx:
            xi_b/=(Win['xi_b'])
        return xi_b
    
    def get_CV_cl(self,cls,tracers,z_indx):
        """
        Get the tracer power spectra, C_ell, for covariance calculations.
        """
        CV2={}
        CV2[13]=cls[(tracers[0],tracers[2])] [(z_indx[0], z_indx[2]) ]
        CV2[24]=cls[(tracers[1],tracers[3])][(z_indx[1], z_indx[3]) ]
        CV2[14]=cls[(tracers[0],tracers[3])][(z_indx[0], z_indx[3]) ]
        CV2[23]=cls[(tracers[1],tracers[2])][(z_indx[1], z_indx[2]) ]
        return CV2

    def xi_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None,
                corrs=None):
        """
            Computed tomographic angular correlation functions. First calls the tomographic
            power spectra and covariance and then does the hankel transform and  binning.
        """
        """
            For hankel transform is done on l-theta grid, which is based on s1_s2. So grid is
            different for xi+ and xi-.
            In the init function, we combined the ell arrays for all s1_s2. This is not a problem
            except for the case of SSV, where we will use l_cut to only select the relevant values
        """

        if cosmo_h is None:
            cosmo_h=self.Ang_PS.PS#.cosmo_h
        if corrs is None:
            corrs=self.corrs

        #Donot use delayed here. Leads to error/repeated calculations
        cls_tomo_nu=self.cl_tomo(cosmo_h=cosmo_h,cosmo_params=cosmo_params,
                            pk_params=pk_params,corrs=corrs)

        cl=cls_tomo_nu['cl'] #Note that if window is turned off, pseudo_cl=cl
#         clz=cls_tomo_nu['clz']
        cov_xi={}
        xi={}
        out={}
        zkernel=None
        client=client_get(self.scheduler_info)
        AP=client.scatter(self.Ang_PS,broadcast=True)
        if self.use_binned_theta:
            wig_norm=1
            wig_l=self.WT.l_bins_center
            wig_grad_l=self.WT.grad_l_bins
            wig_theta=self.WT.theta_bins_center
            wig_grad_theta=self.WT.grad_theta_bins
        else:
            wig_norm=self.WT.wig_norm
            wig_l=self.WT.l
            wig_grad_l=self.WT.grad_l
            wig_theta=self.WT.theta[(0,0)]
            wig_grad_theta=1

        for corr in corrs:
            s1_s2s=self.s1_s2s[corr]
            xi[corr]={}
            xi[corr[::-1]]={}
            for im in np.arange(len(s1_s2s)):
                s1_s2=s1_s2s[im]
                xi[corr][s1_s2]={}
                xi[corr[::-1]][s1_s2]={}
                xi_bin_utils=None
                if not self.use_binned_theta:
                    wig_d=self.WT.wig_d[s1_s2]
                if self.bin_xi:
                    xi_bin_utils=self.xi_bin_utils[s1_s2]
                for indx in self.corr_indxs[corr]:
                    if self.Win is None:
                        win=None
                    else:
                        win=self.Win['cl'][corr][indx]
                    if self.use_binned_theta:
                        wig_d=self.WT_binned[corr][s1_s2][indx]
                        
#                     xi[corr][s1_s2][indx]=delayed(self.get_xi)(cls=cl,corr=corr,indxs=indx,
#                                                         s1_s2=s1_s2,Win=win)
                    xi[corr][s1_s2][indx]=delayed(get_xi)(cl=cl[corr][indx],wig_d=wig_d,wig_norm=wig_norm,
                                         xi_bin_utils=xi_bin_utils,bin_xi=self.bin_xi,use_binned_theta=self.use_binned_theta,Win=win)
                    xi[corr[::-1]][s1_s2][indx[::-1]]=xi[corr][s1_s2][indx] #FIXME: s1_s2 should be reversed as well?... 

        print('Done xi graph',get_size(cl)/1.e6,get_size_pickle(self.cov_utils))
#         dic=self.cov_utils.__dict__
#         for k in dic.keys():
#             print('Done xi graph,cov_utils size',k,get_size_pickle(getattr(self.cov_utils,k)))
        if self.do_cov:
            corrs_iter=[(corrs[i],corrs[j]) for i in np.arange(len(corrs)) for j in np.arange(i,len(corrs))]
            cov_indxs={}
            zkernel=cls_tomo_nu['zkernel']
#             clz=cls_tomo_nu['clz']
            for (corr1,corr2) in corrs_iter:
                s1_s2s_1=self.s1_s2s[corr1]
                s1_s2s_2=self.s1_s2s[corr2]

                corr=corr1+corr2
                cov_xi[corr]={}
                cov_xi['cov_indxs']=cls_tomo_nu['cov']['cov_indxs']

                cov_cl=cls_tomo_nu['cov'][corr]#.compute()
                cov_iter=cls_tomo_nu['cov']['cov_indxs'][corr]
                CU=client.scatter(self.cov_utils,broadcast=True)
                Win_cov=None
                Win_cl=None
                if self.use_window:
#                     if not self.store_win:
                    Win_cov=self.Win['cov'][corr]
                    Win_cl1=self.Win['cl'][corr1]
                    Win_cl2=self.Win['cl'][corr2]
                for im1 in np.arange(len(s1_s2s_1)):
                    s1_s2=s1_s2s_1[im1]
                    start2=0
                    if corr1==corr2:
                        start2=im1
                    for im2 in np.arange(start2,len(s1_s2s_2)):
                        s1_s2_cross=s1_s2s_2[im2]
                        if not self.use_binned_theta:
                            wig_d1=self.WT.wig_d[s1_s2]
                            wig_d2=wig_d1

                        #cov_xi[corr][s1_s2+s1_s2_cross]=dask.bag.from_sequence(cov_cl).map(self.xi_cov,
                        cov_xi[corr][s1_s2+s1_s2_cross]={}
                        for indxs in cov_iter:
                            cls=self.get_CV_cl(cl,corr1+corr2,indxs)
                            if self.use_binned_l:
                                wig_d1=self.WT_binned_cov[corr1][s1_s2][(indxs[0],indxs[1])]
                                wig_d2=self.WT_binned_cov[corr2][s1_s2_cross][(indxs[2],indxs[3])]
                            WT_kwargs={'wig_d1':wig_d1,'wig_d2':wig_d2,'theta':self.WT.theta[s1_s2],
                                      'wig_l':wig_l,'wig_norm':wig_norm,'grad_l':wig_grad_l,
                                      'l_cl':self.l,'s1_s2':s1_s2,'s1_s2_cross':s1_s2_cross,
                                       'wig_theta':wig_theta,'wig_grad_theta':wig_grad_theta,
                                      }
                            if self.use_window:
                                Win_cl1i=Win_cl1[(indxs[0],indxs[1])]
                                Win_cl2i=Win_cl2[(indxs[0],indxs[1])]
                                Win_covi=Win_cov[indxs]
                            else:
                                Win_cl1i,Win_cl2i,Win_covi=None,None,None

                            sig_cL=delayed(self.cov_utils.cov_four_kernels)(z_bins={0:zkernel[corr1[0]][indxs[0]],
                                                                                    1:zkernel[corr1[1]][indxs[1]],
                                                                                    2:zkernel[corr2[0]][indxs[2]],
                                                                                    3:zkernel[corr2[1]][indxs[3]]},
                                                                                     Ang_PS=AP)
                            cov_xi[corr][s1_s2+s1_s2_cross][indxs]=delayed(xi_cov)(indxs,CU,cov_cl=None, #cov_cl[indxs],
                                                                                        cls=cls,s1_s2=s1_s2,SN=self.SN,
                                                                                        s1_s2_cross=s1_s2_cross,#clr=clr,
                                                                                        Win_cov=Win_covi,
                                                                                        xi_bin_utils=self.xi_bin_utils[s1_s2],
                                                                                        Win_cl1=Win_cl1i,Win_cl2=Win_cl2i,
                                                                                        corr1=corr1,corr2=corr2,sig_cL=sig_cL,
                                                                                        WT_kwargs=WT_kwargs,Ang_PS=AP
                                                                                                 )

        out['stack']=delayed(self.stack_dat)({'cov':cov_xi,'xi':xi,'est':'xi'},corrs=corrs)
        out['xi']=xi
        out['cov']=cov_xi
        out['cl']=cls_tomo_nu
        out['zkernel']=zkernel
        return out

    def xi_tomo_short(self,corrs=None,stack_corr_indxs=None,Ang_PS=None,zkernel=None,cosmo_params=None,Win=None,WT_binned=None,WT=None,xi_bin_utils=None):
        """
         Same as xi_tomo / cl_tomo_short, except no delayed is used and it only returns a stacked vector of binned xi.
         This function is useful for mcmc where we only need to compute xi, and want to reduce the
         dask overheard. You should run a parallel mcmc, where each call to this function is placed inside 
         delayed.
        """
        if Win is None:
            Win=self.Win
        if Ang_PS is None:
            Ang_PS=self.Ang_PS
        l=self.l
        cl={corr:{}for corr in corrs}
        out={}
        for corr in corrs:
            corr_indxs=stack_corr_indxs[corr]#+self.cov_indxs
            for (i,j) in corr_indxs:#FIXME: we might want to move to map, like covariance. will be useful to define the tuples in forzenset then.
                cl[corr][(i,j)]=self.cl_func[corr](zbin1=zkernel[corr[0]][i],zbin2=zkernel[corr[1]][j],
                                                             corr=corr,cosmo_params=cosmo_params,Ang_PS=Ang_PS) 

        xi_b=[]
        if self.use_binned_theta:
            wig_norm=1
            wig_l=WT.l_bins_center
            wig_grad_l=WT.grad_l_bins
        else:
            wig_norm=WT.norm*WT.grad_l
            wig_l=WT.l
            wig_grad_l=WT.grad_l
        for corr in corrs:
            s1_s2s=self.s1_s2s[corr]
            for im in np.arange(len(s1_s2s)):
                s1_s2=s1_s2s[im]
                if not self.use_binned_theta:
                    wig_d=WT.wig_d[s1_s2]
                for indx in stack_corr_indxs[corr]:
                    if self.use_binned_theta:
                        wig_d=WT_binned[corr][s1_s2][indx]
                    win=None
                    if self.use_window:
                        win=Win['cl'][corr][indx]

                    xi=get_xi(cl=cl[corr][indx],wig_d=wig_d,wig_norm=wig_norm,
                             xi_bin_utils=xi_bin_utils[s1_s2],bin_xi=self.bin_xi,use_binned_theta=self.use_binned_theta,Win=win)
#                     xi_bi=bin_xi_func(xi=xi,xi_bin_utils=self.xi_bin_utils[s1_s2],bin_xi=self.bin_xi,use_binned_theta=self.use_binned_theta,Win=Win)
                    xi_b+=[xi]#[xi_bi]
        xi_b=np.concatenate(xi_b).ravel()
        return xi_b

    def stack_dat(self,dat,corrs,corr_indxs=None):
        """
            outputs from tomographic caluclations are dictionaries.
            This fucntion stacks them such that the cl or xi is a long
            1-d array and the covariance is N X N array.
            dat: output from tomographic calculations.
            XXX: reason that outputs tomographic bins are distionaries is that
            it make is easier to
            handle things such as binning, hankel transforms etc. We will keep this structure for now.
        """

        if corr_indxs is None:
            corr_indxs=self.stack_indxs

        est=dat['est']
        if est=='xi':
            if self.bin_xi:
                len_bins=len(self.theta_bins)-1
            else:
                k=list(self.WT.theta.keys())[0]
                len_bins=len(self.WT.theta[k])
        else:
            #est='cl_b'
            if self.bin_cl:
                len_bins=len(self.l_bins)-1
            else:
                len_bins=len(self.l)

        n_bins=0
        for corr in corrs:
            n_s1_s2=1
            if est=='xi':
                n_s1_s2=len(self.s1_s2s[corr])
            n_bins+=len(corr_indxs[corr])*n_s1_s2 #np.int64(nbins*(nbins-1.)/2.+nbins)
        D_final=np.zeros(n_bins*len_bins)
        i=0
        for corr in corrs:
            n_s1_s2=1
            if est=='xi':
                s1_s2=self.s1_s2s[corr]
                n_s1_s2=len(s1_s2)

            for im in np.arange(n_s1_s2):
                if est=='xi':
                    dat_c=dat[est][corr][s1_s2[im]]
                else:
                    dat_c=dat[est][corr]#[corr] #cl_b gets keys twice. dask won't allow standard dict merge.. should be fixed
                    
                for indx in corr_indxs[corr]:
                    D_final[i*len_bins:(i+1)*len_bins]=dat_c[indx]
#                     if not np.all(np.isfinite(dat_c[indx])):
#                         print('stack data not finite at',corr,indx,dat_c[indx])
                    i+=1
        print('stack got 2pt')
        if not self.do_cov:
            out={'cov':None}
            out[est]=D_final
            return out

        
        cov_final=np.zeros((len(D_final),len(D_final)))#-999.#np.int(nD2*(nD2+1)/2)
        if self.sparse_cov:
            cov_final=sparse.DOK(cov_final)

        indx0_c1=0
        for ic1 in np.arange(len(corrs)):
            corr1=corrs[ic1]
            indxs_1=corr_indxs[corr1]
            n_indx1=len(indxs_1)

            indx0_c2=indx0_c1
            for ic2 in np.arange(ic1,len(corrs)):
                corr2=corrs[ic2]
                indxs_2=corr_indxs[corr2]
                n_indx2=len(indxs_2)

                corr=corr1+corr2
                n_s1_s2_1=1
                n_s1_s2_2=1
                if est=='xi':
                    s1_s2_1=self.s1_s2s[corr1]
                    s1_s2_2=self.s1_s2s[corr2]
                    n_s1_s2_1=len(s1_s2_1)
                    n_s1_s2_2=len(s1_s2_2)

                for im1 in np.arange(n_s1_s2_1):
                    start_m2=0
                    if corr1==corr2:
                        start_m2=im1
                    for im2 in np.arange(start_m2,n_s1_s2_2):
                        indx0_m1=(im1)*n_indx1*len_bins
                        indx0_m2=(im2)*n_indx2*len_bins
                        for i1 in np.arange(n_indx1):
                            start2=0
                            if corr1==corr2:
                                start2=i1
                            for i2 in np.arange(start2,n_indx2):
                                indx0_1=(i1)*len_bins
                                indx0_2=(i2)*len_bins
                                indx=indxs_1[i1]+indxs_2[i2]
#                                 i_here=np.where(self.cov_indxs[corr]==indx)[0]
                                #i_here=dat['cov']['cov_indxs'][corr].index(indx)
                                i_here=indx
                                if est=='xi':
                                    cov_here=dat['cov'][corr][s1_s2_1[im1]+s1_s2_2[im2]][i_here]['final']
                                else:
                                    cov_here=dat['cov'][corr][i_here]['final_b']

                                if self.sparse_cov:
                                    cov_here=cov_here.todense()
                                # if im1==im2:
                                i=indx0_c1+indx0_1+indx0_m1
                                j=indx0_c2+indx0_2+indx0_m2

                                cov_final[i:i+len_bins,j:j+len_bins]=cov_here
                                cov_final[j:j+len_bins,i:i+len_bins]=cov_here.T
                                if im1!=im2 and corr1==corr2:
                                    i=indx0_c1+indx0_1+indx0_m2
                                    j=indx0_c2+indx0_2+indx0_m1
                                    cov_final[i:i+len_bins,j:j+len_bins]=cov_here.T
                                    cov_final[j:j+len_bins,i:i+len_bins]=cov_here
                                    #gc.collect()

                indx0_c2+=n_indx2*len_bins*n_s1_s2_2
            indx0_c1+=n_indx1*len_bins*n_s1_s2_1
        print('stack got all')
        out={'cov':cov_final}
        out[est]=D_final
        return out
    
def calc_cl(zbin1={}, zbin2={},corr=('shear','shear'),cosmo_params=None,Ang_PS=None):
    """
        Compute the angular power spectra, Cl between two source bins
        zs1, zs2: Source bins. Dicts containing information about the source bins
    """
    clz=Ang_PS.clz
    cls=clz['cls']
    f=clz['cl_f']
    sc=zbin1['kernel_int']*zbin2['kernel_int']
    dchi=clz['dchi']
    cl=np.dot(cls.T*sc,dchi)
#     cl/=f**2 #accounted for in kernel
            # cl*=2./np.pi #FIXME: needed to match camb... but not CCL
    return cl

def calc_pseudo_cl(cl,Win):
    pcl=cl@Win['M']
    return  pcl

def calc_cl_pseudo_cl(zbin1={}, zbin2={},corr=('shear','shear'),cosmo_params=None,Ang_PS=None,Win=None):#FIXME: this can be moved outside the class.thenwe don't need to serialize self.
    """
        Combine calc_cl and calc_pseudo_cl functions
    """
    clz=Ang_PS.clz
    cls=clz['cls']
    f=clz['cl_f']
    sc=zbin1['kernel_int']*zbin2['kernel_int']
    dchi=clz['dchi']
    cl=np.dot(cls.T*sc,dchi)
    pcl=cl@Win['M']
            # cl*=2./np.pi #FIXME: needed to match camb... but not CCL
    return pcl

def bin_cl_func(cl,use_binned_l=False,bin_cl=False,cl_bin_utils=None):
    """
        bins the tomographic power spectra
        results: Either cl or covariance
        bin_cl: if true, then results has cl to be binned
        bin_cov: if true, then results has cov to be binned
        Both bin_cl and bin_cov can be true simulatenously.
    """
    cl_b=None
    if use_binned_l or not bin_cl:
        cl_b=cl*1.
    else:
        cl_b=bin_1d(xi=cl,bin_utils=cl_bin_utils)
    return cl_b

def get_xi(cl=None,wig_d=None,cl_kwargs={},wig_norm=1,
          xi_bin_utils=None,bin_xi=None,use_binned_theta=None,Win=None):
    
    xi=projected_correlation(cl=cl,wig_d=wig_d,norm=wig_norm)
    xib=bin_xi_func(xi=xi,Win=Win,xi_bin_utils=xi_bin_utils,bin_xi=bin_xi,use_binned_theta=use_binned_theta)
    
    return xib

def bin_xi_func(xi=[],Win=None,xi_bin_utils=None,bin_xi=True,use_binned_theta=False):
    xi_b=xi
    if bin_xi and not use_binned_theta: #wig_d is binned when use_binned_l#FIXME: Need window correction when use_binned_l
        if Win is not None: 
            xi=xi*Win['xi']
        xi_b=bin_1d(xi=xi,bin_utils=xi_bin_utils)

    if Win is not None: #win is applied when binning wig_d in case of use_binned_theta
        xi_b/=(Win['xi_b'])
    return xi_b
