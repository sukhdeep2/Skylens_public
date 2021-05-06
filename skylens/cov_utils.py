"""
This file contains a class with functions for covariance calculations.
"""

import os,sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d
from scipy.special import jn, jn_zeros
from skylens.wigner_functions import *
from skylens.binning import *
from skylens.cov_tri import *
from skylens.utils import *
import healpy as hp

d2r=np.pi/180.
sky_area=np.pi*4/(d2r)**2 #in degrees


class Covariance_utils():
    def __init__(self,f_sky=0,l=None,logger=None,l_cut_jnu=None,do_sample_variance=True,
                 use_window=True,window_l=None,window_file=None, do_xi=False,
                 use_binned_l=False,xi_SN_analytical=False,SN={},use_binned_theta=False,
                 do_cov=False,SSV_cov=False,tidal_SSV_cov=False,WT=None, #WT_binned_cov={},
                 Tri_cov=False,sparse_cov=False,cl_bin_utils=None,xi_bin_utils=None,
                 do_pseudo_cl=True,bin_cl=True
                ):
        self.__dict__.update(locals()) #assign all input args to the class as properties
        self.binning=binning()
        self.sample_variance_f=1
        if do_sample_variance is not None:
            if not do_sample_variance:
                self.sample_variance_f=0 #remove sample_variance from gaussian part

        self.set_window_params()

        self.gaussian_cov_norm=(2.*l+1.)*np.gradient(l) #need Delta l here. Even when
                                                        #binning later
                                                        #take care of f_sky later
#         self.gaussian_cov_norm_2D=np.outer(np.sqrt(self.gaussian_cov_norm),np.sqrt(self.gaussian_cov_norm))
        if self.Tri_cov:
            self.CTR=cov_matter_tri(k=self.l)
#         if use_window and use_binned_l:
#             dl=np.sqrt(np.gradient(self.l))
#             self.dl_norm=np.outer(dl,dl)
#         dict_size_pickle(self.__dict__,print_prefact='cov utils self size: ',depth=2)

    def set_window_params(self): #FIXME: size issues for large computes
        """
            set survey area and analytical window power spectra 
            based on f_sky input. f_sky can be a float or a 
            dictionary of values for different correlation
            pairs and quadruples (four tracers, for covariance).
        """
        if self.f_sky is None:
            return
        if self.use_window:
            return
        if isinstance(self.f_sky,float):
            self.Om_W=4*np.pi*self.f_sky
            self.Win,self.Win0=self.window_func()
        else:
            self.Om_W={k1:{}for k1 in self.f_sky.keys()}
            self.Win={k1:{}for k1 in self.f_sky.keys()}
            self.Win0={k1:{}for k1 in self.f_sky.keys()}
            for k1 in self.f_sky.keys():
                for k2 in self.f_sky[k1].keys():
                    self.Om_W[k1][k2]=4*np.pi*self.f_sky[k1][k2]
                    self.Win[k1][k2],self.Win0[k1][k2]=self.window_func(self.Om_W[k1][k2])

    def window_func(self,Om_W=None):
        """
        Set a default unit window used in some calculations (when survey window is not supplied.)
        """
        if Om_W is None:
            Om_W=self.Om_W
        if self.window_file is not None:
            W=np.genfromtxt(self.window_file,names=('l','cl'))
            window_l=W['l']
            self.Win=W['cl']
            win_i=interp1d(window_l,self.Win,bounds_error=False,fill_value=0)
            self.Win0=win_i(self.window_l) #this will be useful for SSV
            return

        #same as eq. 5.4 of https://arxiv.org/pdf/1711.07467.pdf
        if self.window_l is None:
            self.window_l=np.arange(100)

        l=self.window_l
        theta_win=np.sqrt(Om_W/np.pi)
        l_th=l*theta_win
        Win0=2*jn(1,l_th)/l_th
        Win0=np.nan_to_num(Win0)
        Win0=Win0**2 #p-Cl equivalent
#         Win0*=self.f_sky  #FIXME???? Missing 4pi? Using Om_w doesnot match healpix calculation.
        
        win_i=interp1d(l,Win0,bounds_error=False,fill_value=0)
        Win=win_i(self.window_l) #this will be useful for SSV
        Win0=win_i(self.l)
        return Win,Win0

    def sigma_win_calc(self,clz,Win=None,tracers=None,zs_indx=[]):#cls_lin, Win_cl=None,Om_w12=None,Om_w34=None):
        """
        compute mass variance on the scale of the survey window.
        """
        cls_lin=clz['cls_lin']
        if Win is None: #use defaulr f_sky window
            if isinstance(self.f_sky,float):
                Win_cl=self.Win
                Om_w12=self.Om_W
                Om_w34=self.Om_W
            else:
                Win_cl=self.Win[tracers][zs_indx]#FIXME: Need 4 window term here
                Om_w12=self.Om_W[(tracers[0],tracers[1])][(zs_indx[0],zs_indx[1])]
                Om_w34=self.Om_W[(tracers[2],tracers[3])][(zs_indx[2],zs_indx[3])]
        else:
            Win_cl=Win['mask_comb_cl']  #['cov'][tracers][zs_indx]
            Om_w12=Win['Om_w12'] #[tracers][zs_indx]
            Om_w34=Win['Om_w34'] #[tracers][zs_indx]
            
        sigma_win=np.dot(Win_cl*np.gradient(self.window_l)*(2*self.window_l+1),cls_lin.T)
        sigma_win/=Om_w12*Om_w34
        return sigma_win

    def corr_matrix(self,cov=[]):
        """
        convert covariance matrix into correlation matrix.
        """
        diag=np.diag(cov)
        return cov/np.sqrt(np.outer(diag,diag))

    def get_SN(self,SN,tracers,z_indx):
        """
            get shot noise/shape noise for given set of tracer and z_bins.
            Note that we assume shot noise is function of ell in general. 
            Default is to set that function to constant. For e.x., CMB lensing
            noise is function of ell. Galaxy shot noise can also vary with ell, 
            especially on small scales.
        """
        SN2={}
        SN2[13]=SN[(tracers[0],tracers[2])][:,z_indx[0], z_indx[2] ] if SN.get((tracers[0],tracers[2])) is not None else np.zeros_like(self.l)
        SN2[24]=SN[(tracers[1],tracers[3])][:,z_indx[1], z_indx[3] ] if SN.get((tracers[1],tracers[3])) is not None else np.zeros_like(self.l)
        SN2[14]=SN[(tracers[0],tracers[3])][:,z_indx[0], z_indx[3] ] if SN.get((tracers[0],tracers[3])) is not None else np.zeros_like(self.l)
        SN2[23]=SN[(tracers[1],tracers[2])][:,z_indx[1], z_indx[2] ] if SN.get((tracers[1],tracers[2])) is not None else np.zeros_like(self.l)

        return SN2

    def get_CV_cl(self,cls,tracers,z_indx):
        """
        Get the tracer power spectra, C_ell, for covariance calculations.
        """
        CV2={}
        for k in cls.keys():
          CV2[k]=cls[k]*self.sample_variance_f
#         CV2[13]=cls[(tracers[0],tracers[2])] [(z_indx[0], z_indx[2]) ]*self.sample_variance_f
#         CV2[24]=cls[(tracers[1],tracers[3])][(z_indx[1], z_indx[3]) ]*self.sample_variance_f
#         CV2[14]=cls[(tracers[0],tracers[3])][(z_indx[0], z_indx[3]) ]*self.sample_variance_f
#         CV2[23]=cls[(tracers[1],tracers[2])][(z_indx[1], z_indx[2]) ]*self.sample_variance_f
        return CV2

    def cl_gaussian_cov_window(self,cls,SN,tracers,z_indx,Win,Bmode_mf=1):
        """
        Computes the power spectrum gaussian covariance, with proper factors of window. We have separate function for the 
        case when window is not supplied.
        """
        SN2=self.get_SN(SN,tracers,z_indx)
        CV=self.get_CV_cl(cls,tracers,z_indx)
        CV_B=self.get_CV_B_cl(cls,tracers,z_indx)
        
        G={1324:0,1423:0}
        cv_indxs={1324:(13,24),1423:(14,23)}
        add_EB=1
        if self.do_xi and np.all(np.array(tracers)=='shear'):
            add_EB+=1
        
        
        if self.use_window and self.use_binned_l:
            dl=np.sqrt(np.gradient(self.l))
            dl_norm=np.outer(dl,dl)
        
        for corr_i in [1324,1423]:
            W_pm=Win['W_pm'][corr_i]
            c1=cv_indxs[corr_i][0]
            c2=cv_indxs[corr_i][1]

            for k in Win['M'][corr_i].keys():
                for wp in W_pm:
                    CV2=CV
                    for a_EB in np.arange(add_EB):
                        if wp<0 or a_EB>0:
                            CV2=CV_B
                        if k=='clcl': 
                            G_t=np.outer(CV2[c1],CV2[c2])
                        if k=='Ncl': 
                            G_t=np.outer(SN2[c1],CV2[c2])
                        if k=='clN': 
                            G_t=np.outer(CV2[c1],SN2[c2])
                        if k=='NN': 
                            G_t=np.outer(SN2[c1],SN2[c2])
                        if a_EB>0:
                            G_t*=Bmode_mf #need to -1 for xi+/- cross covariance
                        # if bin_window:
                        #     G_t=self.binning.bin_2d(cov=G_t,bin_utils=bin_utils)
                        if not self.use_binned_l:
                            G[corr_i]+=G_t*Win['M'][corr_i][k][wp]
                        else:
                            G[corr_i]+=G_t*Win['M'][corr_i][k][wp]/dl_norm #FIXME: consider using factor of 2l+1 in window and cov separately.
#                             G[corr_i]/=np.gradient(self.l)

        return G[1324],G[1423]
        
    def get_CV_B_cl(self,cls,tracers,z_indx): #
        """ 
            Return power spectra and noise contributions for shear B-mode covariance. We 
            assume that the shear B-mode power spectra is zero (there is still E-mode contribution 
            due to leakage caused by window).
        """
        sv_f={13:1,24:1,14:1,23:1}
        if tracers[0]=='shear' and tracers[2]=='shear':
            sv_f[13]=0
        if tracers[1]=='shear' and tracers[3]=='shear':
            sv_f[24]=0
        if tracers[0]=='shear' and tracers[3]=='shear':
            sv_f[14]=0
        if tracers[1]=='shear' and tracers[2]=='shear':
            sv_f[23]=0
            
        CV2={}
        for k in sv_f.keys():
            CV2[k]=cls[k]*self.sample_variance_f*sv_f[k]
#         CV2[13]=cls[(tracers[0],tracers[2])] [(z_indx[0], z_indx[2]) ]*self.sample_variance_f*sv_f[13]
#         CV2[24]=cls[(tracers[1],tracers[3])][(z_indx[1], z_indx[3]) ]*self.sample_variance_f*sv_f[24]
#         CV2[14]=cls[(tracers[0],tracers[3])][(z_indx[0], z_indx[3]) ]*self.sample_variance_f*sv_f[14]
#         CV2[23]=cls[(tracers[1],tracers[2])][(z_indx[1], z_indx[2]) ]*self.sample_variance_f*sv_f[23]
        return CV2
            
    def cl_gaussian_cov(self,cls,SN,tracers,z_indx,Bmode_mf=1): #no-window covariance
        """
        Gaussian covariance for the case when no window is supplied and only f_sky is used.
        """

        SN2=self.get_SN(SN,tracers,z_indx)
        CV=self.get_CV_cl(cls,tracers,z_indx)
        
        def get_G4(CV,SN2):
            """
            return the two sub covariance matrix.
            """
            G1324= ( CV[13]+ SN2[13])#/self.gaussian_cov_norm
                 #get returns None if key doesnot exist. or 0 adds 0 is SN is none

            G1324*=( CV[24]+ SN2[24])

            G1423= ( CV[14]+ SN2[14])#/self.gaussian_cov_norm

            G1423*=(CV[23]+ SN2[23])
            return G1324,G1423
            
        G1324,G1423=get_G4(CV,SN2)
        if self.do_xi and np.all(np.array(tracers)=='shear'):
            CVB=self.get_CV_B_cl(cls,tracers,z_indx)
            G1324_B,G1423_B=get_G4(CVB,SN2)
            G1324+=G1324_B*Bmode_mf
            G1423+=G1423_B*Bmode_mf

        G1423=np.diag(G1423)
        G1324=np.diag(G1324)
        
        Norm=np.pi*4
        
        fs1324=1
        fs0=1
        fs1423=1
        if self.f_sky is not None:
            if isinstance(self.f_sky,float):
                fs1324=self.f_sky
                fs0=self.f_sky**2
                fs1423=self.f_sky
            else:
                fs1324=self.f_sky[tracers][z_indx]#np.sqrt(f_sky[tracers[0],tracers[2]][z_indx[0],z_indx[2]]*f_sky[tracers[1],tracers[3]][z_indx[1],z_indx[3]])
                fs0=self.f_sky[tracers[0],tracers[1]][z_indx[0],z_indx[1]] * f_sky[tracers[2],tracers[3]][z_indx[2],z_indx[3]]
                fs1423=self.f_sky[tracers][z_indx]#np.sqrt(f_sky[tracers[0],tracers[3]][z_indx[0],z_indx[3]]*f_sky[tracers[1],tracers[2]][z_indx[1],z_indx[2]])
        
        gaussian_cov_norm_2D=np.outer(np.sqrt(self.gaussian_cov_norm),np.sqrt(self.gaussian_cov_norm))
        if not self.do_xi:
            G1324/=gaussian_cov_norm_2D/fs1324*fs0
            G1423/=gaussian_cov_norm_2D/fs1423*fs0
        else:
            G1324*=fs1324/fs0/Norm
            G1423*=fs1423/fs0/Norm

        if np.all(np.array(tracers)=='shear') and Bmode_mf<0:
            G1324*=Bmode_mf
            G1423*=Bmode_mf
            
        return G1324,G1423
        
    def xi_gaussian_cov(self,cls,SN,tracers,z_indx,Win,WT_kwargs,Bmode_mf=1):
        """
        Gaussian covariance for correlation functions. If no window is provided, 
        returns output of xi_gaussian_cov_no_win
        """
        #FIXME: Need to check the case when we are only using bin centers.
        if Win is None:
            return self.xi_gaussian_cov_no_win(cls,SN,tracers,z_indx,Win,WT_kwargs,Bmode_mf)
        SN2=self.get_SN(SN,tracers,z_indx)
        CV=self.get_CV_cl(cls,tracers,z_indx)
        CV_B=self.get_CV_B_cl(cls,tracers,z_indx)
        
        G={1324:0,1423:0}
        cv_indxs={1324:(13,24),1423:(14,23)}
        
        Norm=np.pi*4
        
        add_EB=1
        if self.do_xi and np.all(np.array(tracers)=='shear'):
            add_EB+=1
        
        if Win is None:
            if isinstance(self.f_sky,float):
                fs1324=self.f_sky
                fs0=self.f_sky**2
                fs1423=self.f_sky
            else:
                fs1324=f_sky[tracers][z_indx]
                fs0=f_sky[tracers[0],tracers[1]][z_indx[0],z_indx[1]] * f_sky[tracers[2],tracers[3]][z_indx[2],z_indx[3]]
                fs1423=f_sky[tracers][z_indx]
            Norm=Norm*fs0/fs1324 #FIXME: This is an approximation. Need better expression for correlation functions

        
        for corr_i in [1324,1423]:
            W_pm=Win['W_pm'][corr_i]
            c1=cv_indxs[corr_i][0]
            c2=cv_indxs[corr_i][1]

            for k in Win['xi_cov'][corr_i].keys():
                for wp in W_pm:
                    CV2=CV
                    for a_EB in np.arange(add_EB):
                        if wp<0 or a_EB>0:
                            CV2=CV_B
                        if k=='clcl': 
#                             G_t=np.outer(CV2[c1],CV2[c2])
                            G_t=CV2[c1]*CV2[c2]
                        elif k=='Ncl': 
#                             G_t=np.outer(SN2[c1],CV2[c2])
                            G_t=SN2[c1]*CV2[c2]
                        elif k=='clN': 
#                             G_t=np.outer(CV2[c1],SN2[c2])
                            G_t=CV2[c1]*SN2[c2]
                        elif k=='NN' and not self.xi_SN_analytical: 
#                             G_t=np.outer(SN2[c1],SN2[c2])
                            G_t=SN2[c1]*SN2[c2]
                        if k=='NN' and self.xi_SN_analytical:
#                             if not self.use_binned_theta:
#                                 G_t=np.diag(SN2[c1][0]*SN2[c2][0]/WT_kwargs['theta']['s1_s2']) #Fixme: wont' work with binned_theta
#                             else:
                                G_t=np.diag(SN2[c1][0]*SN2[c2][0]/WT_kwargs['wig_theta']/WT_kwargs['wig_grad_theta']) 
                        else:
#                             th,G_t=WT.projected_covariance2(cl_cov=G_t,**WT_kwargs)
                            th,G_t=self.WT.projected_covariance(cl_cov=G_t,**WT_kwargs)
                        if Win is not None:
#                             G_t*=np.outer(Win['xi'][12][k],Win['xi'][34][k])
                            G_t*=Win['xi_cov'][corr_i][k]
                            
                        G_t/=Norm

                        if a_EB>0:
                            G_t*=Bmode_mf #need to -1 for xi+/- cross covariance
                        G[corr_i]+=G_t
        return G[1324]+G[1423]

    def xi_gaussian_cov_no_win(self,cls,SN,tracers,z_indx,Win,WT_kwargs,Bmode_mf=1):
        """
        Gaussian covariance for correlation functions, when no window is provided.
        """
        #FIXME: Need to implement the case when we are only using bin centers.
        SN2=self.get_SN(SN,tracers,z_indx)
        CV=self.get_CV_cl(cls,tracers,z_indx)
        CV_B=self.get_CV_B_cl(cls,tracers,z_indx)
        
        G={1324:0,1423:0}
        cv_indxs={1324:(13,24),1423:(14,23)}
        
        Norm=np.pi*4
        
        add_EB=1
        if self.do_xi and np.all(np.array(tracers)=='shear'):
            add_EB+=1
        
        if isinstance(self.f_sky,float):
            fs1324=self.f_sky
            fs0=self.f_sky**2
            fs1423=self.f_sky
        else:
            fs1324=f_sky[tracers][z_indx]
            fs0=f_sky[tracers[0],tracers[1]][z_indx[0],z_indx[1]] * f_sky[tracers[2],tracers[3]][z_indx[2],z_indx[3]]
            fs1423=f_sky[tracers][z_indx]
        Norm=Norm*fs0/fs1324

        G_t=0
        G_t_SN=0
        for corr_i in [1324,1423]:
            c1=cv_indxs[corr_i][0]
            c2=cv_indxs[corr_i][1]

            for k in ['clcl','NN','Ncl','clN']:
                CV2=CV
                G_t_SNi=0
                for a_EB in np.arange(add_EB):
                    if a_EB>0:
                        CV2=CV_B
                    if k=='clcl': 
                        G_ti=CV2[c1]*CV2[c2]
                    elif k=='Ncl': 
                        G_ti=SN2[c1]*CV2[c2]
                    elif k=='clN': 
                        G_ti=CV2[c1]*SN2[c2]
                    elif k=='NN' and not self.xi_SN_analytical: 
                        G_ti=SN2[c1]*SN2[c2]
                    if k=='NN' and self.xi_SN_analytical:
                        if not self.use_binned_theta:
                                G_t_SNi=SN2[c1][-1]*SN2[c2][-1]/(WT_kwargs['theta']['s1_s2'])#Fixme: wont' work with binned_theta
                        else:
                                G_t_SNi=SN2[c1][-1]*SN2[c2][-1]/self.WT.theta_bins_center/self.WT.delta_theta_bins

                    if a_EB>0:
                        G_ti*=Bmode_mf #need to -1 for xi+/- cross covariance
                        G_t_SNi*=Bmode_mf #need to -1 for xi+/- cross covariance
                    G_t+=G_ti
                    G_t_SN+=G_t_SNi
        
        th,G_t=self.WT.projected_covariance(cl_cov=G_t,**WT_kwargs)
        #print('cov utils xi_gaussina_cov',G_t.shape)
        if np.any(G_t_SN!=0):
            G_t+=np.diag(G_t_SN)
        G_t/=Norm
        if a_EB>0:
            G_t*=Bmode_mf #need to -1 for xi+/- cross covariance
        G[corr_i]+=G_t
        return G[1324]+G[1423]

    
    def cov_four_kernels(self,z_bins={},Ang_PS=None):
        """
        product of four tracer kernels, for non-gaussian covariance.
        """
        clz=Ang_PS.clz
        zs1=z_bins[0]
        zs2=z_bins[1]
        zs3=z_bins[2]
        zs4=z_bins[3]
        sig_cL=zs1['Gkernel_int']*zs2['Gkernel_int']*zs3['Gkernel_int']*zs4['Gkernel_int']
#Only use lensing kernel... not implemented for galaxies (galaxies have magnification, which is included)
        sig_cL*=clz['dchi']
        return sig_cL

    
    def cl_cov_connected(self,z_bins=None,cls=None, tracers=[],Win_cov=None,Win_cl=None,clz=None,sig_cL=None,zs_indx=None):
        """
        Non gaussian covariance, for power spectra.
        """
        cov={}
        cov['SSC']=0
        cov['Tri']=0
        Win=None
        if Win_cov is not None:
            Win=Win_cov#[zs_indx]

#         if self.Tri_cov or self.SSV_cov:
#             sig_cL=self.cov_four_kernels(z_bins=z_bins)

        if self.SSV_cov :
            sigma_win=self.sigma_win_calc(clz=clz,Win=Win,tracers=tracers,zs_indx=zs_indx)

            clr=clz['clsR']
            if self.tidal_SSV_cov:
                clr=clz['clsR']+ clz['clsRK']/6.

            sig_F=np.sqrt(sig_cL*sigma_win) #kernel is function of l as well due to spin factors
            clr=clr*sig_F.T
            cov['SSC']=np.dot(clr.T,clr)

        if self.Tri_cov:
            cov['Tri']=self.CTR.cov_tri_zkernel(P=clz['cls'],z_kernel=sig_cL/clz['chi']**2,chi=clz['chi']) #FIXME: check dimensions, get correct factors of length.. chi**2 is guessed from eq. A3 of https://arxiv.org/pdf/1601.05779.pdf ... note that cls here is in units of P(k)/chi**2
            
            if isinstance(self.f_sky,float):
                fs0=self.f_sky
            else:
                fs0=self.f_sky[tracers[0],tracers[1]][zs_indx[0],zs_indx[1]]
                fs0*=self.f_sky[tracers[2],tracers[3]][zs_indx[2],zs_indx[3]]
                fs0=np.sqrt(fs0)
    #             cov['Tri']/=self.cov_utils.gaussian_cov_norm_2D**2 #Since there is no dirac delta, there should be 2 factor of (2l+1)dl... eq. A3 of https://arxiv.org/pdf/1601.05779.pdf
            cov['Tri']/=fs0 #(2l+1)f_sky.. we didnot normalize gaussian covariance in trispectrum computation.

        return cov['SSC'],cov['Tri']

    def bin_cl_cov_func(self,cov=None,cl_bin_utils=None):#moved out of class. This is no longer used
        """
            bins the tomographic power spectra
            results: Either cl or covariance
            bin_cl: if true, then results has cl to be binned
            bin_cov: if true, then results has cov to be binned
            Both bin_cl and bin_cov can be true simulatenously.
        """
        cov_b=None
        if self.use_binned_l or not self.bin_cl:
            cov_b=cov*1.
        else:
            cov_b=self.binning.bin_2d(cov=cov,bin_utils=cl_bin_utils)
        return cov_b
    
def cl_cov(zs_indx,CU,z_bins=None,sig_cL=None,cls=None, tracers=[],Win_cov=None,Win_cl1=None,Win_cl2=None,
           cov_utils=None,Ang_PS=None,SN=None,cl_bin_utils=None):
    """
        Computes the covariance between any two tomographic power spectra.
        cls: tomographic cls already computed before calling this function
        zs_indx: 4-d array, noting the indices of the source bins involved
        in the tomographic cls for which covariance is computed.
        For ex. covariance between 12, 56 tomographic cross correlations
        involve 1,2,5,6 source bins
    """
    self=CU
    clz=Ang_PS.clz
    cov={}
    cov['z_indx']=zs_indx
    cov['tracers']=tracers
    cov['final']=None

    cov['G']=None
    cov['G1324_B']=None;cov['G1423_B']=None

    Win=None
    if Win_cov is not None:
        Win=Win_cov

    if self.use_window and self.do_pseudo_cl:
        cov['G1324'],cov['G1423']=self.cl_gaussian_cov_window(cls,SN,
                                        tracers,zs_indx,Win,)
    else:
        fs=self.f_sky
        if self.do_xi and self.use_window : #in this case we need to use a separate function directly from xi_cov
            cov['G1324']=0
            cov['G1423']=0
        else:
            cov['G1324'],cov['G1423']=self.cl_gaussian_cov(cls,SN,tracers,zs_indx)
    cov['G']=cov['G1324']+cov['G1423']
    cov['final']=cov['G']
    cov['SSC'],cov['Tri']=self.cl_cov_connected(zs_indx=zs_indx,cls=cls,clz=clz, tracers=tracers,sig_cL=sig_cL)#,Win_cov=None,Win_cl=None)
    if self.use_window and (self.SSV_cov or self.Tri_cov) and self.do_pseudo_cl: #Check: This is from writing p-cl as M@cl... cov(p-cl)=M@cov(cl)@M.T ... separate  M when different p-cl
        M1=Win_cl1['M'] #12
        M2=Win_cl2['M'] #34
        if self.use_binned_l:
            for k in ['SSC','Tri']:
                cov[k]=self.bin_cl_cov_func(cov=cov[k])
        cov['final']=cov['G']+ M1@(cov['SSC']+cov['Tri'])@M2.T
    else:
        cov['final']=cov['G']+cov['SSC']+cov['Tri']

    if not self.do_xi:
        cov['G1324']=None #save memory
        cov['G1423']=None

    for k in ['final','G','SSC','Tri']:#no need to bin G1324 and G1423
        if self.bin_cl:
            cov[k+'_b']=self.bin_cl_cov_func(cov=cov[k],cl_bin_utils=cl_bin_utils)
        else:
            cov[k+'_b']=cov[k]

        if self.sparse_cov and cov[k+'_b'] is not None:
            if k!='final':
                # print('deleting',k)
                cov[k+'_b']=None
                continue
            cov[k+'_b']=sparse.COO(cov[k+'_b'])
        if not self.do_xi and self.bin_cl:
            del cov[k]
    return cov

def xi_cov(cov_indx,CU,cov_cl=None,cls={},s1_s2=None,s1_s2_cross=None,
           corr1=[],corr2=[], Win_cov=None,Win_cl1=None,Win_cl2=None,SN=None,
          z_bins=None,sig_cL=None,WT=None,WT_kwargs={},xi_bin_utils=None,Ang_PS=None):
    """
        Computes covariance of xi, by performing 2-D hankel transform on covariance of Cl.
        In current implementation of hankel transform works only for s1_s2=s1_s2_cross.
        So no cross covariance between xi+ and xi-.
    """
    clz=Ang_PS.clz
    self=CU
    z_indx=cov_indx
    indxs_1=(z_indx[0],z_indx[1])
    indxs_2=(z_indx[2],z_indx[3])

    tracers=corr1+corr2
    if s1_s2_cross is None:
        s1_s2_cross=s1_s2
    cov_xi={}

    if self.WT.name=='Hankel' and s1_s2!=s1_s2_cross:
        n=len(self.theta_bins)-1
        cov_xi['final']=np.zeros((n,n))
        return cov_xi

    SN1324=0
    SN1423=0
    wig_d1=None
    wig_d2=None
    if self.use_binned_l:
        wig_d1=WT_kwargs['wig_d1']
        wig_d2=WT_kwargs['wig_d2']

    Win=None
    if Win_cov is not None:
        Win=Win_cov#[z_indx]

    bf=1
    if np.all(np.array(tracers)=='shear') and not s1_s2==s1_s2_cross: #cross between xi+ and xi-
        bf=-1

    cov_xi['G']=self.xi_gaussian_cov(cls,SN,tracers,z_indx,Win,WT_kwargs,bf)

    if not self.use_binned_theta:
        cov_xi['G']=self.binning.bin_2d(cov=cov_xi['G'],bin_utils=xi_bin_utils)

    cov_xi['SSC']=0
    cov_xi['Tri']=0

    if cov_cl is None:
        cov_cl={}
        cov_cl['SSC'],cov_cl['Tri']=self.cl_cov_connected(zs_indx=cov_indx,cls=cls,Win_cov=Win_cov,tracers=corr1+corr2,Win_cl=None,sig_cL=sig_cL,clz=clz)

    if self.SSV_cov:
        th0,cov_xi['SSC']=self.WT.projected_covariance2(l_cl=self.l,s1_s2=s1_s2,
                                                        s1_s2_cross=s1_s2_cross,
                                                        wig_d1=wig_d1,
                                                        wig_d2=wig_d2,
                                                        cl_cov=cov_cl['SSC'])
        if not self.use_binned_theta:
            cov_xi['SSC']=self.binning.bin_2d(cov=cov_xi['SSC'],bin_utils=xi_bin_utils)
    if self.Tri_cov:
        th0,cov_xi['Tri']=self.WT.projected_covariance2(l_cl=self.l,s1_s2=s1_s2,
                                                        s1_s2_cross=s1_s2_cross,
                                                        wig_d1=wig_d1,
                                                        wig_d2=wig_d2,
                                                        cl_cov=cov_cl['Tri'])
        if not self.use_binned_theta:
            cov_xi['Tri']=self.binning.bin_2d(cov=cov_xi['Tri'],bin_utils=xi_bin_utils)

    cov_xi['final']=cov_xi['G']+cov_xi['SSC']+cov_xi['Tri']
    if self.use_window:
        cov_xi['G']/=(Win_cl1['xi_b']*Win_cl2['xi_b'])
        cov_xi['final']=cov_xi['G']+cov_xi['SSC']+cov_xi['Tri']
#         cov_xi['final']/=(Win_cl1['xi_b']*Win_cl2['xi_b'])#FIXME: SSC is not multiplied with window.

    if self.sparse_cov:
        for k in ['G','SSC','Tri','final']:
#                 print('xi_cov',corr1,corr2,indxs_1,indxs_2,cov_xi[k].shape,self.WT.theta[(0,0)].shape)
            if  np.atleast_1d(cov_xi[k]).ndim>1:
                cov_xi[k]=sparse.COO(cov_xi[k])

    return cov_xi
