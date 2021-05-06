#FIXME: 
# 1. need to save SN from kappa_class
# 2. save window correlation from treecorr and theory

import numpy as np
import healpy as hp
import treecorr
from skylens.binning import *
from skylens.utils import *
from skylens import *
from dask import delayed
from scipy.stats import norm,mode,skew,kurtosis,percentileofscore

import copy
from resource import getrusage, RUSAGE_SELF
import psutil
from distributed.utils import format_bytes
from jk_utils import *
from sim_args import corrs,corr_ll,corr_gg,corr_ggl,corr_config

Master_algs=['iMaster','Master','nMaster'] #['unbinned','Master','nMaster','iMaster']
class Sim_jk():
    def __init__(self,nsim=150,njk=0,do_norm=False,cl0=None,kappa_class=None,kappa0=None,use_shot_noise=True,
                 nside=None,lognormal=False,lognormal_scale=1,add_SSV=True,add_tidal_SSV=True,
                 use_cosmo_power=True,Master_algs=Master_algs,seed=12334,do_xi=False,#fsky=0,
                 add_blending=False,blending_coeff=-2,fiber_coll_coeff=-1,jkmap=None,kappa_class_xib=None,
                 subsample=False,skylens_kwargs=None,njobs_submit_per_worker=5
                ):
        
        self.__dict__.update(locals()) #assign all input args to the class as properties
        
        self.scheduler_info=kappa_class.scheduler_info
        workers=list(self.scheduler_info['workers'].keys())
        self.nworkers=len(workers)
        client=client_get(scheduler_info=self.scheduler_info)
        
        self.ndim=len(kappa_class.corrs)
        self.Nl_bins=len(kappa_class.l_bins)-1
        kappa_class.gather_data()
        
        self.jk_stat_keys=['jk_mean','jk_err','jk_cov']

        self.sim_clb_shape=None
        self.sim_xib_shape=None
        if self.kappa_class.do_pseudo_cl:
            self.get_coupling_matrices_jk(kappa_class=kappa_class)
            self.sim_clb_shape=(self.nsim,self.Nl_bins*(self.ndim+1)) #shear-shear gives 2 corrs, EE and BB.. 
        
        if self.do_xi:
            self.xi_window_norm={}
            self.n_th_bins=len(self.kappa_class.theta_bins)-1
            self.xi_window_norm=get_xi_window_norm_jk(Sim_jk=self)
            self.sim_xib_shape=(self.nsim,self.n_th_bins*(self.ndim+1)) #shear-shear gives 2 corrs, xi+ and xi-
            self.get_xi_coupling_matrices()

        self.mask={}
        self.window={}
        self.window_N={}
        self.window_N_norm={}
        if self.kappa_class.use_window:
            for tracer in self.kappa_class.z_bins.keys():
                self.window[tracer]=kappa_class.tracer_utils.z_win[tracer][0]['window']
                self.window_N[tracer]=kappa_class.tracer_utils.z_win[tracer][0]['window_N']
                self.mask[tracer]=self.window[tracer]==hp.UNSEEN
                self.window_N_norm[tracer]=(self.window_N[tracer][self.mask[tracer]]**2).mean()
                
        if self.cl0 is None:
            self.cl0={}
            self.pcl0={}
            clG0=self.kappa_class.cl_tomo() 
            for corr in kappa_class.corrs:
                self.cl0[corr]=clG0['cl'][corr][(0,0)].compute()
#                 self.cl0_b[corr]=clG0['cl_b'][corr][(0,0)].compute()
                if self.kappa_class.do_pseudo_cl:
                    self.pcl0[corr]=clG0['pseudo_cl_b'][corr][(0,0)].compute()
                    if corr==corr_ll:
                        self.pcl0['shear_B']=clG0['cl_b'][corr][(0,0)].compute()@self.coupling_M['full']['coupling_M_binned']['iMaster']['shear_B']
        if self.do_xi:
            xiG_L0=kappa0.xi_tomo()
            self.xi_L0=client.compute(xiG_L0['stack']).result() #.compute()
            xiWG_L=kappa_class.xi_tomo()
            self.xiW_L=client.compute(xiWG_L['stack']).result() #.compute()  #####mem crash

        self.clN0={}
        self.pclN0={}
        self.shot_noise={}
        kappa_class.gather_data()
        for corr in self.kappa_class.corrs: #ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True

            shot_noise=kappa_class.SN[corr_gg][:,0,0]*0
            if corr[0]==corr[1]:
                shot_noise=kappa_class.SN[corr][:,0,0]
                self.window_N_norm[corr[0]]*=shot_noise
            self.shot_noise[corr]=shot_noise*use_shot_noise
            self.clN0[corr]=self.shot_noise[corr]#@self.coupling_M['coupling_M_N'][corr]
            self.cl0[corr]=self.cl0[corr]*use_cosmo_power#+shot_noise
            
            if self.kappa_class.do_pseudo_cl and corr[0]==corr[1]:
                self.pclN0[corr]=self.shot_noise[corr]@self.coupling_M['full']['coupling_M_N'][corr]
                if corr==corr_ll:
                    self.pclN0['shear_B']=self.shot_noise[corr]@self.coupling_M['full']['coupling_M_N']['shear_B']
            
            if corr==corr_ll:
                self.cl0['shear_B']=self.cl0[corr]*0
                self.clN0['shear_B']=self.shot_noise[corr]

        print('ndim:',self.ndim)
        self.outp={}
        self.outp['cl0_0']=self.cl0.copy()
        self.outp['clN0_0']=self.clN0.copy()
        self.outp['ndim']=self.ndim

        SN=kappa_class.SN
                
        if njk==0:
            self.jk_stat_keys=[]

        cl_maps={}
#         lmax=max(l)
#         lmin=min(l)

        if self.add_SSV:
            self.set_SSV_params()

        self.cl0_b={corr: kappa_class.binning.bin_1d(xi=self.cl0[corr],bin_utils=kappa_class.cl_bin_utils) for corr in kappa_class.corrs} 

        self.cl_b=None;self.pcl_b=None;self.xi_b=None

        self.binning_func=kappa_class.binning.bin_1d
        self.binning_utils=kappa_class.cl_bin_utils

        if self.ndim>1:
            self.cl0=(self.cl0[corr_gg],self.cl0[corr_ll],self.cl0['shear_B'],self.cl0[corr_ggl])#ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True

            self.clN0=(self.clN0[corr_gg],self.clN0[corr_ll],self.clN0['shear_B'],self.clN0[corr_ggl])#ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True

        else:
            self.cl0=cl0[corr_gg]
            self.clN0=clN0[corr_gg]

        self.gamma_trans_factor=None
        if self.lognormal:
            self.set_gamma_trans_factor()

        self.corr_order=['shear_B',corr_ll,corr_ggl,corr_gg] #order in which sim corrs are output... some shuffling done in process_pcli
        
        print('generating maps')
        self.generate_maps()
        self.get_stats()
        
        self.outp['cl0']=self.cl0
        self.outp['cl0_b']=self.cl0_b
        self.outp['clN0']=self.clN0
        self.outp['pclN0']=self.pclN0
        self.outp['pcl0']=self.pcl0
        self.outp['corr_order']=self.corr_order
        self.outp['window_N_norm']=self.window_N_norm
        
        self.outp['nsim']=nsim
        self.outp['nside']=nside
        self.outp['size']=self.nsim
#         self.outp['fsky']=self.fsky
        self.outp['l']=self.kappa_class.l
        self.outp['window_l']=self.kappa_class.window_l
        self.outp['Win']=self.kappa_class.Win
        self.outp['l_bins']=self.kappa_class.l_bins
        self.outp['use_shot_noise']=self.use_shot_noise

        if self.kappa_class.do_pseudo_cl:
            self.outp['coupling_M']=self.coupling_M
        if self.kappa_class.do_xi:
            self.outp['wig_d_binned']=self.wig_d_binned
    
    def get_stats(self,):        
        client=client_get(scheduler_info=self.scheduler_info)
        if self.kappa_class.do_pseudo_cl:
            self.outp['cl_b_stats']={}
            cl_b={}
            self.cl0_b['shear_B']=self.cl0_b[corr_ll]*0
            cl0_b=np.hstack([self.cl0_b[corr] for corr in self.corr_order])
            for im in self.Master_algs:
                cl_b[im]=self.get_full_samp(self.cl_b[im])
                self.outp['cl_b_stats'][im]=client.compute(delayed(calc_sim_stats)(sim=cl_b[im]['full'],sim_truth=cl0_b))
                
            pcl_b=self.get_full_samp(self.pcl_b)
#             self.cl0_b['shear_B']=self.cl0_b[corr_ll]*0
#             self.cl0_b=np.array([self.cl0_b[corr] for corr in self.corr_order]).flatten()
            self.outp['pcl_b_stats']=client.compute(delayed(calc_sim_stats)(sim=pcl_b['full'],sim_truth=pcl_b['full'].mean(axis=0)))

            for im in self.Master_algs:
                self.outp['cl_b_stats'][im]=self.outp['cl_b_stats'][im].result()
            self.outp['pcl_b_stats']=self.outp['pcl_b_stats'].result()

            self.outp['cl_b']=self.cl_b
            self.outp['pcl_b']=self.pcl_b
        if self.do_xi:
            xi_b=self.get_full_samp(self.xi_b)
            
            im='xi_imaster'
            cl_b[im]=self.get_full_samp(self.cl_b[im])
            self.cl0_b['shear_m']=self.cl0_b[corr_ll]
            cl0_b=[]
            for corr in self.kappa_class.corrs:
                cl0_b+=[self.cl0_b[corr]]
                if corr==corr_ll:
                    cl0_b+=[self.cl0_b[corr]]
            
            cl0_b=np.hstack(cl0_b)
            self.outp['cl_b_stats'][im]=client.compute(delayed(calc_sim_stats)(sim=cl_b[im]['full'],sim_truth=cl0_b))

            self.outp['xi_b_stats']=client.compute(delayed(calc_sim_stats)(sim=xi_b['full'],sim_truth=xi_b['full'].mean(axis=0)))
            self.outp['xi_b_stats']=self.outp['xi_b_stats'].result()
            self.outp['cl_b_stats'][im]=self.outp['cl_b_stats'][im].result()
            self.outp['xi0']=self.xi_L0
            self.outp['xiW0']=self.xiW_L
            self.outp['xi_b']=self.xi_b
            self.outp['theta_bins']=self.kappa_class.theta_bins
            th_bins=self.outp['theta_bins']
            self.outp['thb']=0.5*(th_bins[1:]+th_bins[:-1])
            self.outp['xi_window_norm']=self.xi_window_norm
        return
    
    def generate_maps(self,):
        client=client_get(scheduler_info=self.scheduler_info)
        SJ=client.scatter(self,broadcast=True)
        step= self.nworkers*self.njobs_submit_per_worker # min(nsim,len(client.scheduler_info()['workers']))
        
        i=0
        j=0
        futures=[delayed(get_clsim)(SJ,i) for i in np.arange(self.nsim)]
        futures_done=[]
        while j<self.nsim:
            futures_j=client.compute(futures[j:j+step])
            wait_futures(futures_j)
            futures_done+=futures_j
            j+=step
        del futures
        
        if self.kappa_class.do_pseudo_cl:    
            self.cl_b={im: {'full':np.zeros(self.sim_clb_shape,dtype='float32')} for im in self.Master_algs}
            for im in self.Master_algs:
                self.cl_b[im].update({jks:{} for jks in self.jk_stat_keys}) 

            self.pcl_b={'full':np.zeros(self.sim_clb_shape,dtype='float32')}
            self.pcl_b.update({jks:{} for jks in self.jk_stat_keys})
        if self.do_xi:    
            self.xi_b={'full':np.zeros(self.sim_xib_shape,dtype='float32')}   #  {im:np.zeros(sim_clb_shape,dtype='float32') for im in Master_algs}}
            self.xi_b.update({jks:{} for jks in self.jk_stat_keys})  #{im:{} for im in Master_algs} for jks in jk_stat_keys})
            im='xi_imaster'
            self.cl_b[im]= {'full':np.zeros(self.sim_clb_shape,dtype='float32')}
            self.cl_b[im].update({jks:{} for jks in self.jk_stat_keys}) 

        for i in np.arange(self.nsim):
            tt=futures_done[i].result()
            if self.kappa_class.do_pseudo_cl:
                self.pcl_b[i]=tt[0]
                for k in self.Master_algs:
                    self.cl_b[k][i]=tt[1][k]
            if self.do_xi:
                self.xi_b[i]=tt[2]
                k='xi_imaster'
                self.cl_b[k][i]=tt[1][k]

            client.cancel(futures_done[i])
        proc = psutil.Process()
        print('done map ',i, thread_count(),'mem, peak mem: ',format_bytes(proc.memory_info().rss),
             int(getrusage(RUSAGE_SELF).ru_maxrss/1024./1024.)
             )
#         del futures_done
        print('done map ',i, thread_count(),'mem, peak mem: ',format_bytes(proc.memory_info().rss),
             int(getrusage(RUSAGE_SELF).ru_maxrss/1024./1024.)
             )
        j+=step

    print('done generating maps')

    def get_full_samp(self,cljk={}):
        k=None
        try:
            k=cljk['full'].keys()
        except:
            pass
        for i in np.arange(self.nsim):            
            if k is None:
                cljk['full'][i,:]=cljk[i]['full']
                for jks in self.jk_stat_keys:
                    cljk[jks][i]=cljk[i][jks]
            else:
                for ki in k:
                    cljk['full'][ki][i,:]=cljk[i][ki]['full']
                    for jks in self.jk_stat_keys:
                        cljk[jks][ki][i]=cljk[i][ki][jks]
        if k is None:
            for jks in self.jk_stat_keys:
                cljk[jks]=sample_mean(cljk[jks],self.nsim)
        else:
            for ki in k:
                for jks in self.jk_stat_keys:
                    cljk[jks][ki]=sample_mean(cljk[jks][ki],self.nsim)
        return cljk

    
    def comb_maps(self,futures):
        for i in np.arange(self.nsim):
            x=futures[i]#.compute()
            pcl[i,:,:]+=x[0]
            cl[i,:,:]+=x[1]
        return pcl,cl 
    
    def invert_xi(self,xi):
        sim_clb_shape=(self.Nl_bins*(self.ndim+1))
        cl_b=np.zeros(sim_clb_shape,dtype='float32')      
        corrs=self.kappa_class.corrs
        li=0
        lth=0
        for corr in corrs:
            cl_b[li:li+self.Nl_bins]=self.wig_d_binned['wig_d_binned_inv'][corr]@xi[lth:lth+self.n_th_bins]
            li+=self.Nl_bins
            lth+=self.n_th_bins
            if corr==corr_ll:
                cl_b[li:li+self.Nl_bins]=self.wig_d_binned['wig_d_binned_inv']['shear_m']@xi[lth:lth+self.n_th_bins]
                li+=self.Nl_bins
                lth+=self.n_th_bins
        return cl_b
                
    def process_pcli(self,pcli,coupling_M=None):
        if coupling_M is None:
            coupling_M=self.coupling_M
        SN=self.kappa_class.SN
        if self.ndim>1:
            pcli=pcli[[0,1,3,2],:] #2==BB
            corr_orderi=[corr_gg,corr_ll,corr_ggl,'shear_B']
            if self.use_shot_noise:
                pcli[0]-=(np.ones_like(pcli[0])*SN[corr_gg][:,0,0])@coupling_M['coupling_M_N'][corr_gg]*self.use_shot_noise
                pcli[1]-=(np.ones_like(pcli[1])*SN[corr_ll][:,0,0])@coupling_M['coupling_M_N'][corr_ll]*self.use_shot_noise
                pcli[1]-= (np.ones_like(pcli[1])*SN[corr_ll][:,0,0])@coupling_M['coupling_M_N']['shear_B']*self.use_shot_noise #remove B-mode 
                pcli[3]-=(np.ones_like(pcli[1])*SN[corr_ll][:,0,0])@coupling_M['coupling_M_N']['shear_B']*self.use_shot_noise #remove B-mode 
                pcli[3]-=(np.ones_like(pcli[1])*SN[corr_ll][:,0,0])@coupling_M['coupling_M_N'][corr_ll]*self.use_shot_noise

        else:
            pcli-=(np.ones_like(pcli)*self.shot_noise)@coupling_M['coupling_M']
            cli=pcli@coupling_M['coupling_M_inv']

        pcli=pcli[[3,1,2,0],:]#corr_order=['shear_B',corr_ll,corr_ggl,corr_gg] #FIXME: this should be dynamic, based on corr_order and corr_orderi

        sim_clb_shape=(self.Nl_bins*(self.ndim+1))
        pcl_b=np.zeros(sim_clb_shape,dtype='float32')

        cl_b={k:np.zeros(sim_clb_shape,dtype='float32') for k in self.Master_algs}      

        pcl_b=np.zeros(sim_clb_shape,dtype='float32')

        li=0
        corr_order=self.corr_order
        for ii in np.arange(self.ndim+1):
            li=ii*self.Nl_bins
            #cl_b['unbinned'][:,ii]=binning_func(xi=cli[ii],bin_utils=binning_utils)
            pcl_b[li:li+self.Nl_bins]=self.binning_func(xi=pcli[ii],bin_utils=self.binning_utils)
            # pclB_b[:,ii]=binning_func(xi=pcli_B[ii],bin_utils=binning_utils)
            for k in self.Master_algs:
                cl_b[k][li:li+self.Nl_bins]=pcl_b[li:li+self.Nl_bins]@coupling_M['coupling_M_binned_inv'][k][corr_order[ii]] #be careful with ordering as coupling matrix is not symmetric
        return pcl_b,cl_b#,pclB_b,clB_b
    
    def kappa_to_shear_map(self,kappa_map=[]):
        kappa_alm = hp.map2alm(kappa_map,pol=False)        
        gamma_alm = []
        if self.gamma_trans_factor is None:
            self.set_gamma_trans_factor()
        gamma_alm=kappa_alm*self.gamma_trans_factor#[l_alm]
        k_map, g1_map, g2_map = hp.sphtfunc.alm2map( [kappa_alm,gamma_alm,kappa_alm*0 ], nside=nside,pol=True  )
        return g1_map,g2_map
    
    def set_gamma_trans_factor(self,):
        self.gamma_trans_factor=0
        l_t=np.arange(nside*3-1+1)
        self.gamma_trans_factor = np.array([np.sqrt( (li+2)*(li-1)/(li*(li+1))) for li in l_t   ] )
        self.gamma_trans_factor[0] = 0.
        self.gamma_trans_factor[1] = 0.
        l_alm,m_alm=hp.sphtfunc.Alm.getlm(l_t.max())
        l_alm=np.int32(l_alm)
        m_alm=0
        self.gamma_trans_factor=self.gamma_trans_factor[l_alm]
        l_alm=0

    
    def set_SSV_params(self,):
        self.SSV_sigma={}
        self.SSV_cov={}
        self.SSV_kernel={}
        self.SSV_response={}

        if self.add_SSV:
            self.SSV_response0=self.kappa_class.Ang_PS.clz['clsR']
            if add_tidal_SSV:
                self.SSV_response0=self.kappa_class.Ang_PS.clz['clsR']+self.kappa_class.Ang_PS.clz['clsRK']/6.

            for corr in kappa_class.corrs:
                zs1_indx=0
                zs2_indx=0
                self.SSV_kernel[corr]=kappa_class.z_bins[corr[0]][zs1_indx]['kernel_int']
                self.SSV_kernel[corr]=self.SSV_kernel[corr]*kappa_class.z_bins[corr[1]][zs2_indx]['kernel_int']
                self.SSV_response[corr]=self.SSV_kernel[corr]*self.SSV_response0.T
            self.SSV_sigma=kappa_class.cov_utils.sigma_win_calc(clz=kappa0.Ang_PS.clz,Win=kappa_class.Win,tracers=corr_ll+corr_ll,zs_indx=(0,0,0,0)) 
            self.SSV_cov=np.diag(SSV_sigma**2)


    def get_M_binning_utils(self):
        self.M_binnings=binning()
        self.M_binning_utils={}
        self.Mp_binning_utils={}
        client=client_get(scheduler_info=self.scheduler_info)
        clG=self.kappa_class.cl_tomo() 
        clG0=self.kappa0.cl_tomo() 
        bi=(0,0)
        for corr in self.kappa_class.corrs:

            wt_b=1./client.compute(clG0['cl_b'][corr][bi]).result()
            wt0=client.compute(clG0['cl'][corr][bi]).result()
            self.M_binning_utils[corr]=self.M_binnings.bin_utils(r=self.kappa0.l,r_bins=self.kappa0.l_bins,
                                                        r_dim=2,mat_dims=[1,2],wt_b=wt_b,wt0=wt0)
            wt_b=1./client.compute(clG['pseudo_cl_b'][corr][bi]).result()
            wt0=client.compute(clG['pseudo_cl'][corr][bi]).result()
            self.Mp_binning_utils[corr]=self.M_binnings.bin_utils(r=self.kappa0.l,r_bins=self.kappa0.l_bins,
                                                        r_dim=2,mat_dims=[1,2],wt_b=wt_b,wt0=wt0)
    
    def get_xi_coupling_matrices(self,kappa_class=None): 
        if kappa_class is None:
            kappa_class=self.kappa_class_xib
        corr=kappa_class.corrs
        bi=(0,0)
        s={corr_ll:(2,2),corr_gg:(0,0),corr_ggl:(0,2)}
        wig_d_binned={}
        wig_d_binned_inv={}
        for corr in corrs:
            wig_d_binned[corr]=kappa_class.WT_binned[corr][s[corr]][bi].result()
            wig_d_binned_inv[corr]=kappa_class.inv_WT_binned[corr][s[corr]][bi].result()
#             wig_d_binned_inv[corr]=np.linalg.pinv(wig_d_binned[corr])
            if corr==corr_ll:
                wig_d_binned['shear_m']=kappa_class.WT_binned[corr][(2,-2)][bi].result()
                wig_d_binned_inv['shear_m']=kappa_class.inv_WT_binned[corr][(2,-2)][bi].result()
#                 wig_d_binned_inv['shear_m']=np.linalg.pinv(wig_d_binned['shear_m'])
        outp={}
        outp['wig_d_binned']=wig_d_binned
        outp['wig_d_binned_inv']=wig_d_binned_inv
        self.wig_d_binned=outp
        return outp

        
    def get_coupling_matrices(self,kappa_class=None): 
        if not hasattr(self,'M_binning_utils'):
            self.get_M_binning_utils()
        if kappa_class is None:
            kappa_class=self.kappa_class
        coupling_M={}
        coupling_M_N={}
        coupling_M_N_binned={}
        coupling_M_binned={k:{} for k in self.Master_algs} #{'Master':{},'nMaster':{},'iMaster':{}}    
        coupling_M_inv={}
        coupling_M_binned_inv={k:{}for k in self.Master_algs} #,'nMaster':{},'Master':{}}

        corrs=kappa_class.corrs
        l_bins=kappa_class.l_bins
#         fsky=kappa_class.f_sky[corr_gg][(0,0)]
        dl=l_bins[1:]-l_bins[:-1]
        l=kappa_class.l
        shear_lcut=l>=2
#         nu=(2.*l+1.)*fsky
        for corr in corrs:
            coupling_M[corr]=kappa_class.Win['cl'][corr][(0,0)]['M']
            coupling_M_N[corr]=kappa_class.Win['cl'][corr][(0,0)]['M_noise']
            try:
                coupling_M_N_binned[corr]=kappa_class.binning.bin_2d(cov=coupling_M_N[corr],bin_utils=kappa_class.cl_bin_utils) 
                coupling_M_N_binned[corr]*=dl
            except Exception as err:
                coupling_M_N_binned[corr]=None

            if corr==corr_ll:
                coupling_M['shear_B']=kappa_class.Win['cl'][corr][(0,0)]['M_B']
                coupling_M_N['shear_B']=kappa_class.Win['cl'][corr][(0,0)]['M_B_noise']
                coupling_M_N_binned['shear_B']=kappa_class.binning.bin_2d(cov=coupling_M_N['shear_B'],
                                                                          bin_utils=kappa_class.cl_bin_utils) 
                coupling_M_N_binned['shear_B']*=dl
#                 self.pclN0['shear_B']=self.shot_noise[corr]@coupling_M_N['shear_B']
            if 'Master' in self.Master_algs:
                coupling_M_binned['Master'][corr]=bin_coupling_M(kappa_class,coupling_M[corr])
                if corr==corr_ll:
                    coupling_M_binned['Master']['shear_B']=bin_coupling_M(kappa_class,coupling_M['shear_B'])
            if 'nMaster' in self.Master_algs:
                coupling_M_binned['nMaster'][corr]=kappa_class.binning.bin_2d(cov=coupling_M[corr],bin_utils=kappa_class.cl_bin_utils) 
                coupling_M_binned['nMaster'][corr]*=dl
                if corr==corr_ll:
                    coupling_M_binned['nMaster']['shear_B']=kappa_class.binning.bin_2d(cov=coupling_M['shear_B'],bin_utils=kappa_class.cl_bin_utils)
                    coupling_M_binned['nMaster']['shear_B']*=dl
            if 'iMaster' in self.Master_algs:
                coupling_M_binned['iMaster'][corr]=self.M_binnings.bin_2d_coupling(M=coupling_M[corr].T,bin_utils=self.M_binning_utils[corr],cov=False)
                coupling_M_binned['iMaster'][corr]=coupling_M_binned['iMaster'][corr].T  #to keep the same order in dot product later. Remeber that the coupling matrix is not symmetric.
                if corr==corr_ll:
                    coupling_M_binned['iMaster']['shear_B']=self.M_binnings.bin_2d_coupling(M=coupling_M['shear_B'].T,
                                                                                            bin_utils=self.M_binning_utils[corr],cov=False)
                    coupling_M_binned['iMaster']['shear_B']=coupling_M_binned['iMaster']['shear_B'].T  #to keep the same order in dot product later. Remeber that the coupling matrix is not symmetric.

            cut=l>=0
            if 'shear' in corr:
                cut=shear_lcut 
            coupling_M_inv[corr]=np.zeros_like(coupling_M[corr])
    #             coupling_M_inv[corr][:,cut][cut,:]+=np.linalg.inv(coupling_M[corr][cut,:][:,cut]) #otherwise we get singular matrix since for shear l<2 is not defined.
            try:
                MT=np.linalg.inv(coupling_M[corr][cut,:][:,cut]) #otherwise we get singular matrix since for shear l<2 is not defined.
            except Exception as err:
                print('error in inverting coupling matrix,',err,corr,cut,kappa_class.l,kappa_class.use_binned_l, coupling_M[corr][cut,:][:,cut])
            coupling_M_inv[corr]=np.pad(MT,((~cut).sum(),0),mode='constant',constant_values=0)

            for k in coupling_M_binned.keys():
                coupling_M_binned_inv[k][corr]=np.linalg.inv(coupling_M_binned[k][corr])
                if corr==corr_ll:
                    coupling_M_inv['shear_B']=np.zeros_like(coupling_M['shear_B'])
                    coupling_M_inv['shear_B'][:,cut][cut,:]=np.linalg.inv(coupling_M['shear_B'][cut,:][:,cut]) #otherwise we get singular matrix since for shear l<2 is not defined.
                    coupling_M_binned_inv[k]['shear_B']=np.linalg.inv(coupling_M_binned[k]['shear_B'])

        outp={}
        outp['coupling_M']=coupling_M
        outp['coupling_M_N']=coupling_M_N
        outp['coupling_M_binned']=coupling_M_binned
        outp['coupling_M_N_binned']=coupling_M_N_binned
        outp['coupling_M_inv']=coupling_M_inv
        outp['coupling_M_binned_inv']=coupling_M_binned_inv
        return outp

    def get_coupling_matrices_jk(self,kappa_class=None): 
        self.coupling_M={}
        self.coupling_M['full']=self.get_coupling_matrices(kappa_class=kappa_class)
        for ijk in np.arange(self.njk):
            zs_binjk=copy.deepcopy(kappa_class.z_bins['shear'])
            zl_binjk=copy.deepcopy(kappa_class.z_bins['galaxy'])

            x=self.jkmap==ijk
            for i in np.arange(zs_binjk['n_bins']):
                if self.subsample:
                    zs_binjk[i]['window'][~x]=hp.UNSEEN
                else:
                    zs_binjk[i]['window'][x]=hp.UNSEEN
                zs_binjk[i]['window_alm']=hp.map2alm(zs_binjk[i]['window'])
                zs_binjk[i]['window_cl']=None
            for i in np.arange(zl_binjk['n_bins']):
                if self.subsample:
                    zl_binjk[i]['window'][~x]=hp.UNSEEN
                else:
                    zl_binjk[i]['window'][x]=hp.UNSEEN
                zl_binjk[i]['window_alm']=hp.map2alm(zl_binjk[i]['window'])
                zl_binjk[i]['window_cl']=None
            skylens_kwargs_jk=copy.deepcopy(self.skylens_kwargs)
            skylens_kwargs_jk['shear_zbins']=zs_binjk
            skylens_kwargs_jk['galaxy_zbins']=zl_binjk
            kappa_win_JK=Skylens(**skylens_kwargs_jk)
            kappa_win_JK.gather_data()
            self.coupling_M[ijk]=self.get_coupling_matrices(kappa_class=kappa_win_JK)
            del kappa_win_JK
            del zs_binjk
            del zl_binjk
            print('coupling M jk ',ijk,'done')
        return 
    
def get_clsim(Sim_JK,i):
    self=Sim_JK
    mapi=i*1.
    print('doing map: ',i)
    local_state = np.random.RandomState(self.seed+i)
    cl0i=copy.deepcopy(self.cl0)

    if self.add_SSV:
        SSV_delta=np.random.multivariate_normal(mean=self.SSV_sigma*0,cov=self.SSV_cov,size=1)[0]
        # print('adding SSV')
        SSV_delta2=SSV_delta*self.kappa_class.Ang_PS.clz['dchi']
#                 print('SSV delta shape',SSV_delta2.shape,SSV_response[corr_gg].shape)
        tt=self.SSV_response[corr_gg]@SSV_delta2
#                 print('SSV delta shape',SSV_delta2.shape,tt.shape)
        cl0i[0]+=(self.SSV_response[corr_gg]@SSV_delta2)@self.coupling_M[corr_gg]
        cl0i[1]+=(self.SSV_response[corr_ll]@SSV_delta2)@self.coupling_M[corr_ll]
        cl0i[2]+=(self.SSV_response[corr_ggl]@SSV_delta2)@self.coupling_M[corr_ggl]
    if self.lognormal:
        cl_map=hp.synfast(cl0i,nside=self.nside,rng=local_state,new=True,pol=False,verbose=False)
        cl_map_min=np.absolute(cl_map.min(axis=1))
        lmin_match=10
        lmax_match=100
        scale_f=self.lognormal_scale
        v0=np.std(cl_map.T/cl_map_min*scale_f,axis=0)
        cl_map=np.exp(cl_map.T/cl_map_min*scale_f)*np.exp(-0.5*v0**2)-1 #https://arxiv.org/pdf/1306.4157.pdf
        cl_map*=cl_map_min/scale_f
        cl_map=cl_map.T
        cl_map[1,:],cl_map[2,:]=self.kappa_to_shear_map(kappa_map=cl_map[1])#,nside=nside)
    else:
        cl_map=hp.synfast(cl0i,nside=self.nside,new=True,pol=True,verbose=False) #rng=local_state
    
    N_map=0
    if self.use_shot_noise:
        N_map=hp.synfast(self.clN0,nside=self.nside,new=True,pol=True,verbose=False) #rng=local_state
    
    tracers=['galaxy','shear','shear']
    if self.ndim>1:
        for i in np.arange(self.ndim):
            tracer=tracers[i]
            if self.lognormal:
                cl_map[i]-=cl_map[i].mean()

            window2=self.window[tracer]

            if self.add_blending:
                if tracer=='shear':
                    window2=self.window[tracer]+cl_map[i]*blending_coeff

                if tracer=='galaxy':
                    window2=self.window[tracer]+cl_map[i]*fiber_coll_coeff
                    window2+=cl_map[1]*blending_coeff

                window2[window2<0]=0
                window2/=window2[~self.mask[tracer]].mean()
                window2[self.mask[tracer]]=hp.UNSEEN

            cl_map[i]*=window2

            if self.use_shot_noise:
                N_map[i]*=self.window_N[tracer]
                cl_map[i]+=N_map[i]
                N_map[i][self.mask[tracer]]=hp.UNSEEN
            cl_map[i][self.mask[tracer]]=hp.UNSEEN
        del N_map    

        pcli_jk,xi_jk=get_xi_cljk(cl_map,Sim_jk=self)         
        del cl_map
    else:
        cl_map*=self.window
        cl_map[self.mask]=hp.UNSEEN
        pcli_jk=get_cljk(cl_map,lmax=max(l),pol=True)
        pcli_jk['full']=hp.anafast(cl_map, lmax=max(l),pol=True)[l]        

    pcl_b_jk={};cl_b_jk={im:{} for im in self.Master_algs}
    if self.kappa_class.do_pseudo_cl:
        for ijk in pcli_jk.keys():
            pcl_b_jk[ijk],cl_b_jki=self.process_pcli(pcli_jk[ijk],coupling_M=self.coupling_M[ijk])
            for im in self.Master_algs:
                cl_b_jk[im][ijk]=cl_b_jki[im]
        pcl_b_jk=jk_mean(pcl_b_jk,njk=self.njk,subsample=self.subsample)
        for im in self.Master_algs:
            cl_b_jk[im]=jk_mean(cl_b_jk[im],njk=self.njk,subsample=self.subsample)
    if self.kappa_class.do_xi:
        cl_b_jk['xi_imaster']={}
        for ijk in pcli_jk.keys():
            cl_b_jk['xi_imaster'][ijk]=self.invert_xi(xi_jk[ijk])
        xi_jk=jk_mean(xi_jk,njk=self.njk,subsample=self.subsample)
    return pcl_b_jk,cl_b_jk,xi_jk


def get_treecorr_cat_args(maps,masks=None,nside=None):
    tree_cat_args={}
    if masks is None:
        masks={}
        for tracer in maps.keys():
            masks[tracer]=maps[tracer]==hp.UNSEEN
    for tracer in maps.keys():
        seen_indices = np.where( ~masks[tracer] )[0]
        theta, phi = hp.pix2ang(nside, seen_indices)
        ra = np.degrees(np.pi*2.-phi)
        dec = -np.degrees(theta-np.pi/2.)
        tree_cat_args[tracer] = {'ra':ra, 'dec':dec, 'ra_units':'degrees', 'dec_units':'degrees'}
    return tree_cat_args


def get_xi_window_norm(window=None,nside=None):
    window_norm={corr:{} for corr in corrs}
    mask={}
    for tracer in window.keys():
        # window[tracer]=kappa_class.z_bins[tracer][0]['window']
        mask[tracer]=window[tracer]==hp.UNSEEN
        # window[tracer]=window[tracer][~mask[tracer]]
    fsky=mask[tracer].mean()
    cat0={'fullsky':np.ones_like(mask)}
    tree_cat_args0=get_treecorr_cat_args(cat0,masks=None,nside=nside)

    tree_cat0=treecorr.Catalog(**tree_cat_args0['fullsky'])
    tree_corrs0=treecorr.NNCorrelation(**corr_config)
    _=tree_corrs0.process(tree_cat0,tree_cat0)
    npairs0=tree_corrs0.npairs*fsky
    del cat0,tree_cat0,tree_corrs0
    
    tree_cat_args=get_treecorr_cat_args(window,masks=mask,nside=nside)
    tree_cat= {tracer: treecorr.Catalog(w=window[tracer][~mask[tracer]], **tree_cat_args[tracer]) for tracer in window.keys()}
    del mask
    for corr in corrs:
        tree_corrs=treecorr.NNCorrelation(**corr_config)
        _=tree_corrs.process(tree_cat[corr[0]],tree_cat[corr[1]])
        window_norm[corr]['weight']=tree_corrs.weight
        window_norm[corr]['npairs']=tree_corrs.npairs 
        window_norm[corr]['npairs0']=npairs0
    del tree_cat,tree_corrs
    return window_norm

def get_xi_window_norm_jk(Sim_jk=None):
    self=Sim_jk
    nside=self.nside
    njk=self.njk
    window_norm={}
    window={}
    for tracer in self.kappa_class.z_bins.keys():
        window[tracer]=self.kappa_class.tracer_utils.z_win[tracer][0]['window']
    window_norm['full']=get_xi_window_norm(window=window,nside=nside)
    
    for ijk in np.arange(njk):
        window_i={}
        x=self.jkmap==ijk
        for tracer in window.keys():
            window_i[tracer]=window[tracer]*1.
            window_i[tracer][x]=hp.UNSEEN
        window_norm[ijk]=get_xi_window_norm(window=window_i,nside=nside)
        del window_i,x
#         gc.collect()
        print('window norm jk ',ijk,' done',)#window_norm[ijk][corr_ll].shape,window_norm[ijk].keys())
    return window_norm

def get_xi(map,window_norm,mask=None,Sim_jk=None):
    self=Sim_jk
    maps={'galaxy':map[0]}
    maps['shear']={0:map[1],1:map[2]}
    if mask is None:
        mask={}
        mask['galaxy']=maps['galaxy']==hp.UNSEEN
        mask['shear']=maps['shear'][0]==hp.UNSEEN
    tree_cat_args=get_treecorr_cat_args(maps,masks=mask,nside=Sim_jk.nside)
    tree_cat={}
    tree_cat['galaxy']=treecorr.Catalog(w=maps['galaxy'][~mask['galaxy']], **tree_cat_args['galaxy']) 
    tree_cat['shear']=treecorr.Catalog(g1=maps['shear'][0][~mask['shear']],g2=maps['shear'][1][~mask['shear']], **tree_cat_args['shear'])
    del mask
    ndim=3 #FIXME
    xi=np.zeros(self.n_th_bins*(self.ndim+1))
    th_i=0
    tree_corrs={}
    n_th_bins=self.n_th_bins
    for corr in self.kappa_class.corrs:#note that in treecorr npairs includes pairs with 0 weights. That affects this calc
        if corr==corr_ggl:
            tree_corrs[corr]=treecorr.NGCorrelation(**corr_config)
            tree_corrs[corr].process(tree_cat['galaxy'],tree_cat['shear'])
            xi[th_i:th_i+n_th_bins]=tree_corrs[corr].xi*tree_corrs[corr].weight/window_norm[corr]['weight']*-1 #sign convention 
                #
            th_i+=self.n_th_bins
        if corr==corr_ll:
            tree_corrs[corr]=treecorr.GGCorrelation(**corr_config)
            tree_corrs[corr].process(tree_cat['shear'])
            xi[th_i:th_i+n_th_bins]=tree_corrs[corr].xip#*tree_corrs[corr].npairs/window_norm[corr]['weight']
            th_i+=n_th_bins
            xi[th_i:th_i+n_th_bins]=tree_corrs[corr].xim#*tree_corrs[corr].npairs/window_norm[corr]['weight']
            th_i+=n_th_bins
        if corr==corr_gg:
            tree_corrs[corr]=treecorr.NNCorrelation(**corr_config)
            tree_corrs[corr].process(tree_cat['galaxy'])
            xi[th_i:th_i+n_th_bins]=tree_corrs[corr].weight/tree_corrs[corr].npairs #window_norm[corr]['weight']  #
#             xi[th_i:th_i+n_th_bins]=tree_corrs[corr].weight/window_norm[corr]
            th_i+=n_th_bins
#     del tree_cat,tree_corrs
#     gc.collect()
    return xi

def corr_matrix(cov_mat=[]): #correlation matrix
    diag=np.diag(cov_mat)
    return cov_mat/np.sqrt(np.outer(diag,diag))

def bin_coupling_M(kappa_class,coupling_M): #following https://arxiv.org/pdf/astro-ph/0105302.pdf 
#construct coupling matrix for the binned c_ell. This assumes that the C_ell within a bin follows powerlaw. 
#Without this assumption we cannot undo the effects of binning
    l=kappa_class.l
    bin_M=kappa_class.cl_bin_utils['binning_mat']
    l2=l*(l+1)
    x=l==0
    l2[x]=1
    Q=bin_M.T*np.pi*2/(l2)
    P=bin_M.T*(l2)/(np.pi*2)
    P=P.T/(kappa_class.l_bins[1:]-kappa_class.l_bins[:-1])
    return P.T@coupling_M@Q.T

def bin_coupling_M2(kappa_class,coupling_M): #following https://arxiv.org/pdf/astro-ph/0105302.pdf 
#construct coupling matrix for the binned c_ell. This assumes that the C_ell within a bin follows powerlaw. 
#Without this assumption we cannot undo the effects of binning
    l=kappa_class.l
    bin_M=kappa_win.cl_bin_utils['binning_mat']
    
    l2=l*(l+1)
    x=l==0
    l2[x]=1
    
    Q=bin_M.T*np.pi*2/(l2)**2
    P=bin_M.T*(l2)**2/(np.pi*2)
    P=P.T/(kappa_class.l_bins[1:]-kappa_class.l_bins[:-1])
    return P.T@coupling_M@Q.T

seed=12334
def get_clsim2(cl0,window,mask,kappa_class,coupling_M,coupling_M_inv,ndim,i):
    print(i)
    local_state = np.random.RandomState(seed+i)
    cl_map=hp.synfast(cl0,nside=nside,rng=local_state,new=True,pol=True,verbose=False)

    if ndim>1:
        cl_map[0]*=window['galaxy']
        cl_map[0][mask['galaxy']]=hp.UNSEEN
        cl_map[1]*=window['shear'] #shear_1
        cl_map[2]*=window['shear']#shear_2
        cl_map[1][mask['shear']]=hp.UNSEEN
        cl_map[2][mask['shear']]=hp.UNSEEN
        pcli=hp.anafast(cl_map, lmax=max(l),pol=True) #TT, EE, BB, TE, EB, TB for polarized input map
        pcli=pcli[:,l]
        pcli=pcli[[0,1,3],:]
#             for i in np.arange(6):

    else:
        cl_map*=window
        cl_map[mask]=hp.UNSEEN
        pcli=hp.anafast(cl_map, lmax=max(l),pol=True)[l]
        
    del cl_map

    if ndim>1:
        pcli[0]-=(np.ones_like(pcli[0])*kappa_class.SN[corr_gg][:,0,0])@coupling_M[corr_gg]*use_shot_noise
        pcli[1]-=(np.ones_like(pcli[1])*kappa_class.SN[corr_ll][:,0,0])@coupling_M[corr_ll]*use_shot_noise
        pcli[1]-=(np.ones_like(pcli[1])*kappa_class.SN[corr_ll][:,0,0])@coupling_M['shear_B']*use_shot_noise #remove B-mode leakage

        cli=[pcli[0]@coupling_M_inv[corr_gg],
              pcli[1]@coupling_M_inv[corr_ll],
              pcli[2]@coupling_M_inv[corr_ggl]]
    else:
        pcli-=(np.ones_like(pcli)*shot_noise)@coupling_M
        cli=pcli@coupling_M_inv
    cli=np.array(cli)
    return [pcli.T,cli.T]


def get_xi_cljk(cl_map,Sim_jk=None):
    self=Sim_jk
    pcl_jk={}
    xi_jk={}
    pol=True
    lmax=max(self.kappa_class.l)
    if self.kappa_class.do_pseudo_cl:
        pcl_jk['full']=hp.anafast(cl_map, lmax=lmax,pol=True) #TT, EE, BB, TE, EB, TB for polarized input map
    if self.kappa_class.do_xi:
        xi_jk['full']=get_xi(cl_map, window_norm=self.xi_window_norm['full'],Sim_jk=Sim_jk)
    for ijk in np.arange(self.njk):
        x=self.jkmap==ijk
        cl_map_i=copy.deepcopy(cl_map)
        if self.subsample:
            cl_map_i[:,~x]=hp.UNSEEN
        else:
            cl_map_i[:,x]=hp.UNSEEN

        if self.kappa_class.do_pseudo_cl:
            pcl_jk[ijk]=hp.anafast(cl_map_i,lmax=lmax,pol=pol)
        if self.kappa_class.do_xi:
            xi_jk[ijk]=get_xi(cl_map_i, window_norm=self.xi_window_norm[ijk],Sim_jk=Sim_jk)
        del cl_map_i
        gc.collect()
    return pcl_jk,xi_jk

def jk_mean(p={},njk=0,subsample=False):
    if njk==0:
        return p
    p2={}
    nn=np.arange(njk)
    for i in nn: #p.keys():
        #if i in nn:
        p2[i]=p[i]
    jk_vals=np.array(list(p2.values()))
    mean=np.mean(jk_vals,axis=0)
    var=np.var(jk_vals,axis=0,ddof=0)
    try:
        cov=np.cov(jk_vals,rowvar=0)
    except Exception as err:
#         print(jk_vals.shape, err)
        cov=np.cov(jk_vals.reshape(njk, jk_vals.shape[2]*jk_vals.shape[1]),rowvar=0)
    if subsample:
        var/=njk
        cov/=njk
    else:
        cov*=(njk-1)
    p['jk_mean']=mean
    p['jk_err']=np.sqrt(var) 
    p['jk_cov']=cov
    p['jk_corr']=corr_matrix(cov_mat=cov)
    return p

def sample_mean(p={},nsamp=0):
#     if check_empty(p):
#         print ('sample-mean: got empty dict')
#         return p
    p2={}
    nn=np.arange(nsamp)
    for i in nn: #p.keys():
        #if i in nn:
        p2[i]=p[i]
    jk_vals=np.array(list(p2.values()))
    mean=np.mean(jk_vals,axis=0)
    #print mean
    var=np.var(jk_vals,axis=0,ddof=0)
    p['mean']=mean
    p['err']=np.sqrt(var)
    try:
        cov=np.cov(jk_vals,rowvar=0)
        p['cov']=cov
        p['corr']=corr_matrix(cov_mat=cov)
    except Exception as err:
        p['cov']=err
        p['corr']=err
    return p



def calc_sim_stats(sim=[],sim_truth=[],PC=False):
    sim_stats={}
    sim_stats['std']=np.std(sim,axis=0)    
    sim_stats['mean']=np.mean(sim,axis=0)
    sim_stats['median']=np.median(sim,axis=0)
    sim_stats['percentile']=np.percentile(sim,[16,84],axis=0)
    sim_stats['cov']=np.cov(sim,rowvar=0)
    
    sim_stats['percentile_score']=np.zeros_like(sim_stats['std'])
    if len(sim_stats['std'].shape)==1:
        for i in np.arange(len(sim_stats['std'])):
            sim_stats['percentile_score'][i]=percentileofscore(sim[:,i],sim_truth[i])
    elif len(sim_stats['std'].shape)==2:
        for i in np.arange(len(sim_stats['std'])):
            for i_dim in np.arange(2):
                for j_dim in np.arange(2):
                    sim_stats['percentile_score'][i][i_dim,j_dim]=percentileofscore(sim[:,i,i_dim,j_dim],
                                                                                   sim_truth[i,i_dim,j_dim])
    else:
        sim_stats['percentile_score']='not implemented for ndim>2'
    return sim_stats