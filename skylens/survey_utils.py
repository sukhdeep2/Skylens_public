"""
A collection of functions that are used to generate mock survey data (galaxy, shear and CMB samples).
Functions for LSST, DESI, DES, KiDS and CMB surveys.
"""

from scipy.stats import norm as gaussian
import copy
import numpy as np
#from lensing_utils import *
from skylens.tracer_utils import *
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
cosmo_h_PL=cosmo.clone(H0=100)
from dask.distributed import Client,get_client
import healpy as hp
import sys
sys.path.append('./ForQuE/')
#from cmb import *
#from cmb_lensing_rec import *

cosmo_h=cosmo.clone(H0=100)

d2r=np.pi/180.

nz_F=3600./d2r**2 #convert nz from arcmin^2 to rd ^2
def galaxy_shot_noise_calc(zg1=None,zg2=None):
    SN=np.sum(zg1['W']*zg2['W']*zg1['nz']*nz_F) #FIXME: Check this
    #Assumption: ns(z)=ns*pzg*dzg
    SN/=np.sum(zg1['nz']*nz_F*zg1['W'])
    SN/=np.sum(zg2['nz']*nz_F*zg2['W'])
    return SN

def shear_shape_noise_calc(zs1=None,zs2=None,sigma_gamma=0):
    SN0=sigma_gamma**2#/(zs1['ns']*nz_F)

    SN=SN0*np.sum(zs1['W']*zs2['W']*zs1['nz']*nz_F) #FIXME: this is probably wrong.
    #Assumption: ns(z)=ns*pzs*dzs
    SN/=np.sum(zs1['nz']*nz_F*zs1['W'])
    SN/=np.sum(zs2['nz']*nz_F*zs2['W'])
#     print(zs1['ns'])
    return SN

def lsst_pz_source(alpha=2,z0=0.11,beta=0.68,z=[]): #alpha=1.24,z0=0.51,beta=1.01,z=[]
    p_zs=z**alpha*np.exp(-(z/z0)**beta)
    p_zs/=np.sum(np.gradient(z)*p_zs) if len(z)>1 else 1
    return p_zs

def ztrue_given_pz_Gaussian(zp=[],p_zp=[],bias=[],sigma=[],zs=None):
    """
        zp: photometrix redshift bins
        p_zp: Probability p(zp) of galaxies in photometric bins
        bias: bias in the true redshift distribution relative to zp.. gaussian will be centered at
                pz+bias
        sigma: scatter in the true redshift distribution relative to zp.. sigma error of the
                gaussian
        zs: True z_source, for which to compute p(z_source)
        ns: Source number density. Needed to return n(zs)
    """
    if zs is None:
        zs=np.linspace(min(zp-sigma*5),max(zp-sigma*5),500)

    y=np.tile(zs, (len(zp),1)  )

    pdf=gaussian.pdf(y.T,loc=zp+bias,scale=sigma).T

    dzp=np.gradient(zp) if len(zp)>1 else 1

    p_zs=np.dot(dzp*p_zp,pdf)

    dzs=np.gradient(zs)
    # print(p_zs,dzp,p_zp,pdf,zp,bias)
    if np.sum(p_zs*dzs)>0:
        p_zs/=np.sum(p_zs*dzs)
    else:
        p_zs[:]=0
    
    return zs,p_zs

def set_window(shear_zbins={},f_sky=0.3,nside=256,mask_start_pix=0,window_cl_fact=None,unit_win=False,scheduler_info=None,mask=None,
              delta_W=True):
    from skylens.skylens_main import Skylens
    w_lmax=3*nside
    l0=np.arange(w_lmax,dtype='int')
    corr=('galaxy','galaxy')
    
    kappa0=Skylens(galaxy_zbins=shear_zbins,do_cov=False,bin_cl=False,l_bins=None,l=l0,use_window=False,
                   corrs=[corr],f_sky=f_sky,scheduler_info=scheduler_info)
    cl0G=kappa0.cl_tomo()
    
    npix0=hp.nside2npix(nside)

    npix=np.int(npix0*f_sky)
    if mask is None:
        mask=np.zeros(npix0,dtype='bool')
        #     mask[int(npix):]=0
        mask[mask_start_pix:mask_start_pix+int(npix)]=1

    cl_map0=hp.ma(np.ones(npix0))
    cl_map0[~mask]=hp.UNSEEN

    if scheduler_info is None:
        client=get_client()
    else:
        client=get_client(address=scheduler_info['address'])
    
    for i in np.arange(shear_zbins['n_bins']):
        cl_i=client.compute(cl0G['cl'][corr][(i,i)]).result()
        if np.any(np.isnan(cl_i)):
            print('survey utils, set_window:',cl_i)
            crash
        if unit_win:
            cl_map=hp.ma(np.ones(12*nside*nside))
#             cl_i=1
        else:            
            cl_i+=shear_zbins['SN']['galaxy'][:,i,i]
            if window_cl_fact is not None:
                cl_i*=window_cl_fact
            cl_map=hp.ma(1+hp.synfast(cl_i,nside=nside))
        cl_map[cl_map<=0]=1.e-4
        cl_map[~mask]=hp.UNSEEN
        cl_t=hp.anafast(cl_map)
#         cl_map/=cl_map[mask].mean()
#         if not unit_win:
#             cl_map/=np.sqrt(cl_t[0]) #this is important for shear map normalization in correlation functions.
        cl_map[~mask]=hp.UNSEEN
        cl_map_noise=np.sqrt(cl_map)
        cl_map_noise[~mask]=hp.UNSEEN
        # cl_map.mask=mask
        shear_zbins[i]['window_cl0']=cl_i
        shear_zbins[i]['window']=cl_map
        if delta_W:
            shear_zbins[i]['window_N']=np.sqrt(shear_zbins[i]['window'])
            shear_zbins[i]['window_N'][~mask]=hp.UNSEEN
        else:
            print('not using delta_W window')
            shear_zbins[i]['window_N']=np.sqrt(1./shear_zbins[i]['window'])
            shear_zbins[i]['window_N'][~mask]=hp.UNSEEN
            shear_zbins[i]['window'][:]=1
            shear_zbins[i]['window'][~mask]=hp.UNSEEN
            
    del cl0G,kappa0
    return shear_zbins


def zbin_pz_norm(shear_zbins={},bin_indx=None,zs=None,p_zs=None,ns=0,bg1=1,AI=0,
                 AI_z=0,mag_fact=0,shear_m_bias=1,k_max=0.3):
    dzs=np.gradient(zs) if len(zs)>1 else 1

    x=np.absolute(p_zs)<1.e-10 #to avoid some numerical errors
    p_zs[x]=0
    
    if np.sum(p_zs*dzs)!=0:
        p_zs=p_zs/np.sum(p_zs*dzs)
    else:
        print('zbin_pz_norm empty bin',bin_indx)
        p_zs*=0
    nz=dzs*p_zs*ns

    i=bin_indx
    x= p_zs>-1 #1.e-10
    if x.sum()==0:
        print('zbin_pz_norm cutting z',zs,p_zs)
    shear_zbins[i]['z']=zs[x]
    shear_zbins[i]['dz']=np.gradient(shear_zbins[i]['z']) if len(shear_zbins[i]['z'])>1 else 1
    shear_zbins[i]['nz']=nz[x]
    shear_zbins[i]['ns']=ns
    shear_zbins[i]['W']=1.
    shear_zbins[i]['pz']=p_zs[x]*shear_zbins[i]['W']
    shear_zbins[i]['pzdz']=shear_zbins[i]['pz']*shear_zbins[i]['dz']
    shear_zbins[i]['Norm']=np.sum(shear_zbins[i]['pzdz'])
    shear_zbins[i]['b1']=bg1
    shear_zbins[i]['bz1']=bg1*np.ones_like(shear_zbins[i]['pz'])
    shear_zbins[i]['AI']=AI
    shear_zbins[i]['AI_z']=AI_z
    shear_zbins[i]['mag_fact']=mag_fact
    shear_zbins[i]['shear_m_bias']=shear_m_bias #default value is 1. This is really 1+m
    zm=np.sum(shear_zbins[i]['z']*shear_zbins[i]['pzdz'])/shear_zbins[i]['Norm']
    shear_zbins[i]['lm']=k_max*cosmo_h.comoving_transverse_distance(zm).value
    shear_zbins[i]['k_max']=k_max
    return shear_zbins


def source_tomo_bins(zp=None,p_zp=None,nz_bins=None,ns=26,ztrue_func=None,zp_bias=None,scheduler_info=None,
                    zp_sigma=None,zs=None,n_zs=100,z_bins=None,f_sky=0.3,nside=256,use_window=False,z_true_max=5,
                    mask_start_pix=0,window_cl_fact=None,bg1=1,AI=0,AI_z=0,shear_m_bias=1,l=None,mag_fact=0,
                     sigma_gamma=0.26,k_max=0.3,unit_win=False,use_shot_noise=True,delta_W=True,**kwargs):
    """
        Setting source redshift bins in the format used in code.
        Need
        zs (array): redshift bins for every source bin. if z_bins is none, then dictionary with
                    with values for each bin
        p_zs: redshift distribution. same format as zs
        z_bins: if zs and p_zs are for whole survey, then bins to divide the sample. If
                tomography is based on lens redshift, then this arrays contains those redshifts.
        ns: The number density for each bin to compute shape noise.
    """
    shear_zbins={}

    if nz_bins is None:
        nz_bins=1

    if z_bins is None:
        z_bins=np.linspace(min(zp)-0.0001,max(zp)+0.0001,nz_bins+1)

    if zs is None:
        sigma_max=0
        if zp_sigma is not None:
            sigma_max=max(np.atleast_1d(zp_sigma))*5
#         zs=np.linspace( max(0.01,min(zp)-sigma_max), max(zp)+sigma_max,len(zp)+2)#n_zs)
        print(zp,n_zs,z_true_max)
        zs=np.linspace(min(zp),z_true_max,n_zs)
        zs=zs[zs<=z_true_max]
    print('source_tomo_bins, zmax',zs.max(),max(zp),sigma_max)
    dzs=np.gradient(zs)
    dzp=np.gradient(zp) if len(zp)>1 else [1]
    zp=np.array(zp)

    zl_kernel=np.linspace(0,max(zs),50)
    lu=Tracer_utils()
    cosmo_h=cosmo_h_PL

    zmax=max(z_bins)

    l=[1] if l is None else l
    shear_zbins['SN']={}
    shear_zbins['SN']['galaxy']=np.zeros((len(l),nz_bins,nz_bins))
    shear_zbins['SN']['shear']=np.zeros((len(l),nz_bins,nz_bins))
    shear_zbins['SN']['kappa']=np.zeros((len(l),nz_bins,nz_bins))

    pop_keys=[]

    for i in np.arange(nz_bins):
        shear_zbins[i]={}
        indx=zp.searchsorted(z_bins[i:i+2])

        if ztrue_func is None:
            if indx[0]==indx[1]:
                indx[1]=-1
            zs=zp[indx[0]:indx[1]]
            p_zs=p_zp[indx[0]:indx[1]]
            nz=ns*p_zs*dzp[indx[0]:indx[1]]
            ns_i=nz.sum()
        else:
            ns_i=ns*np.sum(p_zp[indx[0]:indx[1]]*dzp[indx[0]:indx[1]])
            zs,p_zs=ztrue_func(zp=zp[indx[0]:indx[1]],p_zp=p_zp[indx[0]:indx[1]],
                            bias=zp_bias[indx[0]:indx[1]],
                            sigma=zp_sigma[indx[0]:indx[1]],zs=zs)

        shear_zbins=zbin_pz_norm(shear_zbins=shear_zbins,bin_indx=i,zs=zs,p_zs=p_zs,ns=ns_i,bg1=bg1,
                             AI=AI,AI_z=AI_z,mag_fact=mag_fact,shear_m_bias=shear_m_bias,
                             k_max=k_max)
        #sc=1./lu.sigma_crit(zl=zl_kernel,zs=zs[x],cosmo_h=cosmo_h)
        #shear_zbins[i]['lens_kernel']=np.dot(shear_zbins[i]['pzdz'],sc)

#         print(zmax,shear_zbins[i]['z'])
        if shear_zbins[i]['z'].size>0:
            zmax=max([zmax,max(shear_zbins[i]['z'])])
        else:
            pop_keys+=[i]
        if shear_zbins[i]['Norm']==0:
            print('empty bin',i)
            pop_keys+=[i]
        
    if len(pop_keys)>0:
        print('some bad bins', pop_keys)
#         crash
    #     ib=0
    #     print('deleting bins', pop_keys)
    #     for i in np.arange(nz_bins):
    #         if i in pop_keys:
    #             continue
    #         shear_zbins[ib]=copy.deepcopy(shear_zbins[i])
    #         ib+=1
        
    #     for i in np.arange(ib,nz_bins):
    #         del shear_zbins[i]
    #     nz_bins-=len(pop_keys)

    for i in np.arange(nz_bins):
        if use_shot_noise:
            shear_zbins['SN']['galaxy'][:,i,i]=galaxy_shot_noise_calc(zg1=shear_zbins[i],zg2=shear_zbins[i])
            shear_zbins['SN']['shear'][:,i,i]=shear_shape_noise_calc(zs1=shear_zbins[i],zs2=shear_zbins[i],
                                                                 sigma_gamma=sigma_gamma)
            shear_zbins['SN']['kappa'][:,i,i]=shear_shape_noise_calc(zs1=shear_zbins[i],zs2=shear_zbins[i],
                                                                 sigma_gamma=sigma_gamma) #FIXME: This is almost certainly not correct


    shear_zbins['n_bins']=nz_bins #easy to remember the counts
#     shear_zbins['z_lens_kernel']=zl_kernel
    shear_zbins['zmax']=zmax
    shear_zbins['zp']=zp
    shear_zbins['zs']=zs
    shear_zbins['pz']=p_zp
    shear_zbins['z_bins']=z_bins
    shear_zbins['zp_sigma']=zp_sigma
    shear_zbins['zp_bias']=zp_bias
    shear_zbins['bias_func']='constant_bias'
    if use_window:
        shear_zbins=set_window(shear_zbins=shear_zbins,f_sky=f_sky,nside=nside,mask_start_pix=mask_start_pix,
                           window_cl_fact=window_cl_fact,unit_win=unit_win,scheduler_info=scheduler_info,delta_W=delta_W)
    return shear_zbins

def lens_wt_tomo_bins(zp=None,p_zp=None,nz_bins=None,ns=26,ztrue_func=None,zp_bias=None,
                        zp_sigma=None,cosmo_h=None,z_bins=None,scheduler_info=None):
    """
        Setting source redshift bins in the format used in code.
        Need
        zs (array): redshift bins for every source bin. if z_bins is none, then dictionary with
                    with values for each bin
        p_zs: redshift distribution. same format as zs
        z_bins: if zs and p_zs are for whole survey, then bins to divide the sample. If
                tomography is based on lens redshift, then this arrays contains those redshifts.
        ns: The number density for each bin to compute shape noise.
    """
    if nz_bins is None:
        nz_bins=1

    z_bins=np.linspace(min(zp),max(zp)*0.9,nz_bins) if z_bins is None else z_bins

    shear_zbins0=source_tomo_bins(zp=zp,p_zp=p_zp,zp_bias=zp_bias,zp_sigma=zp_sigma,ns=ns,nz_bins=1,scheduler_info=scheduler_info)
    lu=tracer_utils()

    if cosmo_h is None:
        cosmo_h=cosmo_h_PL

    zs=shear_zbins0[0]['z']
    p_zs=shear_zbins0[0]['z']
    dzs=shear_zbins0[0]['dz']
    shear_zbins={}

    zl=np.linspace(0,2,50)
    sc=1./lu.sigma_crit(zl=zl,zs=zs,cosmo_h=cosmo_h)
    scW=1./np.sum(sc,axis=1)

    for i in np.arange(nz_bins):
        i=np.int(i)
        shear_zbins[i]=copy.deepcopy(shear_zbins0[0])
        shear_zbins[i]['W']=1./lu.sigma_crit(zl=z_bins[i],zs=zs,cosmo_h=cosmo_h)
        shear_zbins[i]['W']*=scW
        shear_zbins[i]['pz']*=shear_zbins[i]['W']

        x= shear_zbins[i]['pz']>-1 #1.e-10 # FIXME: for shape noise we check equality of 2 z arrays. Thats leads to null shape noise when cross the bins in covariance
        for v in ['z','pz','dz','W','nz']:
            shear_zbins[i][v]=shear_zbins[i][v][x]

        shear_zbins[i]['pzdz']=shear_zbins[i]['pz']*shear_zbins[i]['dz']
        shear_zbins[i]['Norm']=np.sum(shear_zbins[i]['pzdz'])
        # shear_zbins[i]['pz']/=shear_zbins[i]['Norm']
        shear_zbins[i]['lens_kernel']=np.dot(shear_zbins[i]['pzdz'],sc)/shear_zbins[i]['Norm']
    shear_zbins['n_bins']=nz_bins #easy to remember the counts
    shear_zbins['z_lens_kernel']=zl
    shear_zbins['z_bins']=z_bins
    return shear_zbins


def galaxy_tomo_bins(zp=None,p_zp=None,nz_bins=None,ns=10,ztrue_func=None,zp_bias=None,
                     window_cl_fact=None,mag_fact=0,scheduler_info=None,
                    zp_sigma=None,zg=None,f_sky=0.3,nside=256,use_window=False,mask_start_pix=0,
                    l=None,sigma_gamma=0,k_max=0.3,unit_win=True,use_shot_noise=True,delta_W=True):
    """
        Setting source redshift bins in the format used in code.
        Need
        zg (array): redshift bins for every galaxy bin. if z_bins is none, then dictionary with
                    with values for each bin
        p_zs: redshift distribution. same format as zs
        z_bins: if zg and p_zg are for whole survey, then bins to divide the sample. If
                tomography is based on lens redshift, then this arrays contains those redshifts.
        ns: The number density for each bin to compute shot noise.
    """
    galaxy_zbins={}

    if nz_bins is None:
        nz_bins=1
    z_bins=np.linspace(min(zp)-0.0001,max(zp)+0.0001,nz_bins+1)

    zmax=max(z_bins)
    
    if zg is None:
        zg=np.linspace(0,1.5,100)
    dzg=np.gradient(zg)
    dzp=np.gradient(zp) if len(zp)>1 else [1]
    zp=np.array(zp)

    zl_kernel=np.linspace(0,2,50)
    lu=tracer_utils()
    cosmo_h=cosmo_h_PL

    for i in np.arange(nz_bins):
        galaxy_zbins[i]={}
        indx=zp.searchsorted(z_bins[i:i+2])

        if ztrue_func is None:
            if indx[0]==indx[1]:
                indx[1]=-1
            zg=zp[indx[0]:indx[1]]
            p_zg=p_zp[indx[0]:indx[1]]
            nz=ns*p_zg*dzp[indx[0]:indx[1]]

        else:
            ns_i=ns*np.sum(p_zp[indx[0]:indx[1]]*dzp[indx[0]:indx[1]])
            zg,p_zg,nz=ztrue_func(zp=zp[indx[0]:indx[1]],p_zp=p_zp[indx[0]:indx[1]],
                            bias=zp_bias[indx[0]:indx[1]],
                            sigma=zp_sigma[indx[0]:indx[1]],zg=zg,ns=ns_i)
        
        x= p_zg>1.e-10
        galaxy_zbins[i]['z']=zg[x]
        galaxy_zbins[i]['dz']=np.gradient(galaxy_zbins[i]['z']) if len(galaxy_zbins[i]['z'])>1 else 1
        galaxy_zbins[i]['nz']=nz[x]
        galaxy_zbins[i]['W']=1.
        galaxy_zbins[i]['pz']=p_zg[x]*galaxy_zbins[i]['W']
        galaxy_zbins[i]['pzdz']=galaxy_zbins[i]['pz']*galaxy_zbins[i]['dz']
        galaxy_zbins[i]['Norm']=np.sum(galaxy_zbins[i]['pzdz'])
        
        zmax=max([zmax,max(galaxy_zbins[i]['z'])])
    galaxy_zbins['zmax']=zmax
    galaxy_zbins['n_bins']=nz_bins #easy to remember the counts
    if use_window:
        galaxy_zbins=set_window(shear_zbins=galaxy_zbins,f_sky=f_sky,nside=nside,mask_start_pix=mask_start_pix,
                           window_cl_fact=window_cl_fact,unit_win=unit_win,scheduler_info=scheduler_info,delta_W=delta_W
                           )
    return galaxy_zbins

z_many=np.linspace(0,4,5000)

def lsst_source_tomo_bins(zmin=0.1,zmax=3,ns0=27,nbins=3,z_sigma=0.03,z_bias=None,z_bins=None,
                          window_cl_fact=None,ztrue_func=ztrue_given_pz_Gaussian,z_true_max=5,z_sigma_power=1,
                          f_sky=0.3,nside=256,zp=None,pzp=None,use_window=False,mask_start_pix=0,
                          l=None,sigma_gamma=0.26,AI=0,AI_z=0,mag_fact=0,k_max=0.3,
                          scheduler_info=None,n_zs=100,delta_W=True,
                          unit_win=False,use_shot_noise=True,**kwargs):

    z=zp
    if zp is None:
        z=z_many[np.where(np.logical_and(z_many>=zmin, z_many<=zmax))]
    if pzp is None:
        pzp=lsst_pz_source(z=z)
    N1=np.sum(pzp)

    x=z>zmin
    x*=z<zmax
    z=z[x]
    pzp=pzp[x]
    ns0=ns0*np.sum(pzp)/N1
    print('ns0: ',ns0)

    if z_bins is None:
        z_bins=np.linspace(zmin, min(2,zmax), nbins+1)
        z_bins[-1]=zmax

    if z_bias is None:
        z_bias=np.zeros_like(z)
    else:
        try:
            zb=interp1d(z_bias['z'],z_bias['b'],bounds_error=False,fill_value=0)
            z_bias=zb(z)
        except:#FIXME: Ugly
            do_nothing=1
    if np.isscalar(z_sigma):
        z_sigma=z_sigma*((1+z)**z_sigma_power)
    else:
        try:
            zs=interp1d(z_sigma['z'],z_sigma['b'],bounds_error=False,fill_value=0)
            z_sigma=zs(z)
        except: #FIXME: Ugly
            do_nothing=1

    return source_tomo_bins(zp=z,p_zp=pzp,ns=ns0,nz_bins=nbins,mag_fact=mag_fact,
                            n_zs=n_zs,delta_W=delta_W,
                         ztrue_func=ztrue_func,zp_bias=z_bias,window_cl_fact=window_cl_fact,
                        zp_sigma=z_sigma,z_bins=z_bins,f_sky=f_sky,nside=nside,z_true_max=z_true_max,
                           use_window=use_window,mask_start_pix=mask_start_pix,k_max=k_max,
                           l=l,sigma_gamma=sigma_gamma,AI=AI,AI_z=AI_z,unit_win=unit_win
                            ,use_shot_noise=use_shot_noise,scheduler_info=scheduler_info,**kwargs)


def DESI_lens_bins(dataset='lrg',nbins=1,window_cl_fact=None,z_bins=None,
                    f_sky=0.3,nside=256,use_window=False,mask_start_pix=0,bg1=1,
                       l=None,sigma_gamma=0,mag_fact=0,scheduler_info=None,delta_W=True,
                    **kwargs):

    home='../data/desi/data/desi/'
    # fname=dataset+'_nz.dat'
    fname='nz_{d}.dat'.format(d=dataset)
#     t=np.genfromtxt(home+fname,names=True,skip_header=3)
    t=np.genfromtxt(home+fname,names=('z_lo','z_hi','pz'))

    x=t['pz']>0
    t=t[x]

    t['pz']/=3600 #from /deg^2 to arcmin^2
    ns=np.sum(t['pz'])
    print('desi,',dataset,' n=',ns)
    z_m=0.5*(t['z_lo']+t['z_hi'])
    dz=t['z_hi']-t['z_lo']
    zmax=max(t['z_hi'])
    zmin=min(t['z_lo'])

    z=z_many[np.where(np.logical_and(z_many>=zmin, z_many<=zmax))]
    pz_t=interp1d(z_m,t['pz'],bounds_error=False,fill_value=0)
    pz=pz_t(z)

    if z_bins is None:
        z_bins=np.linspace(zmin, min(2,zmax), nbins+1)
    print(dataset,zmin,zmax,z_bins)
    return source_tomo_bins(zp=z,p_zp=pz,ns=ns,nz_bins=nbins,mag_fact=mag_fact,
                         ztrue_func=None,zp_bias=0,window_cl_fact=window_cl_fact,
                        zp_sigma=0,z_bins=z_bins,f_sky=f_sky,nside=nside,delta_W=delta_W,
                           use_window=use_window,mask_start_pix=mask_start_pix,bg1=bg1,
                            l=l,sigma_gamma=sigma_gamma,scheduler_info=scheduler_info,**kwargs)


def DES_lens_bins(fname='~/Cloud/Dropbox/DES/2pt_NG_mcal_final_7_11.fits',l=None,sigma_gamma=0,nside=256,mask_start_pix=0,window_cl_fact=0,unit_win=True,
                  use_window=True,f_sky=1,scheduler_info=None):
    z_bins={}
    try:
        t=Table.read(fname,format='fits',hdu=6)
        dz=t['Z_HIGH']-t['Z_LOW']
        zmax=max(t['Z_HIGH'])
    except:
        t=np.genfromtxt(fname,names=('Z_MID','BIN1','BIN2','BIN3','BIN4','BIN5'))
        dz=np.gradient(t['Z_MID'])
        zmax=max(t['Z_MID'])+dz[-1]/2.

    nz_bins=5
    nz=[0.0134,0.0343,0.0505,0.0301,0.0089]
    bz=[1.44,1.70,1.698,1.997,2.058]
    z_bins['SN']={}
    z_bins['SN']['galaxy']=np.zeros((len(l),nz_bins,nz_bins))
    z_bins['SN']['shear']=np.zeros((len(l),nz_bins,nz_bins))


    for i in np.arange(nz_bins):
        z_bins[i]={}
        z_bins[i]['z']=t['Z_MID']
        z_bins[i]['dz']=dz
        z_bins[i]['nz']=nz[i]
        z_bins[i]['b1']=bz[i]
        z_bins[i]['pz']=t['BIN'+str(i+1)]
        z_bins[i]['W']=1.
        z_bins[i]['mag_fact']=0
        z_bins[i]['pzdz']=z_bins[i]['pz']*z_bins[i]['dz']
        z_bins[i]['Norm']=np.sum(z_bins[i]['pzdz'])
        z_bins['SN']['galaxy'][:,i,i]=galaxy_shot_noise_calc(zg1=z_bins[i],zg2=z_bins[i])
        z_bins[i]['lm']=1.e7

    z_bins['n_bins']=nz_bins
    z_bins['nz']=nz
    z_bins['zmax']=zmax
    if use_window:
        z_bins=set_window(shear_zbins=z_bins,f_sky=f_sky,nside=nside,mask_start_pix=mask_start_pix,
                           window_cl_fact=window_cl_fact,unit_win=unit_win,scheduler_info=scheduler_info)
    return z_bins

def DES_bins(fname='~/Cloud/Dropbox/DES/2pt_NG_mcal_final_7_11.fits',l=None,sigma_gamma=0,nside=256,mask_start_pix=0,window_cl_fact=0,unit_win=True,use_window=True,
             f_sky=1,scheduler_info=None):
    z_bins={}
    try:
        t=Table.read(fname,format='fits',hdu=6)
        dz=t['Z_HIGH']-t['Z_LOW']
        zmax=max(t['Z_HIGH'])
    except:
        t=np.genfromtxt(fname,names=('Z_MID','BIN1','BIN2','BIN3','BIN4'))
        dz=np.gradient(t['Z_MID'])
        zmax=max(t['Z_MID'])+dz[-1]/2.

    nz_bins=4
    nz=[1.496,1.5189,1.5949,0.7949]

    z_bins['SN']={}
    z_bins['SN']['galaxy']=np.zeros((len(l),nz_bins,nz_bins))
    z_bins['SN']['shear']=np.zeros((len(l),nz_bins,nz_bins))

    for i in np.arange(nz_bins):
        z_bins[i]={}
        z_bins[i]['z']=t['Z_MID']
        z_bins[i]['dz']=dz
        z_bins[i]['nz']=nz[i]
        z_bins[i]['pz']=t['BIN'+str(i+1)]
        z_bins[i]['W']=1.
        z_bins[i]['AI']=0
        z_bins[i]['AI_z']=0
        z_bins[i]['pzdz']=z_bins[i]['pz']*z_bins[i]['dz']
        z_bins[i]['Norm']=np.sum(z_bins[i]['pzdz'])
        #z_bins['SN']['galaxy'][:,i,i]=galaxy_shot_noise_calc(zg1=z_bins[i],zg2=z_bins[i])
        z_bins['SN']['shear'][:,i,i]=shear_shape_noise_calc(zs1=z_bins[i],zs2=z_bins[i],
                                                            sigma_gamma=sigma_gamma)
        z_bins[i]['lm']=1.e7
        z_bins[i]['b1']=1
        z_bins[i]['AI']=0
        z_bins[i]['AI_z']=0
        z_bins[i]['mag_fact']=0
        z_bins[i]['shear_m_bias']=0
    z_bins['n_bins']=nz_bins
    z_bins['nz']=nz
    z_bins['zmax']=zmax
    if use_window:
        z_bins=set_window(shear_zbins=z_bins,f_sky=f_sky,nside=nside,mask_start_pix=mask_start_pix,
                           window_cl_fact=window_cl_fact,unit_win=unit_win,scheduler_info=scheduler_info)
    return z_bins


def Kids_bins(kids_fname='/home/deep/data/KiDS-450/Nz_DIR/Nz_DIR_Mean/Nz_DIR_z{zl}t{zh}.asc',l=None,sigma_gamma=0):
    zl=[0.1,0.3,0.5,0.7]
    zh=[0.3,0.5,0.7,0.9]
    z_bins={}
    nz_bins=4
    nz=[1.94, 1.59, 1.52, 1.09]
    z_bins['SN']={}
    z_bins['SN']['galaxy']=np.zeros((len(l),nz_bins,nz_bins))
    z_bins['SN']['shear']=np.zeros((len(l),nz_bins,nz_bins))


    for i in np.arange(nz_bins):
        z_bins[i]={}
        t=np.genfromtxt(kids_fname.format(zl=zl[i],zh=zh[i]),names=('z','pz','pz_err'))
        z_bins[i]['z']=t['z']
        z_bins[i]['dz']=0.05
        z_bins[i]['nz']=nz[i]
        z_bins[i]['pz']=t['pz']
        z_bins[i]['W']=1.
        z_bins[i]['pzdz']=z_bins[i]['pz']*z_bins[i]['dz']
        z_bins[i]['Norm']=np.sum(z_bins[i]['pzdz'])

        z_bins['SN']['galaxy'][:,i,i]=galaxy_shot_noise_calc(zg1=z_bins[i],zg2=z_bins[i])
        z_bins['SN']['shear'][:,i,i]=shear_shape_noise_calc(zs1=z_bins[i],zs2=z_bins[i],
                                                            sigma_gamma=sigma_gamma)
        z_bins[i]['lm']=1.e7
    z_bins['n_bins']=nz_bins
    z_bins['nz']=nz
    z_bins['zmax']=max(t['z'])+0.05

    return z_bins

def cmb_bins(zs=1100,l=None):
    shear_zbins={}
    shear_zbins[0]={}

    shear_zbins=zbin_pz_norm(shear_zbins=shear_zbins,bin_indx=0,zs=np.atleast_1d(zs),p_zs=np.atleast_1d(1),
                         ns=0,bg1=1)
    shear_zbins['n_bins']=1 #easy to remember the counts
    shear_zbins['zmax']=[1100]
    shear_zbins['zp_sigma']=0
    shear_zbins['zp_bias']=0
    shear_zbins['nz']=0

    cmb = StageIVCMB(beam=3., noise=1., lMin=30., lMaxT=3.e3, lMaxP=5.e3)
    cmbLensRec = CMBLensRec(cmb, save=False)
    SN=cmbLensRec.fN_k_mv(l)
    shear_zbins['SN']={}
    shear_zbins['SN']['kappa']=SN.reshape(len(SN),1,1)
    shear_zbins['SN']['galaxy']=SN.reshape(len(SN),1,1)*0
    return shear_zbins


def combine_zbins(z_bins1={},z_bins2={}):
    if z_bins1['n_bins']>0:
        z_bins3=copy.deepcopy(z_bins1)
    else:
        z_bins3=copy.deepcopy(z_bins2)
    if z_bins1['n_bins']==0 or z_bins2['n_bins']==0:
        return z_bins3 
    j=0
    for i in np.arange(z_bins1['n_bins'],z_bins1['n_bins']+z_bins2['n_bins']):
#         print(i,z_bins2.keys())
        z_bins3[i]=copy.deepcopy(z_bins2[j])
        j+=1
    z_bins3['n_bins']+=z_bins2['n_bins']
    nl=z_bins2['SN']['shear'].shape[0]
    z_bins3['SN']={}
    z_bins3['SN']['galaxy']=np.zeros((nl,z_bins3['n_bins'],z_bins3['n_bins']))
    z_bins3['SN']['shear']=np.zeros((nl,z_bins3['n_bins'],z_bins3['n_bins']))
    j=0
    for i in np.arange(z_bins1['n_bins']):
        z_bins3['SN']['galaxy'][:,j,j]+=z_bins1['SN']['galaxy'][:,i,i]
        z_bins3['SN']['shear'][:,j,j]+=z_bins1['SN']['shear'][:,i,i]
        j+=1
    for i in np.arange(z_bins2['n_bins']):
        z_bins3['SN']['galaxy'][:,j,j]+=z_bins2['SN']['galaxy'][:,i,i]
        z_bins3['SN']['shear'][:,j,j]+=z_bins2['SN']['shear'][:,i,i]
        j+=1
    
    return z_bins3
