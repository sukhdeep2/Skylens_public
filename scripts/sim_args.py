import sys, os
# os.environ['OMP_NUM_THREADS'] = '20'
import pickle
from skylens import *
from skylens.survey_utils import *
from skylens.utils import *

from resource import getrusage, RUSAGE_SELF
import psutil
from distributed.utils import format_bytes

#sims args
nsim=1000
seed=12334
njk=0

lognormal_scale=2

nside=1024
do_pseudo_cl=True
do_xi= False #not do_pseudo_cl

if do_xi:
    nside=512

#cl args
lmax_cl=nside#
Nl_bins=25 #40
lmin_cl=0
l0=np.arange(lmin_cl,lmax_cl)

lmin_cl_Bins=lmin_cl+10
lmax_cl_Bins=lmax_cl-10 #this reduces the bias from ell cutoff in p-cl
# l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins))
l_bins=get_l_bins(l_min=lmin_cl_Bins,l_max=lmax_cl_Bins,N_bins=Nl_bins,binning_scheme='log',min_modes=50)
lb=(l_bins[1:]+l_bins[:-1])*.5

l=l0 #np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*20))) #if we want to use fewer ell
bin_cl=True
use_binned_l=False
#window args
use_window=True
window_lmax=np.int32(2*lmax_cl+1) #0
f_sky=0.3
store_win=True
smooth_window=False
wigner_files={}
# wig_home='/global/cscratch1/sd/sukhdeep/dask_temp/'
#wig_home='/Users/Deep/dask_temp/'
home='/hildafs/projects/phy200040p/sukhdeep/physics2/skylens/'
wig_home=home+'temp/'
wigner_files[0]= wig_home+'/dask_wig3j_l1100_w2200_0_reorder.zarr'
wigner_files[2]= wig_home+'/dask_wig3j_l3500_w2100_2_reorder.zarr'
l0w=np.arange(3*nside-1)

use_shot_noise=True
w_smooth_lmax=1.e7 #some large number
window_cl_fact=np.cos(np.pi/2*(l0w/w_smooth_lmax)**10)
x=window_cl_fact<0
x+=l0w>w_smooth_lmax
window_cl_fact[x]=0

mean=150
sigma=50
ww=1000*np.exp(-(l0w-mean)**2/sigma**2)


#cov args
use_cosmo_power=True
do_cov=True
SSV_cov=False
tidal_SSV_cov=False
xi_win_approx=True
#xi args

th_min=hp.nside2resol(nside, arcmin = True)*2/60 #100/nside #1./60
th_max=600./60
n_th_bins=40
theta_bins=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
th=np.logspace(np.log10(th_min*0.98),np.log10(th_max*1.02),n_th_bins*40)
# th2=np.linspace(1,th_max*1.02,n_th_bins*30)
# theta=np.unique(np.sort(np.append(th,th2)))
theta=th
thb=0.5*(theta_bins[1:]+theta_bins[:-1])

bin_xi=True
use_binned_theta=False

corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')
corrs=[corr_ll,corr_ggl,corr_gg]

l0_wT=np.arange(lmax_cl)
WT_kwargs={'l': l0,'theta': theta*d2r,'s1_s2':[(2,2),(2,-2),(0,2),(2,0),(0,0)]}

corr_config = {'min_sep':th_min*60, 'max_sep':th_max*60, 'nbins':n_th_bins, 'sep_units':'arcmin','metric':'Arc','bin_slop':False}#0.01}
# WT_L=None


#zbins
n_source_bins=1
sigma_gamma=0.3944/np.sqrt(2.)  #*2**0.25
n_zs=100
print('getting win')
z0_galaxy=0.5
z0_shear=1
bi=(0,0)

nz_PS=100
z_max_PS=5
