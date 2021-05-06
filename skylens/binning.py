from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
import numpy as np
import itertools

class binning():
    def __init__(self,rmin=0.1,rmax=100,kmax=10,kmin=1.e-4,n_zeros=1000,n_zeros_step=1000,
                 j_nu=[0],prune_r=0,prune_log_space=True):
        pass

    def bin_utils(self,r=[],r_bins=[],r_dim=2,wt=1,mat_dims=None,wt_b=None,wt0=None):
        bu={}
        bu['bin_center']=0.5*(r_bins[1:]+r_bins[:-1])
        bu['n_bins']=len(r_bins)-1
        bu['bin_indx']=np.digitize(r,r_bins)-1

        bu['wt_b']=wt_b #these two are used for constructing asymmetric binned coupling matrix. see bin_2d_coupling
        bu['wt0']=wt0
        
        binning_mat=np.zeros((len(r),bu['n_bins']))
        for i in np.arange(len(r)):
            if bu['bin_indx'][i]<0 or bu['bin_indx'][i]>=bu['n_bins']:
                continue
            binning_mat[i,bu['bin_indx'][i]]=1.
        bu['binning_mat']=binning_mat

        r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
        dr=np.gradient(r2) #FIXME: can lead to shape errors if r is in r_bins
        r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
        dr=dr[r2_idx]
        bu['r_dr']=r**(r_dim-1)*dr
        bu['r_dr']*=wt
        bu['norm']=np.dot(bu['r_dr'],binning_mat)

        x=np.logical_or(r_bins[1:]<=np.amin(r),r_bins[:-1]>=np.amax(r))
        bu['norm'][x]=np.inf
#         print(bu['norm'])

        if mat_dims is not None:
            bu['r_dr_m']={}
            bu['norm_m']={}
            ls=['i','j','k','l','m']
            for ndim in mat_dims:
                s1=ls[0]
                s2=ls[0]
                r_dr_m=np.copy(bu['r_dr'])
                norm_m=np.copy(bu['norm'])
                for i in np.arange(ndim-1):
                    s1=s2+','+ls[i+1]
                    s2+=ls[i+1]
                    r_dr_m=np.einsum(s1+'->'+s2,r_dr_m,bu['r_dr'])#works ok for 2-d case
                    norm_m=np.einsum(s1+'->'+s2,norm_m,bu['norm'])#works ok for 2-d case
                bu['r_dr_m'][ndim]=r_dr_m
                bu['norm_m'][ndim]=norm_m
        return bu

    def bin_1d(self,xi=[],bin_utils=None):
        xi_b=np.dot(xi*bin_utils['r_dr'],bin_utils['binning_mat'])
        xi_b/=bin_utils['norm']
        return xi_b

    def bin_2d(self,cov=[],bin_utils=None):
        #r_dr=bin_utils['r_dr']
        #cov_r_dr=cov*bin_utils['r_dr_m'][2]#np.outer(r_dr,r_dr)
        binning_mat=bin_utils['binning_mat']
        cov_b=np.dot(binning_mat.T, np.dot(cov*bin_utils['r_dr_m'][2],binning_mat) )
        cov_b/=bin_utils['norm_m'][2]
        return cov_b
    
    def bin_2d_coupling(self,M=[],bin_utils=None,wt_b=None,wt0=None,partial_bin_side=None,lm=0,lm_step=-1,cov=False): #asymmetric binning
        ndim=1
        if cov:
            ndim=2
        binning_mat=bin_utils['binning_mat']
        if wt_b is None:
            wt_b=bin_utils['wt_b']
        if wt0 is None:
            wt0=bin_utils['wt0']
        print('bin_2d_coupling: ',binning_mat.shape,wt_b.shape,wt0.shape)
        if len(wt0.shape)==1:
            binning_mat2=wt0[:,None]*binning_mat*wt_b
        else:
            binning_mat2=wt0@binning_mat@wt_b #FIXME: Test this.... doesnot work. not used anymore.
        print('bin_2d_coupling again: ',binning_mat.shape,wt_b.shape,wt0.shape)
        rdr=bin_utils['r_dr']
        r_dr_m=bin_utils['r_dr_m'][ndim]
        
        binning_mat=binning_mat*rdr[:,None]/bin_utils['norm'][None,:]
        if partial_bin_side is None:
            cov_b=binning_mat.T@M@binning_mat2
        elif partial_bin_side==1:
            cov_b=binning_mat.T@M@binning_mat2[lm:lm+lm_step,:]
        elif partial_bin_side==2:
            cov_b=binning_mat[lm:lm+lm_step,:].T@M@binning_mat2
            
#         cov_b/=bin_utils['norm_m'][1][:,None]
        return cov_b

    def bin_2d_WT(self,wig_mat=[],wig_norm=None,bin_utils_xi=None,bin_utils_cl=None,
                wt_b=None,wt0=None,use_binned_theta=False,win_xi=None):

        wig_mat=wig_mat*wig_norm
        if bin_utils_cl is not None:
            binning_mat_cl=bin_utils_cl['binning_mat']
            if wt_b is None:
                wt_b=bin_utils_cl['wt_b']
            if wt0 is None:
                wt0=bin_utils_cl['wt0']
            if len(wt0.shape)==1:
                binning_mat_cl2=wt0[:,None]*binning_mat_cl*wt_b
            else:
                binning_mat_cl2=wt0@binning_mat_cl@wt_b #FIXME: Test this.
                # rdr=bin_utils_cl['r_dr']
            # binning_mat_cl=binning_mat_cl*rdr[:,None]/bin_utils_cl['norm'][None,:] #cl is inverse binning.
            wm=wig_mat@binning_mat_cl2
        else:
            wm=wig_mat
        
        wig_mat_b=wm
        if bin_utils_xi is not None and use_binned_theta:
            binning_mat_xi=bin_utils_xi['binning_mat']
            
            rdr=bin_utils_xi['r_dr']
            if win_xi is not None:
                rdr=bin_utils_xi['r_dr']*win_xi
            binning_mat_xi=binning_mat_xi*rdr[:,None]/bin_utils_xi['norm'][None,:]

            wig_mat_b=binning_mat_xi.T@wm
        return wig_mat_b

    
    def bin_2d_inv_WT(self,wig_mat=[],wig_norm=None,bin_utils_xi=None,bin_utils_cl=None,
                wt_b=None,wt0=None,use_binned_l=False,win_xi=None):

        wig_mat=wig_mat.T*wig_norm
        if bin_utils_xi is not None:
            binning_mat_xi=bin_utils_xi['binning_mat']
            if wt_b is None:
                wt_b=bin_utils_xi['wt_b']
            if wt0 is None:
                wt0=bin_utils_xi['wt0']
            if len(wt0.shape)==1:
                binning_mat_xi2=wt0[:,None]*binning_mat_xi*wt_b
            else:
                binning_mat_xi2=wt0@binning_mat_xi@wt_b #FIXME: Test this.

            wm=wig_mat@binning_mat_xi2
            
        else:
            wm=wig_mat
        
        wig_mat_b=wm
        if bin_utils_cl is not None and use_binned_l:
            bin_mat_cl=bin_utils_cl['binning_mat']
            
            rdr=bin_utils_cl['r_dr']
            bin_mat_cl=bin_mat_cl*rdr[:,None]/bin_utils_cl['norm'][None,:]
            
            wig_mat_b=bin_mat_cl.T@wm
        return wig_mat_b
    
    def bin_mat(self,r=[],mat=[],r_bins=[],r_dim=2,bin_utils=None):#works for cov and skewness
        ndim=len(mat.shape)
        n_bins=bin_utils['n_bins']
        bin_idx=bin_utils['bin_indx']#np.digitize(r,r_bins)-1
        r_dr=bin_utils['r_dr']
        r_dr_m=bin_utils['r_dr_m'][ndim]

        mat_int=np.zeros([n_bins]*ndim,dtype='float64')
        norm_int=np.zeros([n_bins]*ndim,dtype='float64')

        mat_r_dr=mat*r_dr_m # same as cov_r_dr=cov*np.outer(r_dr,r_dr)
        norm_ijk=bin_utils['norm_m'][ndim]
        for indxs in itertools.product(np.arange(min(bin_idx),n_bins),repeat=ndim):
            x={}#np.zeros_like(mat_r_dr,dtype='bool')
            mat_t=[]
            for nd in np.arange(ndim):
                slc = [slice(None)] * (ndim)
                #x[nd]=bin_idx==indxs[nd]
                slc[nd]=bin_idx==indxs[nd]
                if nd==0:
                    mat_t=mat_r_dr[slc]
                else:
                    mat_t=mat_t[slc]
            mat_int[indxs]=np.sum(mat_t)
        mat_int/=norm_ijk
        return mat_int


def bin_1d(xi=[],bin_utils=None):
    xi_b=np.dot(xi*bin_utils['r_dr'],bin_utils['binning_mat'])
    xi_b/=bin_utils['norm']
    return xi_b


#more basic binning code for testing.
def bin_cov(r=[],cov=[],r_bins=[]):
    bin_center=np.sqrt(r_bins[1:]*r_bins[:-1])
    n_bins=len(bin_center)
    cov_int=np.zeros((n_bins,n_bins),dtype='float64')
    bin_idx=np.digitize(r,r_bins)-1
    r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
    dr=np.gradient(r2)
    r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
    dr=dr[r2_idx]
    r_dr=r*dr
    cov_r_dr=cov*np.outer(r_dr,r_dr)
    for i in np.arange(min(bin_idx+1),n_bins):
        xi=bin_idx==i
        for j in np.arange(min(bin_idx),n_bins):
            xj=bin_idx==j
            norm_ij=np.sum(r_dr[xi])*np.sum(r_dr[xj])
            if i==j:
                print( i,j,norm_ij)
            if norm_ij==0:
                continue
            cov_int[i][j]=np.sum(cov_r_dr[xi,:][:,xj])/norm_ij
    #cov_int=np.nan_to_num(cov_int)
#         print np.diag(cov_r_dr)
    return cov_int
