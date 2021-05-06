from scipy.special import jn, jn_zeros,jv
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
from scipy.optimize import fsolve
from skylens.wigner_functions import *
from skylens.utils import *
from scipy.special import jn, jn_zeros,jv
from dask.distributed import Client,get_client
import numpy as np
import itertools
d2r=np.pi/180.

class wigner_transform():
    def __init__(self,theta=[],l=[],s1_s2=[(0,0)],logger=None,ncpu=None,wig_d_taper_order_low=None,
                 wig_d_taper_order_high=None,scheduler_info=None,l_cut_weights=None,**kwargs):
        self.__dict__.update(locals())
        self.name='Wigner'
        self.logger=logger
        self.grad_l=np.gradient(l)
        self.norm=(2*l+1.)/(4.*np.pi) 
        self.wig_norm=self.norm*self.grad_l
        
        self.grad_theta=np.gradient(theta)
        self.inv_norm=np.sin(self.theta)*2*np.pi
        self.inv_wig_norm=self.inv_norm*self.grad_theta

        self.wig_d={}
        self.theta={}
        self.theta_deg={}
        self.s1_s2s=s1_s2
        self.nscatter=0
        self.taper_f=None
#         self.theta=theta
        for (m1,m2) in s1_s2:
            if self.wig_d.get((m1,m2)) is not None:
                continue
            self.wig_d[(m1,m2)]=wigner_d_parallel(m1,m2,theta,self.l,ncpu=ncpu)
            self.theta[(m1,m2)]=theta #FIXME: Ugly
            self.theta_deg[(m1,m2)]=theta/d2r #FIXME: Ugly
            self.wig_d_smoothing(s1_s2=(m1,m2))
            
            self.wig_d[(m2,m1)]=self.wig_d[(m1,m2)]
            self.theta[(m2,m1)]=self.theta[(m1,m2)]
            self.theta_deg[(m2,m1)]=self.theta_deg[(m1,m2)]
        if l_cut_weights is not None:
           self.theta_coupling_mat=self.l_cut_coupling()
        self.scatter_data()
    
    def scatter_data(self):
        self.nscatter+=1
        print('Scattering WT data',self.nscatter)  
        broadcast=True
        client=client_get(scheduler_info=self.scheduler_info)
        keys=['l','grad_l','norm','wig_norm','grad_theta','inv_norm','inv_wig_norm']
        for k in keys:
            if hasattr(self,k):
                self.__dict__[k]=client.scatter(self.__dict__[k],broadcast=broadcast)
#         self.l=client.scatter(self.l,broadcast=broadcast) #FIXME: creates problem with intepolation function
#         self.grad_l=client.scatter(self.grad_l,broadcast=broadcast)
#         self.norm=client.scatter(self.norm,broadcast=broadcast)
#         self.wig_norm=client.scatter(self.wig_norm)
        self.grad_theta_bins=1
        for k in self.wig_d.keys():
            self.wig_d[k]=client.scatter(self.wig_d[k],broadcast=broadcast)
            self.theta[k]=client.scatter(self.theta[k],broadcast=broadcast)
            self.theta_deg[k]=client.scatter(self.theta_deg[k],broadcast=broadcast)
        
    def gather_data(self):
        self.nscatter-=1
        client=client_get(scheduler_info=self.scheduler_info)
#         self.l=client.gather(self.l)
#         self.grad_l=client.gather(self.grad_l)
#         self.norm=client.gather(self.norm)
        for k in self.wig_d.keys():
            self.wig_d[k]=client.gather(self.wig_d[k])
            self.theta[k]=client.gather(self.theta[k])
            self.theta_deg[k]=client.gather(self.theta_deg[k])
        keys=['l','grad_l','norm','wig_norm','grad_theta','inv_norm','inv_wig_norm']
        for k in keys:
            if hasattr(self,k):
                self.__dict__[k]=client.gather(self.__dict__[k])

    def set_binned_theta(self,theta_bins=[]):
        client=client_get(scheduler_info=self.scheduler_info)
        self.theta_bins=client.scatter(theta_bins)
        self.theta_bins_center=client.scatter(0.5*(theta_bins[1:]+theta_bins[:-1]))
        self.grad_theta_bins=client.scatter(theta_bins[1:]-theta_bins[:-1])

    def set_binned_l(self,l_bins=[]):
        client=client_get(scheduler_info=self.scheduler_info)
        self.l_bins=client.scatter(l_bins)
        self.l_bins_center=client.scatter(0.5*(l_bins[1:]+l_bins[:-1]))
        self.grad_l_bins=client.scatter(l_bins[1:]-l_bins[:-1])

    def reset_theta_l(self,theta=None,l=None):
        """
        In case theta ell values are changed. This can happen when we implement the binning scheme.
        """
        self.gather_data()
        if theta is None:
            theta=self.theta
        if l is None:
            l=self.l
        self.__init__(theta=theta,l=l,s1_s2=self.s1_s2s,logger=self.logger)
    
    def wig_d_smoothing(self,s1_s2):
        if self.wig_d_taper_order_low is None or self.wig_d_taper_order_high is None: 
            return
        if self.wig_d_taper_order_low<=0: #try 16, 20
            return
        if self.wig_d_taper_order_high<=0:
            self.wig_d_taper_order_high=self.wig_d_taper_order_low+2
        bessel_order=np.absolute(s1_s2[0]-s1_s2[1])
        zeros=jn_zeros(bessel_order,max(self.wig_d_taper_order_low,self.wig_d_taper_order_high))
        l_max_low=zeros[self.wig_d_taper_order_low-1]/self.theta[s1_s2]
        if l_max_low.max()>self.l.max():
            print('Wigner ell max of ',self.l.max(),' too low for theta_min. Recommendation based on first few zeros of bessel ',s1_s2,' :',zeros[:5]/self.theta[s1_s2].min())
        l_max_high=zeros[self.wig_d_taper_order_high-1]/self.theta[s1_s2]
        if self.wig_d_taper_order_high==0:
            l_max_high[:]=self.l.max()
        l_max_low[l_max_low>self.l.max()]=self.l.max()
        l_max_high[l_max_high>self.l.max()]=self.l.max()
        taper_f=np.cos((self.l[None,:]-l_max_low[:,None])/(l_max_high[:,None]-l_max_low[:,None])*np.pi/2.)
        x=self.l[None,:]>=l_max_low[:,None]
        y=self.l[None,:]>=l_max_high[:,None]
        taper_f[~x]=1
        taper_f[y]=0
        self.wig_d[s1_s2]=self.wig_d[s1_s2]*taper_f
    
    def cl_grid(self,l_cl=[],cl=[],wig_l=None,taper=False,**kwargs):
        """
        Interpolate a given C_ell onto the grid of ells for which WT is intialized. 
        This is to generalize in case user doesnot want to compute C_ell at every ell.
        Also apply tapering if needed.
        """
        if taper:
            self.taper_f=self.taper(l=l_cl,**kwargs)
            taper_f=self.taper_f['taper_f']
            cl=cl*taper_f
            print('taper:',taper_f)
        if np.all(wig_l==l_cl):
            return cl
        cl_int=interp1d(l_cl,cl,bounds_error=False,fill_value=0,
                        kind='linear')
        if wig_l is None:
            wig_l=self.l
        cl2=cl_int(wig_l)
        return cl2

    def cl_cov_grid(self,l_cl=[],cl_cov=[],taper=False,**kwargs):
        """
        Interpolate a given C_ell covariance onto the grid of ells for which WT is intialized. 
        This is to generalize in case user doesnot want to compute C_ell at every ell.
        Also apply tapering if needed.
        """
        if taper:#FIXME there is no check on change in taper_kwargs
            if self.taper_f2 is None or not np.all(np.isclose(self.taper_f['l'],cl)):
                self.taper_f=self.taper(l=l,**kwargs)
                taper_f2=np.outer(self.taper_f['taper_f'],self.taper_f['taper_f'])
                self.taper_f2={'l':l,'taper_f2':taper_f2}
            cl=cl*self.taper_f2['taper_f2']
        if l_cl==[]:#In this case pass a function that takes k with kwargs and outputs cl
            cl2=cl_cov(l=self.l,**kwargs)
        else:
            cl_int=RectBivariateSpline(l_cl,l_cl,cl_cov,)#bounds_error=False,fill_value=0,
                            #kind='linear')
                    #interp2d is slow. Make sure l_cl is on regular grid.
            cl2=cl_int(self.l,self.l)
        return cl2

    def projected_correlation(self,l_cl=[],cl=[],s1_s2=[],wig_l=None,taper=False,wig_d=None,wig_norm=None,**kwargs):
        """
        Get the projected correlation function from given c_ell.
        """
        if wig_d is None: #when using default wigner matrices, interpolate to ensure grids match.
            wig_d=self.wig_d[s1_s2]
            wig_l=self.l
            wig_norm=self.wig_norm
        cl2=self.cl_grid(l_cl=l_cl,cl=cl,taper=taper,wig_l=wig_l,**kwargs)
        w=np.dot(wig_d*wig_norm,cl2)
        return self.theta[s1_s2],w
    
    def inv_projected_correlation(self,theta_xi=[],xi=[],s1_s2=[],wig_theta=None,taper=False,wig_d=None,wig_norm=None,**kwargs):
        """
        Get the projected power spectra (c_ell) from given xi.
        """
        if wig_d is None: #when using default wigner matrices, interpolate to ensure grids match.
            wig_d=self.wig_d[s1_s2].T
            wig_theta=self.theta[s1_s2]
            wig_norm=self.inv_wig_norm
        if wig_theta is None:
            wig_theta=self.theta[s1_s2]
        xi2=self.cl_grid(l_cl=theta_xi,cl=xi,taper=taper,wig_l=wig_theta,**kwargs)
        cl=np.dot(wig_d*wig_norm,xi2)
        return self.l,cl


    def projected_covariance(self,l_cl=[],cl_cov=[],s1_s2=[],s1_s2_cross=None,
                             wig_d1=None,wig_d2=None,wig_norm=None,wig_l=None,
                             grad_l=None,taper=False,**kwargs):
        """
        Turn the power spectra covariance into correlation function covariance. 
        In this function, cl_cov is assumed to be a 1-d vector (the diagonal of the
        power spectra covariance). See projected_covariance2 for the case when 
        cl_cov is a two-dimensional matrix.
        """
        if s1_s2_cross is None:
            s1_s2_cross=s1_s2
        if wig_d1 is None:
            wig_d1=self.wig_d[s1_s2]
            wig_d2=self.wig_d[s1_s2_cross]
            wig_l=self.l
            wig_norm=self.wig_norm
        
        #return self.theta[s1_s2],wig_d1@np.diag(cl_cov)@wig_d2.T
        #when cl_cov can be written as vector, eg. gaussian covariance
        cl2=self.cl_grid(l_cl=l_cl,cl=cl_cov,taper=taper,wig_l=wig_l,**kwargs)
        if grad_l is None:
            grad_l=np.gradient(wig_l)
        cov=(wig_d1*np.sqrt(wig_norm))@np.diag(cl2*grad_l)@(wig_d2*(np.sqrt(wig_norm))).T
#         cov=np.einsum('rk,k,sk->rs',self.wig_d[s1_s2]*np.sqrt(self.norm),cl2*self.grad_l,
#                     self.wig_d[s1_s2_cross]*np.sqrt(self.norm),optimize=True)
        #FIXME: Check normalization
        #FIXME: need to allow user to input wigner matrices.
        return self.theta[s1_s2],cov

    def projected_covariance2(self,l_cl=[],cl_cov=[],s1_s2=[],s1_s2_cross=None,
                              wig_d1=None,wig_d2=None,
                                taper=False,**kwargs):
        if wig_d1 is not None:
            #print(wig_d1.shape,cl_cov.shape)
            return self.theta[s1_s2],wig_d1@cl_cov@wig_d2.T
        #when cl_cov is a 2-d matrix
        if s1_s2_cross is None:
            s1_s2_cross=s1_s2
        cl_cov2=cl_cov  #self.cl_cov_grid(l_cl=l_cl,cl_cov=cl_cov,s1_s2=s1_s2,taper=taper,**kwargs)

        cov=np.einsum('rk,kk,sk->rs',self.wig_d[s1_s2]*np.sqrt(self.norm)*self.grad_l,cl_cov2,
                    self.wig_d[s1_s2_cross]*np.sqrt(self.norm),optimize=True)
#         cov=np.dot(self.wig_d[s1_s2]*self.grad_l*np.sqrt(self.norm),np.dot(self.wig_d[s1_s2_cross]*np.sqrt(self.norm),cl_cov2).T)
        # cov*=self.norm
        #FIXME: Check normalization
        return self.theta[s1_s2],cov

    def taper(self,l=[],large_l_lower=1000,large_l_upper=1500,low_l_lower=10,low_l_upper=50):
        #FIXME there is no check on change in taper_kwargs
        if self.taper_f is None or not np.all(np.isclose(self.taper_f['l'],l)):
            taper_f=np.zeros_like(l,dtype='float64')
            x=l>large_l_lower
            taper_f[x]=np.cos((l[x]-large_l_lower)/(large_l_upper-large_l_lower)*np.pi/2.)
            x=np.logical_and(l<=large_l_lower , l>=low_l_upper)
            taper_f[x]=1
            x=l<low_l_upper
            taper_f[x]=np.cos((l[x]-low_l_upper)/(low_l_upper-low_l_lower)*np.pi/2.)
            
            x=np.logical_or(l<=low_l_lower , l>=large_l_upper)
            taper_f[x]=0
            self.taper_f={'taper_f':taper_f,'l':l}
        return self.taper_f

    def diagonal_err(self,cov=[]):
        return np.sqrt(np.diagonal(cov))

    def skewness(self,l_cl=[],cl1=[],cl2=[],cl3=[],s1_s2=[],taper=False,**kwargs):
        """
        Because we can do 6 point functions as well :). 
        """
        cl1=self.cl_grid(l_cl=l_cl,cl=cl1,s1_s2=s1_s2,taper=taper,**kwargs)
        cl2=self.cl_grid(l_cl=l_cl,cl=cl2,s1_s2=s1_s2,taper=taper,**kwargs)
        cl3=self.cl_grid(l_cl=l_cl,cl=cl3,s1_s2=s1_s2,taper=taper,**kwargs)
        skew=np.einsum('ji,ki,li',self.wig_d[s1_s2],self.wig_d[s1_s2],
                        self.wig_d[s1_s2]*cl1*cl2*cl3)
        skew*=self.norm
        #FIXME: Check normalization
        return self.theta[s1_s2],skew
    
    def l_cut_coupling(self,l_cl=None,theta_out=None,l_cut_weights=None,s1_s2=None,taper=False,**kwargs):
        if l_cut_weights is None:
            l_cl=self.l
            l_cut_weights=self.l_cut_weights
        if s1_s2 is None:
            s1_s2=self.s1_s2s
        theta_coupling={}
        theta_coupling_mat={}
        for (m1,m2) in s1_s2:
            if theta_out is None:
                theta_out=self.theta[(m1,m2)]
            lw=l_cut_weights*1.
            dth=2*np.pi*np.gradient(theta_out)*np.sin(theta_out)
            s=(m1,m2) #(0,0)
#             th,theta_coupling[(m1,m2)]=self.projected_correlation(l_cl=l_cl,cl=lw,taper=taper,s1_s2=s,**kwargs)
            th,theta_coupling_mat[(m1,m2)]=self.projected_covariance(l_cl=l_cl,cl_cov=lw,taper=taper,s1_s2=s,**kwargs)
            theta_coupling_mat[(m1,m2)]*=dth
        return theta_coupling_mat
            
            

def projected_correlation(norm=1,cl=[],wig_d=None):
    """
    Get the projected correlation function from given c_ell.
    """
    w=np.dot(wig_d*norm,cl) #for binned wig_d norm is already applied. Otherwise it is (2*l+1)/4pi * dl
    return w
