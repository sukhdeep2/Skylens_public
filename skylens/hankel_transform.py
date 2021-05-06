#deprecated.

from scipy.special import jn, jn_zeros,jv
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
from scipy.optimize import fsolve
import numpy as np
import itertools
d2r=np.pi/180.

class hankel_transform():
    def __init__(self,theta_min=0.1,theta_max=100,l_max=10,l_min=1.e-4,n_zeros=1000,n_zeros_step=1000,
                 s1_s2=[(0,0)],prune_theta=0,prune_log_space=True,logger=None):
    #FIXME: try to get init in same form as for wigner_transform
        self.name='Hankel'
        self.logger=logger
        self.theta_min=theta_min
        self.theta_max=theta_max
        self.l_max={}
        self.l_min={}
        self.n_zeros=n_zeros
        self.n_zeros_step=n_zeros_step
        self.l={}
        self.theta={}
        self.theta_deg={}
        self.J={}
        self.J_nu1={}
        self.zeros={}
        self.s1_s2s=s1_s2
        if len(s1_s2)>1:
            print('cross covariance not implemented with Hankel Transform. s1_s2s: %s',s1_s2)
        for i in s1_s2:
            self.l[i],self.l_max[i],self.theta[i],self.J[i],self.J_nu1[i],self.zeros[i]=self.get_k_r_j(
                                                   j_nu=np.absolute(i[1]-i[0]),
                                                   n_zeros=n_zeros,theta_min=theta_min,theta_max=theta_max,
                                                   l_max=l_max,l_min=l_min,n_zeros_step=n_zeros_step,
                                                   prune_theta=prune_theta,
                                                   prune_log_space=prune_log_space)
            self.theta_deg[i]=self.theta[i]/d2r #FIXME: Ugly

    def get_k_r_j(self,j_nu=0,n_zeros=1000,theta_min=0.1,theta_max=100,l_max=10,l_min=1.e-4,
                  n_zeros_step=1000,prune_theta=0,prune_log_space=True):
        while True:
            zeros=jn_zeros(j_nu,n_zeros)

            l=zeros/zeros[-1]*l_max
            theta=zeros/l_max
            if min(theta)>theta_min:
                l_max=min(zeros)/theta_min
                print('changed l_max to cover theta_min. j_nu=',l_max,j_nu)
                continue
            elif max(theta)<theta_max:
                n_zeros+=n_zeros_step
                print(' not enough zeros to cover theta_max, increasing by ',j_nu,n_zeros_step,n_zeros)
            elif min(l)>l_min:
                n_zeros+=n_zeros_step
                print(' not enough zeros to cover l_min, increasing by ',j_nu,n_zeros_step,n_zeros)
            else:
                break
        theta_min2=theta[theta<=theta_min][-1]
        theta_max2=theta[theta>=theta_max][0]
        x=theta<=theta_max2
        x*=theta>=theta_min2
        theta=theta[x]
        if prune_theta!=0:
            print('pruning theta, log_space:%s n_f:%s',prune_log_space,prune_theta)
            N=len(theta)
            if prune_log_space:
                idx=np.unique(np.int64(np.logspace(0,np.log10(N-1),np.int32(N/prune_theta))))#pruning can be worse than prune_theta factor due to repeated numbers when logspace number are convereted to int.
                idx=np.append([0],idx)
            else:
                idx=np.arange(0,N-1,step=prune_theta)
            idx=np.append(idx,[N-1])
            theta=theta[idx]
#             self.logger.info ('pruned theta:%s',len(theta))
        theta=np.unique(theta)
#         self.logger.info ('nr:%s',len(theta))
        J=jn(j_nu,np.outer(theta,l))
        J_nu1=jn(j_nu+1,zeros)
        return l,l_max,theta,J,J_nu1,zeros

    def _cl_grid(self,l_cl=[],cl=[],s1_s2=[],taper=False,**kwargs):
        if taper:
            sself.taper_f=self.taper(l=l_cl,**kwargs)
            cl=cl*taper_f
        if l_cl==[]:#In this case pass a function that takes k with kwargs and outputs cl
            cl2=cl(l=self.l[s1_s2],**kwargs)
        else:
            cl_int=interp1d(l_cl,cl,bounds_error=False,fill_value=0,
                            kind='linear')
            cl2=cl_int(self.l[s1_s2])
        return cl2

    def _cl_cov_grid(self,l_cl=[],cl_cov=[],s1_s2=[],taper=False,**kwargs):
        if taper:#FIXME there is no check on change in taper_kwargs
            if self.taper_f2 is None or not np.all(np.isclose(self.taper_f['l'],l_cl)):
                self.taper_f=self.taper(l=l_cl,**kwargs)
                taper_f2=np.outer(self.taper_f['taper_f'],self.taper_f['taper_f'])
                self.taper_f2={'l':l_cl,'taper_f2':taper_f2}
            cl_cov=cl_cov*self.taper_f2['taper_f2']
        if l_cl==[]:#In this case pass a function that takes k with kwargs and outputs cl
            cl2=cl_cov(l=self.l[s1_s2],**kwargs)
        else:
            cl_int=RectBivariateSpline(l_cl,l_cl,cl_cov,)#bounds_error=False,fill_value=0,
                            #kind='linear')
                    #interp2d is slow. Make sure l_cl is on regular grid.
            cl2=cl_int(self.l[s1_s2],self.l[s1_s2])
        return cl2

    def projected_correlation(self,l_cl=[],cl=[],s1_s2=[],taper=False,**kwargs):
        cl2=self._cl_grid(l_cl=l_cl,cl=cl,s1_s2=s1_s2,taper=taper,**kwargs)
        w=np.dot(self.J[s1_s2],cl2/self.J_nu1[s1_s2]**2)
        w*=(2.*self.l_max[s1_s2]**2/self.zeros[s1_s2][-1]**2)/(2*np.pi)
        return self.theta[s1_s2],w

    def spherical_correlation(self,l_cl=[],cl=[],s1_s2=[],taper=False,**kwargs):
    #we will use relation spherical_jn(z)=j{n+0.5}(z)*sqrt(pi/2z)
    #cl will be written as k*cl
        cl2=self._cl_grid(l_cl=l_cl,cl=cl,s1_s2=s1_s2,taper=taper,**kwargs)
        j_f=np.sqrt(np.pi/2./np.outer(self.theta[s1_s2],self.l[s1_s2]))
        w=np.dot(self.J[s1_s2],cl2*self.l[s1_s2]/self.J_nu1[s1_s2]**2)
        w*=(2.*self.l_max[s1_s2]**2/self.zeros[s1_s2][-1]**2)/(2*np.pi)
        return self.theta[s1_s2],w

    def projected_covariance(self,l_cl=[],cl_cov=[],s1_s2=[],taper=False,**kwargs):
        #when cl_cov can be written as vector, eg. gaussian covariance
        cl1=self._cl_grid(l_cl=l_cl,cl=cl_cov,s1_s2=s1_s2,taper=taper,**kwargs)
        # cov=np.dot(self.J[s1_s2],(self.J[s1_s2]*cl1*cl2/self.J_nu1[s1_s2]**2).T)
        cov=np.einsum('rk,k,sk->rs',self.J[s1_s2],cl1/self.J_nu1[s1_s2]**2,
                    self.J[s1_s2],optimize=True)
        cov*=(2.*self.l_max[s1_s2]**2/self.zeros[s1_s2][-1]**2)/(2*np.pi)
        return self.theta[s1_s2],cov

    def projected_covariance2(self,l_cl=[],cl_cov=[],s1_s2=[],taper=False,**kwargs):
        #when cl_cov is a 2-d matrix
        return self.projected_covariance(l_cl=l_cl,cl_cov=np.diag(cl_cov),s1_s2=s1_s2,taper=taper)
        cl_cov2=cl_cov#self._cl_cov_grid(l_cl=l_cl,cl_cov=cl_cov,s1_s2=s1_s2,taper=taper,**kwargs)
        cov=np.dot(self.J[s1_s2],np.dot(self.J[s1_s2]/self.J_nu1[s1_s2]**2,cl_cov2).T)
        cov*=(2.*self.l_max[s1_s2]**2/self.zeros[s1_s2][-1]**2)/(2*np.pi)
        return self.theta[s1_s2],cov

    def taper(self,l=[],large_k_lower=10,large_k_upper=100,low_k_lower=0,low_k_upper=1.e-5):
        #FIXME there is no check on change in taper_kwargs
        if self.taper_f is None or not np.all(np.isclose(self.taper_f['l'],k)):
            taper_f=np.zeros_like(l)
            x=k>large_k_lower
            taper_f[x]=np.cos((k[x]-large_k_lower)/(large_k_upper-large_k_lower)*np.pi/2.)
            x=k<large_k_lower and k>low_k_upper
            taper_f[x]=1
            x=k<low_k_upper
            taper_f[x]=np.cos((k[x]-low_k_upper)/(low_k_upper-low_k_lower)*np.pi/2.)
            self.taper_f={'taper_f':taper_f,'l':k}
        return self.taper_f

    def diagonal_err(self,cov=[]):
        return np.sqrt(np.diagonal(cov))

    def skewness(self,l_cl=[],cl1=[],cl2=[],cl3=[],s1_s2=[],taper=False,**kwargs):
        cl1=self._cl_grid(l_cl=l_cl,cl=cl1,s1_s2=s1_s2,taper=taper,**kwargs)
        cl2=self._cl_grid(l_cl=l_cl,cl=cl2,s1_s2=s1_s2,taper=taper,**kwargs)
        cl3=self._cl_grid(l_cl=l_cl,cl=cl3,s1_s2=s1_s2,taper=taper,**kwargs)
        skew=np.einsum('ji,ki,li',self.J[s1_s2],self.J[s1_s2],
                        self.J[s1_s2]*cl1*cl2*cl3/self.J_nu1[s1_s2]**2)
        skew*=(2.*self.l_max[s1_s2]**2/self.zeros[s1_s2][-1]**2)/(2*np.pi)
        return self.theta[s1_s2],skew


def covariance_brute_force(l=[],theta=[],cl12=[],j_nu=0):
    J=jn(j_nu,np.outer(theta,l))
    dl=np.gradient(l)
    cov=np.dot(J,(J*cl12*l*dl).T)
    cov/=2*np.pi
    return cov
