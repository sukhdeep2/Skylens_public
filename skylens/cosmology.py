#TO do:
# 1. Distance in presence of massive netrinos. Currently we are assuming them to be massless. eq. 27 of https://arxiv.org/pdf/1001.4538.pdf
# 2. Use same name convention as Astropy.cosmology so that user can interchange between the two classes.
# 3. implement sigma8 calculation, useful for EH pk normalization.
# 4. change c to speed_of_light everywhere.

import numpy as np
#from galsim.integ import int1d
from scipy.integrate import quad as scipy_int1d
import warnings
from astropy.cosmology import Planck15 as cosmo
cosmo_h=cosmo.clone(H0=100)
cosmo_planck15=cosmo

from astropy.constants import c,G
from astropy import units as u

c=c.to(u.km/u.second)
G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2)
H100=100*(u.km/u.second/u.Mpc)

c_unit=c.unit
G2_unit=G2.unit
H0_unit=H100.unit
c=c.value
G2=G2.value
H100=H100.value


tol=1.e-9

cosmo_fid=dict({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'Ase9':2.2,'mnu':cosmo.m_nu[-1].value,'Ok':cosmo.Ok0,'tau':0.06,'ns':0.965, 'OmR':cosmo.Ogamma0+cosmo.Onu0,
                'Tcmb':cosmo.Tcmb0,'w':-1,'wa':0})

class cosmology():
    def __init__(self,cosmo_params=cosmo_fid,dz=0.005,do_calcs=1,rtol=1.e-4,h_inv=True,use_astropy=True,
                 astropy_cosmo=None,**kwargs):
        self.__dict__.update(locals()) #assign all input args to the class as properties
        self.__dict__.update(cosmo_params)
        self.c=3*10**5
        self.set_z(z_max=cosmo_params['z_max'])
        self.comoving_distance=self.comoving_distance_trapz
        self.efunc=self.E_z
        self.set_cosmology(cosmo_params=cosmo_params)

    def set_z(self,z_max=None):
        self._z_max=z_max
        self._z=np.arange(start=0,stop=z_max+self.dz*2,step=self.dz)
        self._dz=np.gradient(self._z)
        print('cosmology interpolation range',self._z.min(),self._z.max())
        if hasattr(self,'_dc'):
            del self._dc
    
    def set_cosmology(self,cosmo_params=None):
        if cosmo_params is None:
            return
        self.__dict__.update(cosmo_params) #assign all input args to the class as properties
        self.Om=self.Omb+self.Omd
        self.OmL=1-self.Om-self.OmR#FIXME
        self.H0=self.h*100.0
        self.Dh=self.c/self.H0
        if self.Omk!=0:
            self.Dh=self.c*(np.sqrt(np.absolute(self.Omk))) #a0h0=1./sqrt(Omega_k)
        if self.h_inv:
            self.Dh=self.c/100.
#         self.rho=self.Rho_crit()*self.Om
        if self.use_astropy:
            self.set_astropy(cosmo_params=cosmo_params)
        else:
            self.comoving_distance(self._z)
    
    def astropy_cosmo_w0_wa(self,cosmo=None,w0=-1,wa=0):
        attrs=['H0','Om0', 'Ode0','Tcmb0', 'Neff', 'm_nu', 'Ob0']
        args={}
        args['w0']=w0
        args['wa']=wa
        for a in attrs:
            args[a]=getattr(cosmo,a)
        cosmo_w=astropy.cosmology.w0waCDM(**args)
        return cosmo_w

    
    def set_astropy(self,cosmo_params=None,cosmo_h=None):
        if cosmo_params is None or cosmo_params==self.cosmo_params:
            if not self.astropy_cosmo is None:
                return
        if self.astropy_cosmo is None:
            self.astropy_cosmo=cosmo_planck15

        m_nu=self.astropy_cosmo.m_nu.value
        m_nu[-1]=cosmo_params['mnu']
        m_nu*=self.astropy_cosmo.m_nu.unit
        self.astropy_cosmo=self.astropy_cosmo.clone(H0=cosmo_params['h']*100,Ob0=cosmo_params['Omb'],Om0=cosmo_params['Om'],
                                   m_nu=m_nu)#,Ok0=cosmo_params['Omk'])
        if cosmo_params.get('w0') is not None:
            if cosmo_params.get('wa') is None:
                cosmo_params['wa']=0
            if cosmo_params['w0']!=-1 or cosmo_params['wa']!=0:
                self.astropy_cosmo=self.astropy_cosmo_w0_wa(cosmo=self.astropy_cosmo,w0=cosmo_param['w0'],wa=cosmo_param['wa'])
        if self.h_inv:
                self.astropy_cosmo=self.astropy_cosmo.clone(H0=100)
        self.efunc=self.astropy_cosmo.efunc
        self.comoving_distance=self.astropy_comoving_distance
        self.comoving_transverse_distance=self.astropy_comoving_transverse_distance


    def E_z(self,z):
        z=np.array(z)
        z1=1+z
        if self.wa!=0:
            return self.E_z_wa(z)
        return np.sqrt(self.Om*(z1)**3+self.Omk*(z1)**2+self.OmR*(z1)**4+self.OmL*(z1)**(3*(1+self.w)))

    def E_z_inv(self,z,Ez_func=None): #1/E_z
        if not Ez_func:
            Ez_func=self.E_z
        return 1./Ez_func(z)

    def H_z(self,z,Ez_func=None):#hubble parameter for redshift z
        if not Ez_func:
            Ez_func=self.E_z
        return self.H0*Ez_func(z)

    def w_z(self,z):
        return self.w0+self.wa*z/(1+z) # w=w0+wa*(1-a)

    def E_z_wa(self,z):
        def DE_int_funct(z2):
            return (1+self.w_z(z2))/(1+z2) #huterer & turner 2001
        if hasattr(z, "__len__"):
            j=0
            ez=np.zeros_like(z,dtype='float64')
            for i in z:
                try:
                    DE_int=int1d(DE_int_funct,0,i)
                except:
                    DE_int=scipy_int1d(DE_int_funct,0,i,epsrel=self.rtol,epsabs=tol)[0]
                ez[j]= np.sqrt(self.Om*(1+i)**3+self.Omk*(1+i)**2+self.OmR*(1+i)**4 + self.OmL*np.exp(DE_int))
                #print i,DE_int,self.Omega_m*(1+i)**3+self.Omega_k*(1+i)**2+self.Omega_R*(1+i)**4,ez[j]
                j+=1
        else:
            try:
                DE_int=int1d(DE_int_funct,0,z)
            except:
                DE_int=scipy_int1d(DE_int_funct,0,z,epsrel=self.rtol,epsabs=tol)[0]

            ez= np.sqrt(self.Om*(1+z)**3+ self.Omk*(1+z)**2+ self.OmR*(1+z)**4
                        +self.OmL*DE_int)
        return ez

    def comoving_distance_int(self,z=[0]): #line of sight comoving distance
        if hasattr(z, "__len__"):
            j=0
            ez=np.zeros_like(z)
            for i in z:
                try:
                    ezi=int1d(self.E_z_inv,0,i)
                except:
                    ezi=scipy_int1d(self.E_z_inv,0,i,epsrel=self.rtol,epsabs=tol)[0]#integration for array (vector) of z
                if len(z)==1:
                    ez=ezi
                else:
                    ez[j]=ezi
                #print j,i,ezi,ez[j]
                j=j+1
        else:
            try:
                ez=int1d(self.E_z_inv,0,z)
            except:
                ez=scipy_int1d(self.E_z_inv,0,z,epsrel=self.rtol,epsabs=tol)[0]#integration for scalar z
        return ez*self.Dh

    def comoving_distance_trapz(self,z=[0]): #line of sight comoving distance... computed as sum         
#         if max(z)>self._z_max: #race condition
#             self.set_z(max(z))
        if not hasattr(self,'_dc'):
            ez_inv=self.E_z_inv(self._z)*self._dz
            self._dc=np.cumsum(ez_inv)-ez_inv
            self._dc*=self.Dh
        dc=np.interp(z,xp=self._z,fp=self._dc,left=0,right=np.nan)
        return dc

    def astropy_comoving_distance(self,z=[0]):
        return self.astropy_cosmo.comoving_distance(z).value
    def astropy_comoving_transverse_distance(self,z=[0]):
        return self.astropy_cosmo.comoving_transverse_distance(z).value
    
    def comoving_transverse_distance(self,z=[0]): #transverse comoving distance
        Dc=self.comoving_distance(z)
        if self.Omk==0:
            return Dc
        curvature_radius=self.Dh/np.sqrt(np.absolute(self.Omega_k))
        if self.Omk>0:
            return curvature_radius*np.sinh(Dc/curvature_radius)
        if self.Omk<0:
            return curvature_radius*np.sin(Dc/curvature_radius)

    def comoving_transverse_distance_z1z2(self,z1=[0],z2=[]): #transverse comoving distance between 2 redshift
        Dc1=self.comoving_distance(z1)
        Dc2=self.comoving_distance(z2)
        Dc=Dc2-Dc1
        if self.Omega_k==0:
            return Dc
        curvature_radius=self.Dh/np.sqrt(np.absolute(self.Omega_k))
        if self.Omega_k>0:
            return curvature_radius*np.sinh(Dc/curvature_radius)
        if self.Omega_k<0:
            #warnings.warn('This formula for DM12 is apparently not valid for Omega_k<0')#http://arxiv.org/pdf/astro-ph/9603028v3.pdf... this function doesnot work if E(z)=0 between z1,z2
            return curvature_radius*np.sin(Dc/curvature_radius)

    def angular_diameter_distance(self,z=[0]):#angular diameter distance
        Dm=self.comoving_transverse_distance(z)
        return Dm/(1+z)

    def angular_diameter_distance_z1z2(self,z1=[],z2=[]):
        return self.comoving_transverse_distance_z1z2(z1=z1,z2=z2)/(1+z2)

    def DZ_approx(self,z=[0]):# linear growth factor.. only valid for LCDM
#fitting formula (eq 67) given in lahav and suto:living reviews in relativity.. http://www.livingreviews.org/lrr-2004-8
        if not hasattr(self,'_DZ0') and not np.all(z==0):
            self._DZ0=1
            self._DZ0=self.DZ_approx(z=np.atleast_1d([0]))
        hr=self.E_z_inv(z)
        omega_z=self.Om*((1+z)**3)*((hr)**2)
        lamda_z=self.OmL*(hr**2)
        gz=5.0*omega_z/(2.0*(omega_z**(4.0/7.0)-lamda_z+(1+omega_z/2.0)*(1+lamda_z/70.0)))
        dz=gz/(1.0+z)
        return dz/self._DZ0   #check normalisation

    def DZ_int(self,z=[0],Ez_func=None): #linear growth factor.. full integral.. eq 63 in Lahav and suto
        if not hasattr(self,'_DZ0_int') and not np.all(z==0):
            self._DZ0_int=1
            self._DZ0_int=self.DZ_int(z=np.atleast_1d([0]))
        def intf(z):
            return (1+z)/self.H_z(z=z,Ez_func=Ez_func)**3
        j=0
        dz=np.zeros_like(z)
        inf=np.inf
        #inf=1.e10
        for i in z:
            try:
                dz[j]=self.H_z(i)*int1d(intf,i,inf)
            except:
                dz[j]=self.H_z(i)*scipy_int1d(intf,i,inf,epsrel=self.rtol,epsabs=tol)[0]
            j=j+1
        dz=dz*2.5*self.Om*self.H0**2
        return dz/self._DZ0_int #check for normalization

    def Omega_Z(self,z=[0]):
        z=np.array(z)
        omz=(self.H0**2/self.H_z(z)**2)*self.Om*(1+z)**3 #omega_z=rho(z)/rho_crit(z)
        return omz

    def f_z(self,z=[0]):
        return self.Omega_Z(z=z)**0.55 #fitting func

    def f_z0(self,z=[0]): #from full func f(z)=a/D(dD/da), =-(1+z)dD/dz.. full integral here but slow.. better to interpolate self.fz
        fz=np.zeros_like(z)
        j=0
        for i in z:
            z2=np.linspace(i-0.01,i+0.01,101)
            Dz=self.DZ_int(z=z2)
            f2=-1.*(1+z2)/(Dz)*np.gradient(Dz,z2[1]-z2[0])
            fz[j]=f2[50]
            j+=1
        return fz

    def EG_z(self,z=[0]):
        return self.Om/self.f_z(z=z)

    def Rho_crit(self):
        H0=H100 if cosmo_h is None else cosmo_h.H0
        rc=3*H0**2/(8*np.pi*G2)
        rc=rc.to(u.Msun/u.pc**2/u.Mpc)# unit of Msun/pc^2/mpc
        return rc.value

    def comoving_volume(self,z=[]): #z should be bin edges
        z_mean=0.5*(z[1:]+z[:-1])
        dc=self.DC(z)
        dc_mean=self.DC(z_mean)
        return (dc_mean**2)*(dc[1:]-dc[:-1])

class EH_pk(): #eisenstein_hu power spectra https://arxiv.org/pdf/astro-ph/9709112.pdf
    def __init__(self,cosmo_params={},k=[]):
#         super().__init__(cosmo_params=cosmo_params)
        self.__dict__.update(cosmo_params) #assign all input args to the class as properties
        self.k=k
        self.c=c
        self.theta=self.Tcmb/2.7
        try:
            self.theta=self.theta.value
        except:
            pass
        self.Om_h2=self.Om*(self.h**2)
        self.Ob_h2=self.Omb*(self.h**2)
        
    def z_eq(self):
        self.z_eq=2.5*1.e4*self.Om_h2*(self.theta**-4)
        self.k_eq=self.k_z(self.z_eq) #np.sqrt(2*z_eq*Om_h2*100**2)
        self.R_eq=self.R_z(self.z_eq)
            
    def k_z(self,z):
        return np.sqrt(2*z*self.Om_h2*100**2/self.c**2)
    
    def z_drag(self):
        b1=0.313*self.Om_h2**-0.419*(1+ 0.607*(self.Om_h2)**0.674)
        b2=0.238*(self.Om_h2)**.223
        self.zd = 1291/(1 + 0.659*(self.Om_h2)**0.828)*(self.Om_h2**.251)*(1 + b1*self.Ob_h2**b2)
        self.Rd=self.R_z(1+self.zd)
        
    def R_z(self,z): #ratio of baryon-photon momentum density 
        R=31.5*self.Ob_h2*1e3/(z*self.theta**4)
        return R
    
    def k_silk(self):
        self.ksilk = 1.6*(self.Ob_h2**0.52)*(self.Om_h2**0.73)*(1 + (10.4*self.Om_h2)**(-.95))
        
    def s_drag(self): #sound horizon
        self.sd=2/3/self.k_eq*np.sqrt(6/self.R_eq)*np.log((np.sqrt(1+self.Rd)+np.sqrt(self.Rd+self.R_eq))/(1+np.sqrt(self.R_eq)))
    
    
    def transfer_0(self,alpha_c,beta_c):
        q=self.k/13.41/self.k_eq
        e=2.71828
        C=14.2/alpha_c+386/(1+69.9*q**1.08)
        T0=np.log(e+1.8*beta_c*q)
        T0=T0/(T0+C*q**2)
        return T0
    
    def transfer_cdm(self):
        a1=(46.9*self.Om_h2)**0.670*(1+(32.1*self.Om_h2)**(-0.532))
        a2=(12.0*self.Om_h2)**0.424*(1+(45.0*self.Om_h2)**(-0.582))
        bf=self.Ob_h2/self.Om_h2
        alpha_c=a1**-bf*a2**(-bf**3)
        
        b1 = 0.944/(1 + (458*self.Om_h2)**-.708)
#         b2 = (0.395*self.Om_h2)**-.0266 #this is in paper.
        b2 = 0.395 * self.Om_h2 ** -0.0266 #from nbodykit
        beta_c=1/(1+b1*((1-bf)**b2)-1)
        f=1./(1+(self.k*self.sd/5.4)**4)
        self.cdm_f=f
        self.transfer_c=f*self.transfer_0(1,beta_c)+(1-f)*self.transfer_0(alpha_c,beta_c)
#         ac*(np.log(1.8*bc*q))/14.2/q**2
        
    def transfer_baryon(self):
        def G_Y(y):
            sy=np.sqrt(1+y)
            return y*(-6*sy+(2+3*y)*np.log((sy+1)/(sy-1)))
        alpha_b=2.07*self.k_eq*self.sd*(1+self.Rd)**(-3/4)*G_Y((1+self.z_eq)/(1+self.zd))
        beta_n=8.41*(self.Om_h2)**0.435
        self.beta_n=beta_n
        
        beta_b=0.5+self.Omb/self.Om+(3-2*self.Omb/self.Om)*np.sqrt(1+(17.2*self.Om_h2)**2)
        self.beta_b=beta_b
        
        ks=self.k*self.sd
        s_app=self.sd/(1+(beta_n/ks)**3)**(1./3)
        
        self.transfer_b=self.transfer_0(1,1)/(1+(ks/5.2)**2)+alpha_b/(1+(beta_b/ks)**3)*np.exp(-(self.k/self.ksilk)**1.4)
        self.transfer_b*=np.sin(self.k*s_app)/self.k/s_app
        
    
    def transfer_total(self):
        self.z_eq()
        self.z_drag()
        self.s_drag()
        self.k_silk()
        self.transfer_cdm()
        self.transfer_baryon()
        t_c=self.transfer_c
        t_b=self.transfer_b
        self.transfer_cb=self.Omb/self.Om*t_b+self.Omd/self.Om*t_c

    def pk(self):
        self.transfer_total()
        ns1=1-self.ns
        delta_H=1#self.Omega_m**(-0.785-.05*np.log(self.Omega_m))*np.exp(-.95*ns1-0.169*ns1**2) #1.94*10**-5*
        pk=self.k**self.ns*self.transfer_cb**2
        pk*=delta_H**2
        pk*=self.Ase9*1.e-9
        pk*=2*np.pi**2
        pk*=(self.c/self.h/100)**(3+self.ns)
        self.pk0=pk
        
def Rho_crit(cosmo_h=None):
    H0=H100 if cosmo_h is None else cosmo_h.H0
    rc=3*(H0*H0_unit)**2/(8*np.pi*G2*G2_unit)
    rc=rc.to(u.Msun/u.pc**2/u.Mpc)# unit of Msun/pc^2/mpc
    return rc.value

Rho_crit100=Rho_crit()
sigma_crit_norm100=(3./2.)*((H100/c)**2)/Rho_crit100
sigma_crit_norm100=sigma_crit_norm100#.value
Rho_crit100=Rho_crit100