from lensing_lensing import *
import numpy as np

def fisher():
  def __init__(self,):
    pass

  def calc_derivative(self,func=None,params=None,var=None,N_deriv=3,dx=0.01,do_log=False,
                      func_kwargs={}):
    derivs={}
    v0=params[var]
    v_sign=np.sign(v0)
    dv0s=0

    if do_log and v0!=0:
      log_v0=np.log10(np.absolute(v0))
      v0s=np.linspace(log_v0*(1.-dx),log_v0*(1.+dx),N_deriv)
      dv0s=v_sign*(v0s[1]-v0s[0])
      v0s=v_sign*10**v0s
    else:
      v0s=np.linspace(v0*(1.-dx),v0*(1.+dx),N_deriv)
      dv0s=(v0s[1]-v0s[0])
    
    Fs={}
    for i in np.arange(N_deriv):
      params_i=params.copy()
      params_i[v]=v0s[i]
      Fs[i]=func(**params_i)

    Fs['deriv']=(Fs[N_deriv-1]-Fs[0])/dv0s/(N_deriv-1)
    Fs['v0s']=v0s
    Fs['dv0s']=dv0s
    return Fs

  def calc_fisher_fix_cov(self,func=None,params=None,vars=None,N_deriv=3,dx=0.01,
                  do_log=False,func_kwargs={},cov=[]):
        
    derivs={}
    nvar=lens(vars)
    for v in vars:
      derivs[v]=self.calc_derivative(func=func,params=params,var=v,N_deriv=N_deriv,dx=dx,
                                  do_log=do_log,func_kwargs=func_kwargs)
    
    fisher=np.zeros((nvar,nvar))
    cov_inv=np.linalg.inv(cov)
    for i in np.arange(nvar):
      di=derivs[vars[i]]['deriv']
      for j in np.arange(nvar):
        dj=derivs[vars[i]]['deriv']
        fisher[i][j]=np.dot(di,np.dot(cov_inv,dj))
    out={'fisher':fisher,'derivs':derivs}
    out['cov']=np.linalg.inv(fisher)
    out['error']=np.sqrt(np.diag(out['cov']))
    return out


def fisher_calc(params=['As'],Nx=3,dx_max=0.01,do_log=False,kappa_class=None):
    cosmo_fid=kappa_class.Ang_PS.PS.cosmo_params.copy()
    
    cl0G=kappa_class.kappa_cl_tomo()
    cl_t=cl0G['stack'].compute()
    cov=cl_t['cov']
    kappa_class.Ang_PS.reset()
    kappa_class.do_cov=False

    Dx=np.linspace((1-dx_max),(1+dx_max),Nx)
    ndim=len(params)
    
    x_vars={}
    models={}
    model_derivs={}
    covs={}
    for p in params:
        x0=cosmo_fid[p]
        if do_log:
            x0=np.absolute(x0)
            x_vars[p]=x0**Dx
            if x0==1:
                x_vars[p]=(2.**Dx)/2. # 1**x=1
            x_vars[p]*=np.sign(cosmo_fid[p])
        else:
            x_vars[p]=x0*Dx #np.linspace(x0*(1-dx_max),x0*(1+dx_max),Nx)
        
        models[p]={}
#         covs[p]={}
        model_derivs[p]={}
        for i in np.arange(Nx):
            cosmo_t=cosmo_fid.copy()
            cosmo_t[p]=x_vars[p][i]
            cl0G=kappa_class.kappa_cl_tomo(cosmo_params=cosmo_t)
            cl_t=cl0G['stack'].compute()
            models[p][i]=cl_t['cl']
#             covs[p][i]=cl_t['cov']
            kappa_class.Ang_PS.reset()
        model_derivs[p]=models[p][Nx-1]-models[p][0]
        if do_log:
            model_derivs[p]/=np.log(x_vars[p][Nx-1]/x_vars[p][0])
        else:
            model_derivs[p]/=(x_vars[p][Nx-1]-x_vars[p][0])
#     cov=covs[p][1]
    cov_inv=np.linalg.inv(cov)
    cov_p_inv=np.zeros([ndim]*2)
    i1=0
    for p1 in params:
        i2=0
        for p2 in params:
            cov_p_inv[i1,i2]=np.dot(model_derivs[p1],np.dot(cov_inv,model_derivs[p2]))
            i2+=1
        i1+=1
    print (cov_p_inv)
    out={}
    out['cov_p']=np.linalg.inv(cov_p_inv)
    out['error']=np.sqrt(np.diag(out['cov_p']))
    return out