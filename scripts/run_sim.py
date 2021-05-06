from sim_args import *
from sim_functions import *
import argparse
from jk_utils import *
import faulthandler; faulthandler.enable()

if __name__=='__main__':
    test_run=False
    parser = argparse.ArgumentParser()
    parser.add_argument("--cw", "-cw",type=int, help="use complicated window")
    parser.add_argument("--uw", "-uw",type=int, help="use unit window")
    parser.add_argument("--dW", "-dW",type=int, help="use delta_W")
    parser.add_argument("--lognormal", "-l",type=int, help="use complicated window")
    parser.add_argument("--blending", "-b",type=int, help="use complicated window")
    parser.add_argument("--ssv", "-ssv",type=int, help="use complicated window")
    parser.add_argument("--noise", "-sn",type=int, help="use shot noise")
    parser.add_argument("--scheduler", "-s", help="Scheduler file")
    parser.add_argument("--dask_dir", "-Dd", help="dask log directory")
    args = parser.parse_args()

    gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
    gc.enable()

    # Read arguments from the command line
    args = parser.parse_args()

    use_complicated_window=False if args.cw is None else np.bool(args.cw)
    unit_window=False if args.uw is None else np.bool(args.uw)
    delta_W=True if args.dW is None else np.bool(args.dW)
    lognormal=False if args.lognormal is None else np.bool(args.lognormal)

    do_blending=False if args.blending is None else np.bool(args.blending)
    do_SSV_sim=False if args.ssv is None else np.bool(args.ssv)
    use_shot_noise=True if args.noise is None else np.bool(args.noise)

    Scheduler_file=args.scheduler
    dask_dir=args.dask_dir

    print(use_complicated_window,unit_window,args.uw,use_shot_noise,delta_W,Scheduler_file) #lognormal,do_blending,do_SSV_sim,

    nsim=1000
    if test_run:
        nside=128
        lmax_cl=int(nside)
        l0=np.arange(lmin_cl,lmax_cl)
        l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl),Nl_bins))
        l=l0
        window_lmax=lmax_cl*2#50
        nsim=10
        print('this will be test run', do_xi)
        l0w=np.arange(3*nside-1)
        window_cl_fact=np.cos(np.pi/2*(l0w/w_smooth_lmax)**10)
        ww=1000*np.exp(-(l0w-mean)**2/sigma**2)
        WT_kwargs['l']=l0   

    print('initializing dask, scheduler: ',Scheduler_file)
    LC,scheduler_info=start_client(Scheduler_file=Scheduler_file,local_directory=dask_dir,n_workers=5,threads_per_worker=1,
                                      memory_limit='120gb',dashboard_address=8801,processes=True)
    client=client_get(scheduler_info=scheduler_info)
    print('client: ',client)
    WT_kwargs['scheduler_info']=scheduler_info
    if unit_window:
        window_cl_fact=0

    np.random.seed(0)
    print('getting win')
    z0_galaxy=0.5
    galaxy_zbins=lsst_source_tomo_bins(zp=np.array([z0_galaxy]),ns0=10,use_window=use_window,nbins=1,
                                  scheduler_info=scheduler_info,n_zs=n_zs,delta_W=delta_W,
                                window_cl_fact=window_cl_fact*(1+ww*use_complicated_window),
                                f_sky=f_sky,nside=nside,unit_win=unit_window,use_shot_noise=True)

    np.random.seed(1)
    z0_shear=1 #1087
    shear_zbins=lsst_source_tomo_bins(zp=np.array([z0_shear]),ns0=26,use_window=use_window,n_zs=n_zs,nbins=1,
                                        window_cl_fact=window_cl_fact*(1+ww*use_complicated_window),delta_W=delta_W,
                                        f_sky=f_sky,nside=nside,scheduler_info=scheduler_info,
                                        unit_win=unit_window,use_shot_noise=True)

    
#     mask=shear_zbins[0]['window']>-1000
#     mask=mask.astype('bool')
#     jkmap=jk_map(mask=mask,nside=nside,njk1=njk1,njk2=njk2)

    
    print('zbins done')#,thread_count())
    if not use_shot_noise:
        for t in shear_zbins['SN'].keys():
            shear_zbins['SN'][t]*=0
            galaxy_zbins['SN'][t]*=0

    Skylens_kwargs=parse_dict(locals())
    kappa_win=Skylens(**Skylens_kwargs)
    clG_win=kappa_win.cl_tomo(corrs=corrs)
    cl0_win=clG_win['stack'].compute()

    kappa_win_xib=None
    if do_xi:
        xiWG_L=kappa_win.xi_tomo()
        xiW_L=xiWG_L['stack'].compute()
        Skylens_kwargs['use_binned_l']=True
        Skylens_kwargs['use_binned_theta']=True
        kappa_win_xib=Skylens(**Skylens_kwargs)
        Skylens_kwargs['use_binned_l']=use_binned_l
        Skylens_kwargs['use_binned_theta']=use_binned_theta

    l=kappa_win.window_l
    Om_W=np.pi*4*f_sky
    theta_win=np.sqrt(Om_W/np.pi)
    l_th=l*theta_win
    Win0=2*jn(1,l_th)/l_th
    Win0=np.nan_to_num(Win0)

    use_window0=copy.deepcopy(Skylens_kwargs['use_window'])
    Skylens_kwargs['use_window']=False
    kappa0=Skylens(**Skylens_kwargs)
    Skylens_kwargs['use_window']=use_window0

    clG0=kappa0.cl_tomo(corrs=corrs) 
    cl0=clG0['stack'].compute()

    if do_xi:
         xiG_L0=kappa0.xi_tomo()
         xi_L0=xiG_L0['stack'].compute()

    cl0={'cl_b':{},'cov':{},'cl':{}}
    cl0_win={'cl_b':{},'cov':{}}

    for corr in corrs:
        cl0['cl'][corr]=client.compute(clG0['cl'][corr][bi]).result()

        cl0['cl_b'][corr]=client.compute(clG0['pseudo_cl_b'][corr][bi]).result()
#         cl0['cov'][corr]=client.compute(clG0['cov'][corr+corr]).result()[0]

        cl0_win['cl_b'][corr]=client.compute(clG_win['pseudo_cl_b'][corr][bi]).result()
#         cl0_win['cov'][corr]=client.compute(clG_win['cov'][corr+corr]).result()[0]['final_b']

    from binning import *

    mask=shear_zbins[0]['window']>-1.e-20    
    mask=mask.astype('bool')
    
    njk1=np.int32(np.sqrt(njk))
    njk2=njk1
    njk=njk1*njk2
    jkmap=jk_map(mask=mask,nside=nside,njk1=njk1,njk2=njk2)

    print('binning coupling matrices done',WT_kwargs['theta']/d2r)

    cl_sim_W=Sim_jk(nsim=nsim,do_norm=False,njk=njk,kappa0=kappa0,jkmap=jkmap,
                    skylens_kwargs=Skylens_kwargs,kappa_class_xib=kappa_win_xib,
                    kappa_class=kappa_win,use_shot_noise=use_shot_noise,use_cosmo_power=use_cosmo_power, #,fsky=f_sky
                    nside=nside,lognormal=lognormal,lognormal_scale=lognormal_scale,do_xi=do_xi,
                    add_SSV=do_SSV_sim,add_tidal_SSV=do_SSV_sim,add_blending=do_blending)
    
    test_home=home+'/tests/'
    if do_xi and not do_pseudo_cl:
            fname=test_home+'/xi0_sims_newN'+str(nsim)+'_ns'+str(nside)+'_lmax'+str(lmax_cl)+'_wlmax'+str(window_lmax)+'_fsky'+str(f_sky)
    elif not do_xi and do_pseudo_cl:
        fname=test_home+'/cl0_sims_newN'+str(nsim)+'_ns'+str(nside)+'_lmax'+str(lmax_cl)+'_wlmax'+str(window_lmax)+'_fsky'+str(f_sky)
    elif do_xi and do_pseudo_cl:
        fname=test_home+'/xi0_cl0_sims_newN'+str(nsim)+'_ns'+str(nside)+'_lmax'+str(lmax_cl)+'_wlmax'+str(window_lmax)+'_fsky'+str(f_sky)
    if lognormal:
        fname+='_lognormal'+str(lognormal_scale)
    if not use_shot_noise:
        fname+='_noSN'
    if do_SSV_sim:
        fname+='_SSV'
    if do_blending:
        fname+='_blending'

    if unit_window:
        fname+='_unit_window'
    if use_complicated_window:
        fname+='_cWin'
    if smooth_window:
        fname+='_smooth_window'
    if delta_W:
        fname+='_deltaW'
    elif not delta_W:
        fname+='_delta'
        
    fname+='.pkl'

    print(fname)

    outp={}
    outp['simW']=cl_sim_W.outp
    outp['shear_zbins']=shear_zbins
    outp['galaxy_zbins']=galaxy_zbins
    outp['cl0']=cl0
    outp['cl0_win']=cl0_win
    outp['nsim']=nsim
    outp['Skylens_kwargs']=Skylens_kwargs
#     print('all done:',outp)
    with open(fname,'wb') as f:
        pickle.dump(outp,f)
    written=True

    print(fname)
    print('all done')
    # client.shutdown() #this kills the dask cluster.
    # try:
    #     if Scheduler_file is None:
    #         LC.close()
    # except Exception as err:
    #     print('LC close error:', err)
    # sys.exit(0)