import sys, os, gc, threading, subprocess,pickle,multiprocessing,dask,time
import numpy as np
from dask.distributed import Client,get_client,wait
from distributed import LocalCluster
from collections.abc import Mapping #to check for dicts
from distributed.client import Future

# print('pid: ',pid, sys.version)

def thread_count():
    """
        Get the number of threads spawned by the process. 
        This was useful for certain diagnostics.
    """
    pid=os.getpid()
    nlwp=subprocess.run(['ps', '-o', 'nlwp', str(pid)], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')[1]
    nlwp=int(nlwp)
    thc=threading.active_count()
    current_th=threading.current_thread()
    #print(pid, ' thread count, os:',nlwp, 'py:', thc)
    #print('thread id, os: ',os.getpid(), 'py: ' , current_th, threading.get_native_id() )

    return nlwp, thc


def get_size(obj, seen=None): #https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size
    
def get_size_pickle(obj):
    """
        Get the size of an object via pickle.
    """
    yy=pickle.dumps(obj)
    return np.around(sys.getsizeof(yy)/1.e6,decimals=3)

def dict_size_pickle(dic,print_prefact='',depth=2): #useful for some memory diagnostics
    """
        Get a size of the dictionary using pickle. Will also output size of 
        dict elements depending on depth.
    """
    print(print_prefact,'dict full size ',get_size_pickle(dic))
    if not isinstance(dic,dict):
        print(dic, 'is not a dict',depth)
        return 
    if depth <=0:
        return
    for k,v in dic.items():
        if isinstance(v,dict) and depth>0:
            dict_size_pickle(v,print_prefact=print_prefact+' '+str(k),depth=depth-1)
        else:
            print(print_prefact,'dict obj size: ',k, get_size_pickle(v))
    return

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    lst2=[]
    for i in range(0, len(lst), n):
#         yield lst[i:i + n]
        lst2+=[lst[i:i + n]]
    return lst2

def pickle_deepcopy(obj):
       return pickle.loads(pickle.dumps(obj, -1))

def scatter_dict(dic,scheduler_info=None,depth=2,broadcast=False,return_workers=False,workers=None): #FIXME: This needs some improvement to ensure data stays on same worker. Also allow for broadcasting.
    """
        depth: Need to think this through. It appears dask can see one level of depth when scattering and gathering, but not more.
    """
    if dic is None:
        print('scatter_dict got empty dictionary')
    else:
        client=client_get(scheduler_info=scheduler_info)
#         dic['scatter_depth']=depth
        for k in dic.keys():
            if k=='scatter_depth':
                continue
            if isinstance(dic[k],dict) and depth>0:
                dic[k],workers=scatter_dict(dic[k],scheduler_info=scheduler_info,depth=depth-1,
                                            broadcast=broadcast,return_workers=True,workers=workers)
            elif isinstance(dic[k],Future): #don't want future of future.
                print('scatter dict was passed a future, gathering and re-scattering', k)
                dic[k]=client.gather(dic[k])
                dic[k]=client.scatter(dic[k],broadcast=broadcast,workers=workers)
                workers=list(client.who_has(dic[k]).values())[0]
            else:
                if isinstance(dic[k], np.ma.MaskedArray): #there is a numpy bug that creates problems inside dask scatter.
                        #https://github.com/numpy/numpy/issues/10217
#                     print('scatter-dict got masked array: ',k,type(dic[k]),' will be scattered with filled values ')
                    dic[k]=client.scatter(dic[k].filled(),broadcast=broadcast,workers=workers)
                else:
#                     print('scatter_dict:',k,dic[k],workers)
                    dic[k]=client.scatter(dic[k],broadcast=broadcast,workers=workers)
                workers=list(client.who_has(dic[k]).values())[0]
#                 print('scatter-dict ',k,workers)
    if return_workers:
        return dic,workers
    return dic


def replicate_dict(dic,scheduler_info=None,workers=None,branching_factor=1): #FIXME: This needs some improvement to ensure data stays on same worker. Also allow for broadcasting.
    """
        depth: Need to think this through. It appears dask can see one level of depth when scattering and gathering, but not more.
    """
    if dic is None:
        print('replicate_dict got empty dictionary')
    else:
        print('replicate_dict: ',dic)
        client=client_get(scheduler_info=scheduler_info)
        if isinstance(dic,dict):
            for k in dic.keys():
                dic[k]=replicate_dict(dic[k],scheduler_info=scheduler_info,workers=workers,branching_factor=branching_factor)
        elif isinstance(dic,Future): #don't want future of future.
            dic=client.replicate(dic,workers=workers,branching_factor=branching_factor)
        else:
            print('replicate_dict: not a future or dict',dic[k])
    return dic


def wait_futures(futures,sleep_time=.05,threshold=0.5):
#     wait(futures,timeout=200*len(futures))
    all_done=False
    while not all_done:
        time.sleep(sleep_time)
        all_done=np.mean([future.status=='finished' for future in futures])>threshold
#         all_done=np.all([future.status=='finished' for future in futures])

def gather_dict(dic,scheduler_info=None): 
    """
        depth: Need to think this through. It appears dask can see one level of depth when scattering and gathering, but not more.
    """
    if dic is None:
        print('gather_dict got empty dictionary')
        return dic
    client=client_get(scheduler_info=scheduler_info)
#     depth=1
#     if isinstance(dic,dict):
#         depth=dic.get('scatter_depth')
#         if depth is None:
#             depth=1
    if isinstance(dic,Future):# or depth <=0:
        dic=client.gather(dic) #this may not always be clean for nested dicts
#     else:
    if isinstance(dic,dict):# and depth >0:
        for k in dic.keys():
            if isinstance(dic[k],dict):
                dic[k]=gather_dict(dic[k],scheduler_info=scheduler_info)
            elif isinstance(dic[k],Future):
                try:
                    dic[k]=client.gather(dic[k])
                except Exception as err:
                    print('gather dict got error at key: ', k)
                    raise(err)
                dic[k]=gather_dict(dic[k],scheduler_info=scheduler_info)
    return dic

def client_get(scheduler_info=None):
    """
        Get the dask client running on the scheduler
    """
    if scheduler_info is not None:
        client=get_client(address=scheduler_info['address'])
    else:
        client=get_client()
    return client

def clean_client(scheduler_info=None):
    if scheduler_info is not None:
        client=client_get(scheduler_info=scheduler_info)
        self.client.shutdown() #this will close the scheduler as well.


worker_kwargs={}#{'memory_spill_fraction':.95,'memory_target_fraction':.95,'memory_pause_fraction':1}
def start_client(Scheduler_file=None,local_directory=None,ncpu=None,n_workers=1,threads_per_worker=None,
                  worker_kwargs=worker_kwargs,LocalCluster_kwargs={},dashboard_address=8801,
                 memory_limit='120gb',processes=False):
    """
        Start a dask client. If no schduler is passed, a new local cluster is started.
    """
    LC=None
    if local_directory is None:
        local_directory='./temp_skylens/'
    local_directory+='pid'+str(os.getpid())+'/'
    try:  
        os.makedirs(local_directory)  
    except Exception as error:  
        print('error in creating local directory: ',local_directory,error) 
    if threads_per_worker is None:
        if ncpu is None:
            ncpu=multiprocessing.cpu_count()-1
        threads_per_worker=ncpu
    if n_workers is None:
        n_workers=1
    if Scheduler_file is None:
        print('Start_client: No scheduler file, will start local cluster at ',local_directory)
                #     dask_initialize(nthreads=27,local_directory=dask_dir)
                #     client = Client()
#         dask.config.set(scheduler='threads')
        LC=LocalCluster(n_workers=n_workers,processes=processes,threads_per_worker=threads_per_worker,
                        local_directory=local_directory,dashboard_address=dashboard_address,
                        memory_limit=memory_limit,**LocalCluster_kwargs,**worker_kwargs
                   )
        client=Client(LC)
    else:
        print('Start_client: Using scheduler file', Scheduler_file)
        client=Client(scheduler_file=Scheduler_file,processes=False)
    client.wait_for_workers(n_workers=1)
    scheduler_info=client.scheduler_info()
    scheduler_info['file']=Scheduler_file
    return LC,scheduler_info #client can be obtained from client_get

def l_max_modes(l):
    return l**2+l

def get_l_bins(l_min=2,l_max=1000,N_bins=20,binning_scheme='linear',max_modes=None,min_modes=None):
    
    n_modes_min=l_max_modes(l_min)
    n_modes_max=l_max_modes(l_max)
    n_modes=n_modes_max-n_modes_min
    
    if binning_scheme=='log':
        n_modes_bin=np.logspace(np.log10(n_modes_min),np.log10(n_modes_max),N_bins+1)
    if binning_scheme=='linear':
        n_modes_bin=np.linspace(n_modes_min,n_modes_max,N_bins+1)
    if binning_scheme=='constant':
        n_modes_bin=np.ones(N_bins+1)*np.int(n_modes/N_bins)
    
    def bin_norm(n_modes_bin,n_modes):
        n_modes_bin=n_modes_bin/n_modes_bin.sum()
        n_modes_bin*=n_modes
        n_modes_bin=np.int32(n_modes_bin)
        n_modes_bin[n_modes_bin==0]+=1
        return n_modes_bin
    
    n_modes_bin=bin_norm(n_modes_bin,n_modes)
    
    if max_modes is not None:
        n_modes_bin[n_modes_bin>max_modes]=max_modes
    if min_modes is not None:
        n_modes_bin[n_modes_bin<min_modes]=min_modes
        
    n_modes_bin=bin_norm(n_modes_bin,n_modes)
    
    l_bins=np.zeros(N_bins+1)
    l_bins[0]=l_min
    l_bins[-1]=l_max
    for i in np.arange(N_bins):
        n_mode_i=0
        l_i=l_bins[i]+1
        while n_mode_i<n_modes_bin[i]:
            n_mode_i+=l_i*2+1
            l_i+=1
            if l_i>=l_max:
                break
        l_bins[i+1]=l_i
    if l_bins[-1]<l_max:
        l_bins[-1]=l_max
    return np.int32(l_bins)