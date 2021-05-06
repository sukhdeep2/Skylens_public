"""
This file contains scripts to compute the wigner_3j matrix in parallel, with m1=m2=m3=0. The main 
funtion used to compute the matrix is defined in wigner_functions.py.

The matrix returned is of the form:   l_1  l_2  wl
                                        0   0   0 
l_1 and l_2 are both assumed to have the same range: [0,lmax]. This can easily changed in the code
if desired (lmax is user defined), including allowing different range for l_1 and l_2.
wl has the range [0,wlmax], wlmax is used defined. wlmax is recommended to be 2*lmax

The wigner files are stored as compressed arrays using the zarr package.
"""

from wigner_functions import *
import zarr
import time
from multiprocessing import Pool

lmax=5000 #~nside*3.. or the lmax to be used in the analysis.
wlmax=5000 #This needs to be atleast 2X lmax in general, unless you are certain window is narrow in ell space.

m1=0
m2=0
m3=0

lmax=np.int(lmax)
wlmax=np.int(wlmax)

#define path to save the file.
home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/'
fname=home+'/wig3j_l{lmax}_w{wlmax}_{i}_reorder.zarr'  #path to save the files
fname=fname.format(i=m2,lmax=lmax,wlmax=wlmax)
print('will save to ',fname)

lmax+=1
wlmax+=1
ncpu=30
l_step=100 
w_l=np.arange(wlmax)
l=np.arange(lmax)

lb=np.arange(lmax,step=l_step)
z1 = zarr.open(fname, mode='w', shape=(wlmax,lmax,lmax), #0-lmax
               chunks=(wlmax/10, lmax/10,lmax/10),
               dtype='float32',overwrite=True)

j_max=np.amax(lmax+lmax+wlmax+10)
calc_factlist(j_max)

j3=np.arange(wlmax)
    
def wig3j_recur_2d(j1b,m1,m2,m3,j3_outmax,step,j2b):
    """
    Computes a smaller part of the wigner_3j matrix.
    Called multiple times in parallel.
    """
    if j2b<j1b: #we exploit j1-j2 symmetry and hence only compute for j2>=j1
        return [j1b,j2b,0]
    if np.absolute(j2b-j1b-step-1)>j3_outmax: #given j1-j2, there is a min j3 for non-zero values. If it falls outside the required j3 range, nothing to compute
        return [j1b,j2b,0]
    #out= np.zeros((j3_outmax,min(step,lmax-j1b),min(step,lmax-j2b)))

    j1=np.arange(j1b,min(lmax,j1b+step))
    j2=np.arange(j2b,min(lmax,j2b+step))

    j1s=j1.reshape(1,len(j1),1)
    j2s=j2.reshape(1,1,len(j2))
    j3s=j3.reshape(len(j3),1,1)

    out=wigner_3j_000(j1s,j2s,j3s,0,0,0)
    
    t3=time.time()
    print('done ',j1b,j2b,t3-t1)
    return [j1b,j2b,j1,j2,out]

t0=time.time()
step=l_step
for lb1 in lb:
    ww_out={}
    t1=time.time()
    funct=partial(wig3j_recur_2d,lb1,m1,m2,m3,wlmax,l_step)
    pool=Pool(ncpu)
    out_ij=pool.map(funct,lb,chunksize=1)
    pool.close()
    i=0
    for lb2 in lb:
        if lb2<lb1:
            i+=1
            continue
        if np.absolute(lb2-lb1-step-1)>wlmax:
            i+=1
            continue
        out=out_ij[i]
        j1b=out[0]
        j2b=out[1]
        #print(lb2,lb1,j1b,j2b)        
        j1=out[2]
        j2=out[3]
        out=out[4]
        z1[:,lb1:lb1+step,lb2:lb2+step]=out #cannot write in parallel, race conditions
        #for j1i in np.arange(len(j1)):
         #   for j2i in np.arange(len(j2)):
          #      if j2[j2i]==j1[j1i]:
           #         out[:,j1i,j2i]*=0 #don't want to add diagonal twice below.
        z1[:,lb2:lb2+step,lb1:lb1+step]=out.transpose(0,2,1) #exploit j1-j2 symmetry
        i+=1
    t2=time.time()
    print('done',lb1,t2-t1)
t2=time.time()
print('done all',t2-t0)
