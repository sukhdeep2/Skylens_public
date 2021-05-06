import sys
sys.path.insert(0, '/home/sukhdees/project/code/python_scripts/')
from astropy.coordinates import ICRS, Galactic,SkyCoord
import numpy as np
import healpy as hp
from astropy import units as units

def jk_mean(p={},njk=100):
    if check_empty(p):
        print ('jk-mean: got empty dict')
        return p
    p2={}
    nn=np.arange(njk)
    for i in nn: #p.keys():
        #if i in nn:
        p2[i]=p[i]
    jk_vals=np.array(list(p2.values()))
    mean=np.mean(jk_vals,axis=0)
    #print mean
    var=np.var(jk_vals,axis=0,ddof=0)*(njk-1)
    p['jk']=mean
    p['jk_err']=np.sqrt(var)
    return p

def jk_mean_percentile(p={},njk=100):
    if check_empty(p):
        print ('jk-mean: got empty dict')
        return p
    dat=[p[i] for i in np.arange(njk)]
    p['jk']=np.mean(dat,axis=0)
    p['jk_err']=np.percentile(dat,[16,84],axis=0)-p['jk']
    p['jk_err']*=np.sqrt((njk-1.)*(njk-1.)/njk)
    return p

def rotate_mask(mask=[],nside=512,frame_in='galactic',frame_out='',nest=False,nside_smooth_fact=4):
    order_out='Ring'
    if nest:
        order_out='nested'
    lm2=hp.pixelfunc.ud_grade(mask,nside_out=nside,pess=True,order_in='Ring',                                                                                        
                              order_out=order_out,dtype='int')
    mask2=lm2.astype('bool')
    if frame_out=='icrs':
        print('Assuming mask is in galactic. Converting to icrs')
        npix=np.arange(hp.nside2npix(nside))
        theta,phi=hp.pix2ang(nside=nside,ipix=npix)
        c=SkyCoord(b=(-theta+np.pi/2.)*units.radian, l=phi*units.radian,frame='galactic')  
        c_icrs=c.icrs
        gpix=hp.pixelfunc.ang2pix(nside=nside,nest=nest,theta=c_icrs.dec.radian*-1+np.pi/2.,                                                                         
                              phi=c_icrs.ra.radian)                                                                                                             
        lm3=np.zeros_like(lm2)
        lm3[gpix]=lm2
        lm4=hp.pixelfunc.ud_grade(np.float64(lm3),nside_out=nside/nside_smooth_fact,pess=False,order_in='Ring',                                                                                        
                              order_out=order_out)
        x=lm4>0
        lm4=x
        lm4=hp.pixelfunc.ud_grade(lm4,nside_out=nside,pess=True,order_in='Ring',                                                                                        
                              order_out=order_out,dtype='int')
        mask2=lm4.astype('bool')
    return mask2



def jk_map(mask=[],nside=512,njk1=10,njk2=10,nest=False,frame_in='galactic',frame_use='galactic'):
    if frame_in!=frame_use:
        mask2=rotate_mask(mask=mask,nside=nside,nest=nest,frame_in=frame_in,frame_out=frame_use)
    else:
        mask2=mask
    njk=njk1 #np.sqrt(njk_tot)                                                                                                                                       
    x=mask2>0                                                                                                                                                        
    nii=np.int64(sum(x)/njk+1)                                                                                                                                       
                                                                                                                                                                     
    jkmap=np.zeros_like(mask2,dtype='float64')-999                                                                                                                   
    ii=np.int64(np.arange(len(mask2)))                                                                                                                               
    ii=ii[x]                                                                                                                                                         
    for i in np.int64(np.arange(njk)):                                                                                                                               
        if i==nii%njk:                                                                                                                                               
            nii-=1                                                                                                                                                   
        indx=ii[i*nii:(i+1)*nii]                                                                                                                                     
        jkmap[indx]=i                                                                                                                                                
                                                                                                                                                                     
    #njk2=njk_tot/njk                                                                                                                                                
    jkmap2=np.zeros_like(jkmap)-999                                                                                                                                  
    ii=np.int64(np.arange(len(jkmap2)))                                                                                                                              
    for i in np.int64(np.arange(njk)):                                                                                                                               
        x=jkmap==i                                                                                                                                                   
        ii2=ii[x]                                                                                                                                                    
        add_f=0                                                                                                                                                      
        if i==0:#to get circle at galactic pole                                                                                                                      
            nii=np.int64(sum(x)/njk2)                                                                                                                                
            jkmap2[ii2[:nii]]=0                                                                                                                                      
            ii2=ii2[nii:]                                                                                                                                            
            add_f=1                                                                                                                                                  
        theta,phi=hp.pixelfunc.pix2ang(nside,ii2,nest=nest)                                                                                                          
        phi_bins=np.percentile(phi,np.linspace(0,100,njk2+1-add_f))                                                                                                  
        phi_bins[0]-=0.1                                                                                                                                             
        phi_bins[-1]+=0.1                                                                                                                                            
        for j in np.int64(np.arange(njk2-add_f)):                                                                                                                    
            xphi=phi>phi_bins[j]                                                                                                                                     
            xphi*=phi<phi_bins[j+1]                                                                                                                                  
            indx=ii2[xphi]                                                                                                                                           
            jkmap2[indx]=i*njk2+j+add_f
    return jkmap2
            
def gal_jk(mask=[],dat=[],nside=512,njk1=10,njk2=10,nest=False,frame_in='galactic',frame_use='galactic',return_jkmap=False):
    #dat_galactic=c.transform_to(Galactic)
    jkmap2=jk_map(mask=mask,nside=nside,njk1=njk1,njk2=njk2,nest=nest,frame_in=frame_in,frame_use=frame_use)
    c=SkyCoord(ra=dat['RA']*units.degree, dec=dat['DEC']*units.degree,frame='icrs')   
    if frame_use=='icrs':
        gpix=hp.pixelfunc.ang2pix(nside=nside,nest=False,theta=c.dec.radian*-1+np.pi/2.,                                                                
                              phi=c.ra.radian)           
    elif frame_use=='galactic':
        dat_galactic=c.galactic                                                        
    jkgal=jkmap2[gpix]
    if return_jkmap:
        return jkmap2,jkgal
    else:
        return jkgal
