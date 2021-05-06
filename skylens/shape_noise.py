#written by Sukhdeep Singh
#Test code to compute the shape noise component in the shear-shear correlations.
#This code uses brute force hankel transforms, to can be bit noisy, especially for small bins.

from scipy.special import jn, jn_zeros,jv

d2r=np.pi/180.

def hankel_brute_force(l=[],theta=[],cl12=[],j_nu=0): 
#only auto covariance. cl12 and l should have same shape, i.e. cl12 is not a matrix but a vector.
#we transform like eq. 34 in https://arxiv.org/pdf/0708.0387.pdf .
    J=jn(j_nu,np.outer(theta,l))
    dl=np.gradient(l)
    cov=np.dot(J,(J*cl12*l*dl).T)
    cov/=2*np.pi
    return cov

#more basic binning code for testing.
def bin_cov(r=[],cov=[],r_bins=[]):
    bin_center=0.5*(r_bins[1:]+r_bins[:-1])
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
#             if i==j:
#                 print( i,j,norm_ij)
            if norm_ij==0:
                continue
            cov_int[i][j]=np.sum(cov_r_dr[xi,:][:,xj])/norm_ij
    #cov_int=np.nan_to_num(cov_int)
#         print np.diag(cov_r_dr)
    return cov_int


n_gal=1.496 #DES first bin
area= 5000.0
sigma_e= 0.39441348451  
sigma_gamma=sigma_e #/np.sqrt(2)
th_min=2.5/60
th_max=250./60
n_th_bins=20

n_gal_rad=n_gal*60.*60./d2r**2
SN=sigma_gamma**2/2./n_gal_rad

f_sky=area/(4*np.pi/d2r**2)

ell_max=1.e4
ell=np.arange(ell_max)+1

th_bins=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
th=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins*60)
thb=0.5*(th_bins[1:]+th_bins[:-1])

SN2=SN**2*np.ones_like(ell)

cov_th=hankel_brute_force(l=ell,theta=th*d2r,cl12=SN2)
cov_th/=4*np.pi*f_sky

cov_th*=2 #this is auto covariance, hence we have 2 terms, C1324 abd C1423. 
            #This is the reason by Joachimi+2007 have pi in denominators of eq. 34 and not 2pi.
    
cov_th*=2 #because we have e1e1 +/- e2e2

cov_th_binned=bin_cov(r=th,cov=cov_th,r_bins=th_bins)

fname='shape_noise_cov.dat'
np.savetxt(fname,cov_th_binned)

fname='shape_noise_theta_bins.dat'
np.savetxt(fname,th_bins)

fname='shape_noise_cov_unbinned.dat'
np.savetxt(fname,cov_th)

fname='shape_noise_cov_unbinned_theta.dat'
np.savetxt(fname,th)

#using direct expression from Joachimi+ for real space
area_rad=4*np.pi*f_sky
N1=np.pi*area_rad*(2*n_gal_rad)**2 #area and number density in the normalization

dthb=th_bins[1:]-th_bins[:-1]
N2=(thb*dthb*d2r**2)**2  #bin volumes in the normalization
N2/=thb*dthb*d2r**2 #bin volume in the numerator.

SN_diag=sigma_e**4/N1/N2
SN_diag*=2 #because we have e1e1 +/- e2e2

fname='shape_noise_diag.dat'
np.savetxt(fname,SN_diag)


# N1*=2 #factor of 2 issue in Joachimi's paper? N1*N2 is the definition of Np above eq. 34.

"""
Direct real space expression for shape noise should be:

unbinned, SN(theta_1,theta_2)= 2 * 1./(2*pi*A) * (sigma_e**2/(2n) )**2 \delta_D(theta_1,theta_2)/theta_1

factor of 2 in the beginning is because we have 2 terms. 

binned SN= \int_b1 d\theta_1 theta_1 \int_b2 d\theta_2 theta_2 SN(theta_1,theta_2) / N
where N=(\int_b1 d\theta_1 theta_1)*(\int_b2 d\theta_2 theta_2)

b1,b2 are the bins
\int_b1 d\theta_1 theta_1 \approx \theta_1'\Delta\theta_1' (mean theta of bin times the size of the bin)

This gives

binned SN= Unbinned SN * \theta_1'\Delta\theta_1' / (\theta_1'\Delta\theta_1')**2

        = 2* 1./(2*pi*A) * (sigma_e**2/(2n) )**2 / (\theta_1'\Delta\theta_1')
        
        =\sigma_2**4 / (pi A 4 n**2 \theta_1'\Delta\theta_1')
"""

