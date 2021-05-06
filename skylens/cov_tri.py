"""
Matter trispectrum for covariance. Implementation of Harnois-Deraps&Pen 2011. 
Developed by Yin Li.
"""
import numpy as np
#

class cov_matter_tri():
    def __init__(self,k=None,l=None):
        self.k=k
        self.l=l
        if l is None:
            self.l=[0,2,4]
        
        self.V={li:self._V(k,li) for li in self.l}
        self.lamU2={li:self._lamU2(k,li) for li in self.l}
        
        self.lamU_1D={li:self._lamU_1D(k,li) for li in self.l}
        
        self.N=self._N(k)
 
        
    def _V(self,k, l):
        alpha = {
            0: 0.2095,
            2: 0.5481,
            4: 1.6025
        }
        beta  = {
            0: 1.9980,
            2: 0.7224,
            4: 1.0674
        }
        return 1 + (k / alpha[l]) ** beta[l]

    def _lamU2(self,k, l):
        lamda = {
            0: [61.9058, 0, 0, 0],
            2: [35.7400, 4.4144, 1.7198, 0.9997],
            4: [22.0881, 4.5984, 2.2025, 1.4062]
        }
        alpha = {
            #0: [0.0501, 0, 0, 0],
            0: [0.0738, 0, 0, 0],  # update from Joachim
            2: [0.273, 0.15772, 0.14426, 0.14414],
            4: [0.060399, 0.1553, 0.1569, 0.15233]
        }
        beta = {
            #0: [0.0207, 0, 0, 0],
            0: [0.0505, 0, 0, 0],  # update from Joachim
            2: [0.8266, 2.4207, 4.0613, 5.422],
            4: [0.10344, 2.3370, 3.6937, 5.1617]
        }
        gamma = {
            #0: [0.6614, 0, 0, 0],
            0: [0.6416, 0, 0, 0],  # update from Joachim
            2: [1.962, 0.79153, 0.76611, 0.84826],
            4: [0.64008, 0.9307, 0.92304, 0.8899]
        }
        delta = {
            #0: [2.3045, 0, 0, 0],
            0: [1.3145, 0, 0, 0],  # update from Joachim
            2: [0.816, 0.032207, -0.26272, 0.31324],
            4: [2.2584, -0.1154, 0.04006, -0.14503]
        }

        # fix numerical problem at k=0
        k[k==0] = 1e-8

        U = alpha[l][0] * (beta[l][0] / k + gamma[l][0]) ** - delta[l][0]
        result = lamda[l][0] * U[:, None] * U

        #U = alpha[l][1:] * k[:, None] ** beta[l][1:] \
        #        * np.sin(gamma[l][1:] * k[:, None] ** delta[l][1:])
        # different parametrizations in Joachim's paper (above) vs code (below)
        U = alpha[l][1:] * k[:, None] ** delta[l][1:] \
                * np.sin(beta[l][1:] * k[:, None] ** gamma[l][1:])
        result += (lamda[l][1:] * U[:, None, :] * U[None, :, :]).sum(axis=-1)

        return result
    
    def _lamU_1D(self,k, l):
        lamda = {
            0: [61.9058, 0, 0, 0],
            2: [35.7400, 4.4144, 1.7198, 0.9997],
            4: [22.0881, 4.5984, 2.2025, 1.4062]
        }
        alpha = {
            #0: [0.0501, 0, 0, 0],
            0: [0.0738, 0, 0, 0],  # update from Joachim
            2: [0.273, 0.15772, 0.14426, 0.14414],
            4: [0.060399, 0.1553, 0.1569, 0.15233]
        }
        beta = {
            #0: [0.0207, 0, 0, 0],
            0: [0.0505, 0, 0, 0],  # update from Joachim
            2: [0.8266, 2.4207, 4.0613, 5.422],
            4: [0.10344, 2.3370, 3.6937, 5.1617]
        }
        gamma = {
            #0: [0.6614, 0, 0, 0],
            0: [0.6416, 0, 0, 0],  # update from Joachim
            2: [1.962, 0.79153, 0.76611, 0.84826],
            4: [0.64008, 0.9307, 0.92304, 0.8899]
        }
        delta = {
            #0: [2.3045, 0, 0, 0],
            0: [1.3145, 0, 0, 0],  # update from Joachim
            2: [0.816, 0.032207, -0.26272, 0.31324],
            4: [2.2584, -0.1154, 0.04006, -0.14503]
        }

        # fix numerical problem at k=0
        k[k==0] = 1e-8

        U={}
        U['lambda']=lamda[l]
        U [0]= alpha[l][0] * (beta[l][0] / k + gamma[l][0]) ** - delta[l][0]
#         result = lamda[l][0] * U[:, None] * U

        #U = alpha[l][1:] * k[:, None] ** beta[l][1:] \
        #        * np.sin(gamma[l][1:] * k[:, None] ** delta[l][1:])
        # different parametrizations in Joachim's paper (above) vs code (below)
        
        for i in np.arange(1,len(alpha[l])):
            U[i] = alpha[l][i] * k** delta[l][i] \
                * np.sin(beta[l][i] * k ** gamma[l][i])
#         result += (lamda[l][1:] * U[:, None, :] * U[None, :, :]).sum(axis=-1)

        return U

    def _N(self,k):
        """Number of modes, in Harnois-Deraps and Pen 2012."""
        L = 200

        dk = 2 * np.pi / L

        k_lo, k_hi = k - 0.5 * dk, k + 0.5 * dk

        Vk = 4 * np.pi / 3 * (k_hi**3 - k_lo**3)

        Nk = Vk * L**3 / (2 * np.pi)**3

        return Nk

    def cov_tri(self,P, l=None,k=None):
        """Covariance from trispectrum, without super-sample covariance."""
        
        if k is None:
            k=self.k
#         else:
#             self.__init__(l=l,k=k)

        assert k.ndim == 1
        assert P.ndim == 1 

        ang_proj_coef = {  # int_0^2pi Legendre_l(cos theta) d theta / 2pi
            0: 1,
            2: 1 / 4,
            4: 9 / 64
        }

        CG = 2 * P**2 / self.N  # Gaussian cov (of Harnois-Deraps&Pen)

        if l is None:
            result=0
            result_diag = 0
            for l in 0, 2, 4:
                VCG = self.V[l] * CG
                result += (2*l+1) * ang_proj_coef[l] \
                        * self.lamU2[l] * np.sqrt(VCG[:, None] * VCG)
                result_diag += VCG
                

        else:  # debugging: check each ell
            CGl = CG * 4 * np.pi
            VCGl = self.V[l] * CGl
            result = self.lamU2[l] * np.sqrt(VCGl[:, None] * VCGl)
            result_diag = VCGl

        ## debugging: if you want to add the Gaussian cov (of Harnois-Deraps&Pen)
        #result[np.diag_indices_from(result)] = result_diag

        return result
    
    def cov_tri_zkernel(self,P, l=None,k=None,z_kernel=None,chi=None):
        """Covariance from trispectrum, without super-sample covariance."""
        
        if k is None:
            k=self.k
#         else:
#             self.__init__(l=l,k=k)

        assert k.ndim == 1
#         assert P.ndim == 1  #we are using multiple redshift

        ang_proj_coef = {  # int_0^2pi Legendre_l(cos theta) d theta / 2pi
            0: 1,
            2: 1 / 4,
            4: 9 / 64
        }
        
        sig_F=z_kernel#/self.Ang_PS.clz['chi']**2
        sig_F=np.sqrt(sig_F) #kernel is function of l as well due to spin factors
        
        self.N_chi=np.outer(1./chi**2,self.N)
        CG = P*sig_F.T*np.sqrt(2. / self.N_chi)  # sqrt of Gaussian cov (of Harnois-Deraps&Pen)... FIXME: need to account for this

        
        if l is None:
            result=0
            for l in 0, 2, 4: #sum over multipoles
                lamb=self.lamU_1D[l]['lambda']
                result_l=0
                CGl=CG*np.sqrt(self.V[l]*(2*l+1) * ang_proj_coef[l])
                for i in np.arange(len(lamb)): #sum over eigenmodes given l
                    CGi=self.lamU_1D[l][i]*CGl
                    result_l+=lamb[i] * np.dot(CGi.T,CGi)
   
#                 print(l,np.all(result_l.T==result_l))
                result+=result_l
#                 print(np.all(result.T==result))
        return result

    def cov2cor(self,C):
        assert C.ndim == 2
        norm = np.sqrt(C.diagonal())
        return C / norm[:, None] / norm
