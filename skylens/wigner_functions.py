"""
This file contains functions to compute wigner matrices used in wigner transform (power spectra to correlation functions) 
and wigner_3j matrices used in window function calculations. 
"""

import numpy as np
from scipy.special import binom,jn,loggamma
from scipy.special import eval_jacobi as jacobi
from multiprocessing import Pool,cpu_count
from functools import partial
import sparse
from sympy import Integer
from sympy import sqrt as sy_sqrt
from sympy import exp as sy_exp
from sympy import log as sy_log

from mpmath import exp as mp_exp
from mpmath import log as mp_log
from sympy.physics.wigner import wigner_3j


def wigner_d(m1,m2,theta,l,l_use_bessel=1.e4):
    """
    Function to compute wigner small-d matrices used in wigner transforms.
    """
    l0=np.copy(l)
    if l_use_bessel is not None:
    #FIXME: This is not great. Due to a issues with the scipy hypergeometric function,
    #jacobi can output nan for large ell, l>1.e4
    # As a temporary fix, for ell>1.e4, we are replacing the wigner function with the
    # bessel function. Fingers and toes crossed!!!
    # mpmath is slower and also has convergence issues at large ell.
    #https://github.com/scipy/scipy/issues/4446
    
        l=np.atleast_1d(l)
        x=l<l_use_bessel
        l=np.atleast_1d(l[x])
    k=np.amin([l-m1,l-m2,l+m1,l+m2],axis=0)
    a=np.absolute(m1-m2)
    lamb=0 #lambda
    if m2>m1:
        lamb=m2-m1
    b=2*l-2*k-a
    d_mat=(-1)**lamb
    d_mat*=np.sqrt(binom(2*l-k,k+a)) #this gives array of shape l with elements choose(2l[i]-k[i], k[i]+a)
    d_mat/=np.sqrt(binom(k+b,b))
    d_mat=np.atleast_1d(d_mat)
    x=k<0
    d_mat[x]=0

    d_mat=d_mat.reshape(1,len(d_mat))
    theta=theta.reshape(len(theta),1)
    d_mat=d_mat*((np.sin(theta/2.0)**a)*(np.cos(theta/2.0)**b))
    x=d_mat==0
    d_mat*=jacobi(k,a,b,np.cos(theta)) #l
    d_mat[x]=0
    
    if l_use_bessel is not None:
        l=np.atleast_1d(l0)
        x=l>=l_use_bessel
        l=np.atleast_1d(l[x])
#         d_mat[:,x]=jn(m1-m2,l[x]*theta)
        d_mat=np.append(d_mat,jn(m1-m2,l*theta),axis=1)
    return d_mat

def wigner_d_parallel(m1,m2,theta,l,ncpu=None,l_use_bessel=1.e4):
    """
    Compute wigner small-d matrix in parallel.
    """
    if ncpu is None:
        ncpu=cpu_count()
    p=Pool(ncpu)
    d_mat=np.array(p.map(partial(wigner_d,m1,m2,theta,l_use_bessel=l_use_bessel),l))
    p.close()
    p.join()
    return d_mat[:,:,0].T

def wigner_3j_000(j_1,j_2,j_3,m_1,m_2,m_3): 
    """
    This is the function we use to compute wigner_3j matrices for the case of 
    m1=m2=m3=0. Algorithm from Hivon+ 2002
    """
    J=j_1+j_2+j_3
    a1 = j_1 + j_2 - j_3
    a2 = j_1 - j_2 + j_3
    a3 = -j_1 + j_2 + j_3
    x=a1<0
    x=np.logical_or(x,a2<0)
    x=np.logical_or(x,a3<0)
    x=np.logical_or(x,J%2==1)
#     print(x)
    
    logwj=log_factorial(J/2)
    logwj-=log_factorial(J/2-j_1)
    logwj-=log_factorial(J/2-j_2)
    logwj-=log_factorial(J/2-j_3)
    logwj-=0.5*log_factorial(J+1)
    logwj+=0.5*log_factorial(J-2*j_1)
    logwj+=0.5*log_factorial(J-2*j_2)
    logwj+=0.5*log_factorial(J-2*j_3)
    logwj[x]=-308
    wj=(-1)**(np.int32(J/2))*np.exp(logwj)  #0, when J/2 is not int
    wj[x]=0
#     x=J%2==1 #already applied in calling functions
#     wj[x]*=0
    return np.real(wj)

def log_factorial(n):
    return loggamma(n+1)

"""
Following are helper functions for a recursive algorithm implemented below in wig3j_recur.
"""
def A_J(j,j2,j3,m2,m3):
    out=np.float64(j**2-(j2-j3)**2)
    x=j**2<(j2-j3)**2
    out[x]=0
    out=out**.5
    out*=((j2+j3+1)**2-j**2)**.5
    out*=(j**2-(m2+m3)**2)**.5
    return out

def B_J(j,j2,j3,m2,m3):
    out=np.float64((m2+m3)*(j2*(j2+1)-j3*(j3+1)))
    out-=(m2-m3)*j*(j+1)
    out*=2*j+1
    return out

def X_Np1(j,j2,j3,m2,m3): #X_n+1
    return j*A_J(j+1,j2,j3,m2,m3)

def X_N(j,j2,j3,m2,m3): #X_n+1
    return B_J(j,j2,j3,m2,m3)

def X_Nm1(j,j2,j3,m2,m3): #X_n+1
    return (j+1)*A_J(j,j2,j3,m2,m3)


def wig3j_recur(j1,j2,m1,m2,m3,j3_outmax=None):
    """
    Using recursion relation to compute wigner matrices. Works well. Validated with the sympy function.
    Reference is Luscombe, James H. 1998, Physical Review E.
    """
#     assert m3==-m1-m2
    
    if (abs(m1) > j1) or (abs(m2) > j2) or m1+m2+m3!=0:
        return 0

    j3_min=np.absolute(j1-j2)
    j3_max=j1+j2+1 #j3_max is j1+j2, +1 for 0 indexing
    
    j3=np.arange(j3_max)
    
    if j3_outmax is None:
        j3_outmax=j3_max
    
    wig_out=np.zeros(max(j3_max,j3_outmax))
    
    if j3_min>j3_outmax:
        return wig_out[:j3_outmax]#.reshape(1,1,j3_outmax)

    wig_out[j3_min]=1#wigner_3j(j1,j2,j3_min,m1,m2,m3)
    if j3_min==0: #in this case the recursion as implemented doesnot work (all zeros). use sympy function.
        wig_out[j3_min]=wigner_3j(j1,j2,j3_min,m1,m2,m3)
        wig_out[j3_min+1]=wigner_3j(j1,j2,j3_min+1,m1,m2,m3) #not strictly needed when j3_min>0
                
    x_Np1=X_Np1(j3,j2,j1,m1,m2)*-1 #j==j3
    x_N=X_N(j3,j2,j1,m1,m2) #j==j3
    x_Nm1=X_Nm1(j3,j2,j1,m1,m2) #j==j3
    
    for i in np.arange(j3_min,j3_max):
        if x_Np1[i]==0:
            continue
        wig_out[j3[i+1]]=x_Nm1[i]*wig_out[j3[i-1]]+x_N[i]*wig_out[j3[i]]
        wig_out[j3[i+1]]/=x_Np1[i]    
        
    Norm=np.sum((2*j3+1)*(wig_out[:j3_max]**2))
    wig_out/=Norm**.5
    
    if np.sign(wig_out[j1+j2])!=np.sign((-1)**(j3_min)):
        wig_out*=-1
    
    #FIXME: this commented out part is for validation against sympy. Should implement it properly as test.
#     if j3_min==0 and not np.isclose(Norm,1): #in this case we started recursion with exact values at j3_min and hence norm should be 1.
#         print("Norm problem: ",j1,j2,j3_min,Norm,wig_out[:j3_max],)
#     else:
#         tt=wigner_3j(j1,j2,j3_min+1,m1,m2,m3) # This is only for testing
#         tt=np.float(tt)
#         if not np.isclose(wig_out[j3_min+1],tt):
#             print('j3min+1 comparison problem: ',j1,j2,j3_min+1,tt,wig_out[j3_min+1])
        
        
#     xxi=np.random.randint(j3_min+1,j3_max-1) #randomly compare a value with sympy calculation
#     wig_out2=wigner_3j(j1,j2,j3[xxi],m1,m2,m3)
#     try:
#         wig_out2=wig_out2.evalf()
#         print('random test ',xxi,wig_out[xxi],wig_out2)
#     except Exception as err:
#         print('warning, random test failed', err)
#         pass


    return wig_out[:j3_outmax]#.reshape(j3_outmax,1,1)


def wigner_3j_2(j_1, j_2, j_3, m_1, m_2, m_3): 
    #this and some helper functions below are a copy-paste of sympy function. We use it for testing.
    r"""
    Calculate the Wigner 3j symbol `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.

    INPUT:

    -  ``j_1``, ``j_2``, ``j_3``, ``m_1``, ``m_2``, ``m_3`` - integer or half integer

    OUTPUT:

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0

    It is an error to have arguments that are not integer or half
    integer values::

        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer

    NOTES:

    The Wigner 3j symbol obeys the following symmetry rules:

    - invariant under any permutation of the columns (with the
      exception of a sign change where `J:=j_1+j_2+j_3`):

      .. math::

         \begin{aligned}
         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
          &=\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\
          &=\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\
          &=(-1)^J \operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)
         \end{aligned}

    - invariant under space inflection, i.e.

      .. math::

         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
         =(-1)^J \operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 additional symmetries based on
      the work by [Regge58]_

    - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation

    - zero for `m_1 + m_2 + m_3 \neq 0`

    - zero for violating any one of the conditions
      `j_1 \ge |m_1|`,  `j_2 \ge |m_2|`,  `j_3 \ge |m_3|`

    ALGORITHM:

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    REFERENCES:

    .. [Regge58] 'Symmetry Properties of Clebsch-Gordan Coefficients',
      T. Regge, Nuovo Cimento, Volume 10, pp. 544 (1958)

    .. [Edmonds74] 'Angular Momentum in Quantum Mechanics',
      A. R. Edmonds, Princeton University Press (1974)

    AUTHORS:

    - Jens Rasch (2009-03-24): initial version
    """
    if int(j_1 * 2) != j_1 * 2 or int(j_2 * 2) != j_2 * 2 or \
            int(j_3 * 2) != j_3 * 2:
        raise ValueError("j values must be integer or half integer")
    if int(m_1 * 2) != m_1 * 2 or int(m_2 * 2) != m_2 * 2 or \
            int(m_3 * 2) != m_3 * 2:
        raise ValueError("m values must be integer or half integer")
    if m_1 + m_2 + m_3 != 0:
        return 0
    prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    m_3 = -m_3
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return 0
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return 0
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return 0
    if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
        return 0

    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2),
                  j_3 + abs(m_3))
#     _calc_factlist(int(maxfact))

    argsqrt = Integer(_Factlist[int(j_1 + j_2 - j_3)] *
                     _Factlist[int(j_1 - j_2 + j_3)] *
                     _Factlist[int(-j_1 + j_2 + j_3)] *
                     _Factlist[int(j_1 - m_1)] *
                     _Factlist[int(j_1 + m_1)] *
                     _Factlist[int(j_2 - m_2)] *
                     _Factlist[int(j_2 + m_2)] *
                     _Factlist[int(j_3 - m_3)] *
                     _Factlist[int(j_3 + m_3)]) / \
        _Factlist[int(j_1 + j_2 + j_3 + 1)]
    ressqrt = sy_sqrt(argsqrt)
#     print('sqrt',ressqrt.evalf())
    if ressqrt.is_complex:
        ressqrt = ressqrt.as_real_imag()[0]

    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * \
            _Factlist[int(ii + j_3 - j_1 - m_2)] * \
            _Factlist[int(j_2 + m_2 - ii)] * \
            _Factlist[int(j_1 - ii - m_1)] * \
            _Factlist[int(ii + j_3 - j_2 + m_1)] * \
            _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + Integer((-1) ** ii) / den
#         print(ii,sumres.evalf(),sy_log(den).evalf(24))
#     print(sumres.evalf())
    res = ressqrt * sumres * prefid
    return res

_Factlist=[1,1]
def _calc_factlist(nn):
    r"""
    Function calculates a list of precomputed factorials in order to
    massively accelerate future calculations of the various
    coefficients.

    INPUT:

    -  ``nn`` -  integer, highest factorial to be computed

    OUTPUT:

    list of integers -- the list of precomputed factorials

    EXAMPLES:

    Calculate list of factorials::

        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    """
    global _Factlist
    if nn >= len(_Factlist):
        for ii in range(len(_Factlist), int(nn + 1)):
            _Factlist.append(_Factlist[ii - 1] * ii)
    return _Factlist[:int(nn) + 1]

_calc_factlist(1000)

def calc_factlist(nn):
    return _calc_factlist(nn)


##############################################################################################
