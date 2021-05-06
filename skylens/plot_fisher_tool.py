import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.stats import norm
import copy

iblack  = list(np.array([  0,  0,  0])/256.)
iblue   = list(np.array([ 33, 79,148])/256.)
ired    = list(np.array([204,  2,  4])/256.)
iorange = list(np.array([255,169,  3])/256.)
igray   = list(np.array([130,130,120])/256.)
igreen  = list(np.array([  8,153  ,0])/256.)

colors=[iblue,ired,iorange,igray,igreen]

par_labels={'As':r'$A_s$','Ase9':r'$A_s\times 10^9$', 'Om':r'$\Omega_m$','w':r'$w_0$','wa':r'$w_a$', 'mnu':r'$m_\nu$'}


def cal_variance_FlatDist(a,b):
    return (b-a)**2/12.

def tune_xtick(ax,fontsize=15):
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
    return ax

def tune_ytick(ax,fontsize=15):
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
    for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
    return ax

class fisher_tool():
    def __init__(self,Fishers,par_cen,par_prior_variance=None,pars=None,par_labels=par_labels,ax_lim_nsigma=3,fisher_titles={},
                print_fom=True,print_par_error=True):
        self.init_fisher(Fishers=Fishers,par_prior_variance=par_prior_variance,pars=pars)
        self.par_cen       = par_cen
        self.ax_lim_nsigma=ax_lim_nsigma
        self.fisher_titles=fisher_titles
        self.par_labels=par_labels
        self.print_par_error=print_par_error
        self.print_fom=print_fom
        self.cal_Cov_par()
        self.cal_Par_err()
        self.cal_Corr_par()
        self.set_alpha_ellip(nsigma_2Dposterior=1)  # 1 : 68.3% 2D contour, 2 : 95.4%, 3 : 99.7%
        
    def init_fisher(self,Fishers,par_prior_variance=None,pars={}):
        self.Npars = {}
        self.pars={}
        for i in Fishers.keys():
            self.Npars[i]=len(Fishers[i])
            self.pars[i]=list(pars[i])
            
        if par_prior_variance is None:
            self.Fishers = Fishers
        else:
            self.Fishers_0prior = copy.deep_copy(Fishers)
            self.Fisher_prior = np.diag(1./np.array(par_prior_variance))
            for i in Fishers.keys():
                self.Fishers[i] = self.Fisher_0prior[i] + self.Fisher_prior
    
    def cal_Cov_par(self):
        self.Cov_par={}
        for i in self.Fishers.keys():
            self.Cov_par[i] = np.linalg.inv(self.Fishers[i])
    
    def cal_Par_err(self):
        self.par_sigma1D={}
        for i in self.Fishers.keys():
            self.par_sigma1D[i] = np.sqrt(np.diag(self.Cov_par[i]))
    
    def cal_Corr_par(self):
        #compute parameter correlation coefficient
        self.Corr_par={}
        for i in self.Fishers.keys():
            diag=np.diag(self.Cov_par[i])
            self.Corr_par[i]=self.Cov_par[i]/np.sqrt(np.outer(diag,diag))
#             self.Corr_par[i] = np.zeros([self.Npar[i],self.Npar[i]])
#             for i in range(self.Npar):
#                 for j in range(self.Npar):
#                     self.Corr_par[i][j]=self.Cov_par[i][j]/np.sqrt(self.Cov_par[i][i]*self.Cov_par[j][j])

        
    def get_2D_ellips_info(self,id_1,id_2,fish_id): # id number respect to Fisher matrix id

        Cov_par2D = self.Cov_par[fish_id][[id_1,id_2],:][:,[id_1,id_2]]
        ellip_sigma2, v = np.linalg.eig(Cov_par2D)
        id_rank = np.argsort(ellip_sigma2)[::-1]
        ellip_sigma2 = ellip_sigma2[id_rank]  ;  v = v[id_rank]    # change the ranking of eigenvalues and corr. eigenvectors (bigger first)
        a = np.sqrt(ellip_sigma2[0])
        b = np.sqrt(ellip_sigma2[1])
        theta_rad = np.arctan2(2*Cov_par2D[0,1],Cov_par2D[0,0]-Cov_par2D[1,1])/2.
        theta_deg = theta_rad*180/np.pi
        area=a*b #this is same as sqrt of determinant. Area of ellipse is pi*a*b
        return a, b, theta_deg,area
    
    def set_alpha_ellip(self,nsigma_2Dposterior):
        sigma_alpha = {1:1.52,2:2.48,3:3.44}  #Table5 of arXiv0906.4123
        self.alpha  = sigma_alpha[nsigma_2Dposterior]

    def plot_ellips2D(self,ax,par_1,par_2,alpha=0.2,ls="-",fid=None,axlim=None,text_size=15):
        axlim_1={}
        axlim_2={}
        mu_1    = 0 if self.par_cen.get(par_1) is None else self.par_cen[par_1]
        mu_2    = 0 if self.par_cen.get(par_2) is None else self.par_cen[par_2]
        text_x=0.75-(text_size-18)*0.02

        text_y=0.86
        for fish_id in self.Fishers.keys():
            id_1=self.pars[fish_id].index(par_1)
            id_2=self.pars[fish_id].index(par_2)
            color=colors[fish_id%len(colors)]
            
            sigma_1 = self.par_sigma1D[fish_id][id_1]
            sigma_2 = self.par_sigma1D[fish_id][id_2]

            a, b, theta_deg,area = self.get_2D_ellips_info(id_1,id_2,fish_id)
            ells = Ellipse((mu_1, mu_2), 2*a*self.alpha, 2*b*self.alpha, theta_deg)

            ax.add_artist(ells)
            ells.set_clip_box(ax.bbox)
            ells.set_edgecolor(tuple(color+[1.]))
            ells.set_facecolor(tuple(color+[alpha]))

            ells.set_linestyle(ls)
            ells.set_linewidth(2)

            #corr_coe="%.2f"%self.Corr_par[fish_id][id_1][id_2]
            #ax.text(0.75,0.82, corr_coe ,transform=ax.transAxes,color=color,fontsize=15)
#             area_t=np.format_float_scientific((1./area),precision=1,exp_digits=1)
            area_t="%d"%(1./area)
            if self.print_fom:
                ax.text(text_x+0.05*(4-len(area_t)),text_y, area_t ,transform=ax.transAxes,color=color,fontsize=text_size,zorder=10,
                   bbox=dict(facecolor='white', alpha=0.5,lw=0,pad=0))
#             text_y=0.11
            text_y-=0.15

            if fid is not None:
                ax.axvline(x=fid[0],color='silver',zorder=-1)
                ax.axhline(y=fid[1],color='silver',zorder=-2)

            axlim_1[fish_id]=[mu_1-sigma_1*self.ax_lim_nsigma,mu_1+sigma_1*self.ax_lim_nsigma] if axlim.get(par_1) is None else axlim[par_1]
            axlim_2[fish_id]=[mu_2-sigma_2*self.ax_lim_nsigma,mu_2+sigma_2*self.ax_lim_nsigma] if axlim.get(par_2) is None else axlim[par_2]
        
        axv1=np.array(list(axlim_1.values())).flatten()
        axv2=np.array(list(axlim_2.values())).flatten()
        ax.set_xlim([axv1.min(),axv1.max()])
        ax.set_ylim([axv2.min(),axv2.max()])
        
        ax.set_xticks(np.around([mu_1/2.+axv1.min()/2,mu_1,mu_1/2.+axv1.max()/2.],decimals=3))
        ax.set_yticks(np.around([mu_2/2.+axv2.min()/2,mu_2,mu_2/2.+axv2.max()/2.],decimals=3))
#         ax.set_yticks(np.around([mu_2-sigma_2*2,mu_2,mu_2+sigma_2*2],decimals=2))
        return ax
    
    def plot_gauss1D(self,ax,par,color=iblue,ls='-',fid=None,axlim={}):
        
        mu    = 0 if self.par_cen.get(par) is None else self.par_cen[par]
        axlim={}
        for fish_id in self.Fishers.keys():
            id_1=self.pars[fish_id].index(par)
            color=colors[fish_id%len(colors)]
            sigma = self.par_sigma1D[fish_id][id_1]

            x     = np.linspace(mu-3*sigma, mu+3*sigma, 300)
            y     = norm.pdf(x,mu,sigma)
            x_1s  = np.linspace(mu-1*sigma, mu+1*sigma, 300)
            y_1s  = norm.pdf(x_1s,mu,sigma)

            ax.plot(x,y,linestyle=ls,linewidth=2,color=color)
            ax.fill_between(x_1s,[0.]*len(x_1s), y_1s, facecolor=color, alpha=0.2)

            if fid is not None:
                ax.axvline(x=fid,color='silver',zorder=-1)

            axlim[fish_id]=[mu-sigma*self.ax_lim_nsigma,mu+sigma*self.ax_lim_nsigma] if axlim.get(par) is None else axlim[par]

    #         if axlim is not None: #FIXME: axlim can be a property of the class. if none, it can be set based on mu,sigma
        axv=np.array(list(axlim.values())).flatten()
        ax.set_xlim(axv.min(),axv.max())
        ax.set_xticks(np.around([mu/2.+axv.min()/2.,mu,mu/2.+axv.max()*.5],decimals=3))
            
            
    def plot_fish(self,pars=[],par_labels=None,par_axlim={},fontsize=None):
        if par_labels is None:
            par_labels=self.par_labels
            
        Ndim = len(pars)
        fig, ax = plt.subplots(Ndim,Ndim,figsize=(2*Ndim+2,2*Ndim+1))
        fig.subplots_adjust(left=0.08, bottom=0.07, right=0.98, top=0.98 ,hspace=0.05,wspace=0.05)
        
        if fontsize is None:
            label_fontsize=15*np.sqrt(Ndim/2.)
            text_fontsize=15*np.sqrt(Ndim/2.)
            tick_label_fontsize=label_fontsize*.85
            
        plt.rc('text', usetex=True)
        plt.rc('font',size=text_fontsize)

        ytext=.9
        for i,par_i in enumerate(pars):
            for j,par_j in enumerate(pars):
                if i < j:
                    ax[i,j].axis('off')
                    continue
                elif i==j:
                    self.plot_gauss1D(ax[i,j],par_i,color=iblue,axlim=par_axlim)
                    for fish_id in self.Fishers.keys():
                        color=colors[fish_id%len(colors)]
                        id_1=self.pars[fish_id].index(par_i)
                        sigma = self.par_sigma1D[fish_id][id_1]
                        sigma2=0
                        decimals=2
                        while sigma2==0:
                            sigma2=np.around(sigma,decimals=decimals)
                            decimals+=1
                        if self.print_par_error:
                            ax[0,-2].text(x=1.1,y=ytext,s=self.par_labels[par_i]+'$='+str(sigma2)+'$',
                                     color=color,fontsize=text_fontsize)
                        ytext-=0.3
                    ax[i,j].set_yticks([])
                    ax[i,j].set_yticklabels('')
                    if i < Ndim-1:
                        ax[i,j].set_xticklabels('')
                    if i == Ndim-1:
                        tune_xtick(ax[i,j],fontsize=tick_label_fontsize)
                        ax[i,j].set_xlabel(par_labels[par_i],fontsize=label_fontsize)

                else:
                    self.plot_ellips2D(ax[i,j],par_j,par_i,alpha=0.1,axlim=par_axlim,text_size=tick_label_fontsize)

                    if j!=0:
                        ax[i,j].set_yticklabels('')
                    if i!=Ndim-1:
                        ax[i,j].set_xticklabels('')
                    if j==0:
                        tune_ytick(ax[i,j],fontsize=tick_label_fontsize)
                        ax[i,j].set_ylabel(par_labels[par_i],fontsize=label_fontsize)

                    if i==Ndim-1:
                        tune_xtick(ax[i,j],fontsize=tick_label_fontsize)
                        ax[i,j].set_xlabel(par_labels[par_j],fontsize=label_fontsize)
            
                ax[i,j].tick_params(direction='out', length=8, color='k', zorder=-1)
        ytext=.9
        for fish_id in self.Fishers.keys():
            color=colors[fish_id%len(colors)]
            ax[0,1].text(x=.1,y=ytext,s=self.fisher_titles[fish_id],color=color,fontsize=text_fontsize)
            ytext-=0.2
        return fig


