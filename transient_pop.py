#Tim van der Vuurst, 29/05

#imports
import numpy as np
import matplotlib.pyplot as plt
from sjoert.stellar import flux2mag
import pandas as pd
import forced_phot as fp #import flux_jy, chi2, gaussian, chi2_peak_finder
from astropy.io import ascii
from scipy.constants import c,h,k
import os

#==
#Load in the data
dpath = r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data'
fullcatalogue = pd.read_csv(os.path.join(dpath,'transient_catalogue.txt'))
catalogue = fullcatalogue[fullcatalogue['in_selection']].iloc[:,:-2]

#==

param_label_dict = {'sigma_rise':r'$\sigma_{\text{rise}}$','tau_dec':r'$\tau_{\text{dec}}$',
                    'log10_sigma_rise':r'$\sigma_{\text{rise}}$','log10_tau_dec':r'$\tau_{\text{dec}}$',
                    'F_p':r'F$_{\text{peak}}$','peak_g':r'F$_{\text{p,g}}$','peak_r':r'F$_{\text{p,r}}$'}

def param_scatterplot(data,param1,param2,colorby = None,colorby_label='',colorbar=True,logspace=True,mask=0,boundaries=True):
    param1data = data[param1].to_numpy()
    param2data = data[param2].to_numpy()
    
    if type(mask) == int: mask = np.ones(param1data.shape).astype(bool)
    param1data = param1data[mask]
    param2data = param2data[mask]

    try:
        #if the parameter has an error (which most always the case for relevant parameters) get that as well
        if logspace:
            param1data_err = data['log10_'+param2+'_err'][mask]
        else:
            param1data_err = data[+param2+'_err'][mask]
            # param1data_err = 1/np.log(10) * param_df[param1+'_err'] / param1data
        param1errFlag = True
    except KeyError:
        param1errFlag = False

    try:
        #if the parameter has an error (which most always the case for relevant parameters) get that as well
        if logspace:
            param2data_err = data['log10_'+param2+'_err'][mask]
        else:
            param2data_err = data['log10_'+param2+'_err'][mask]
        # param2data_err = 1/np.log(10) * param_df[param2+'_err'] / param2data
        param2errFlag = True
    except KeyError:
        param2errFlag = False

    # param1errFlag = False
    # param2errFlag = False
    fig,ax = plt.subplots(figsize=(8,8))
    if param1errFlag:
        if param2errFlag:
            if colorby is not None:
                if colorbar:
                    # ax.errorbar(param1data,param2data,xerr=param1data_err,yerr=param2data_err,fmt=',',capsize=1,color='black')
                    colorscatter = ax.scatter(param1data,param2data,c=np.clip(colorby[mask],-4,4),cmap='viridis',s=1,zorder=10)
                    clbr = fig.colorbar(colorscatter)
                    clbr.set_label(colorby_label,fontsize=14)
                else:
                    colorby = colorby[mask]
                    colordict = {"AGN":'gray',"SN":"green","TDE":'blue','Unknown':'red'}
                    markerdict = {"AGN":'s',"SN":"o","TDE":'P','Unknown':'p'}
                    for elem in np.unique(colorby):
                        colormask = (colorby == elem)
                        if elem == 'Unknown':
                            ax.scatter(param1data[colormask],param2data[colormask],edgecolors=colordict[elem],marker=markerdict[elem],s=10,zorder=1,label=elem,alpha=0.5,facecolors='none')
                        else:
                            ax.scatter(param1data[colormask],param2data[colormask],edgecolors=colordict[elem],marker=markerdict[elem],s=15,zorder=10,label=elem,facecolors='none')

            else:
                ax.errorbar(param1data,param2data,xerr=param1data_err,yerr=param2data_err,fmt=',',capsize=4,color='black')
        else:
            ax.errorbar(param1data,param2data,xerr=param1data_err,fmt='.',capsize=2)
    else:
        if param2errFlag:
            ax.errorbar(param1data,param2data,yerr=param2data_err,fmt='.',capsize=2)
        else:
            ax.scatter(param1data,param2data,s=4)
    ax.set_xlabel(param_label_dict[param1],fontsize=12)
    ax.set_ylabel(param_label_dict[param2],fontsize=12)
    #plt.savefig(param1+'_vs_'+param2+'.png',dpi=600)
    ax.set(xscale="log",yscale='log')
    ax.grid()
    if colorbar == False:
        ax.legend()
    if boundaries:
        ax.vlines(150,0,500,colors='black',linestyles='dashed',zorder=10,alpha=0.75)
        ax.hlines(500,0,150,colors='black',linestyles='dashed',zorder=10,alpha=0.75)
    plt.show()



def scatter_against_mbh(param: str,catalogue = catalogue) -> None:
    mbh_indx = np.nonzero(catalogue['MBH'].values)

    paramdata = catalogue[param].values[mbh_indx]
    paramerr = catalogue[param+'_err'].values[mbh_indx]
    MBH = catalogue['MBH'].values[mbh_indx]

    plt.figure()
    plt.ylabel(param_label_dict[param],fontsize=12)
    plt.xlabel(r'$\log_{10}\left(\text{M}_{\text{BH}}\right)$ $[M_{\odot}]$ ',fontsize=12)
    plt.errorbar(MBH,paramdata,yerr=paramerr,fmt='.',c='black')
    plt.vlines(8,0,np.max(paramdata)*1.05,zorder=10,colors='red',linestyle='dashed')
    plt.show()


# scatter_against_mbh('log10_sigma_rise',catalogue) 


def mbh_histogram(density=False,save = True,catalogue=catalogue) -> None:
    mbh_indx = np.nonzero(catalogue['MBH'].values)
    catalogue = catalogue.iloc[mbh_indx] #filter out only the ones with known mbh data
    classes = catalogue['classification'].unique()

    plt.figure()
    for cl in classes:
        cl_mask = (catalogue['classification'].values == cl) #picks out all the units with given MBH of type AGN
        cl_mbh = catalogue['MBH'][cl_mask].values

        # bins = np.linspace(np.min(class_info['MBH'][mbhindx]),np.max(class_info['MBH'][mbhindx]),20)
        bins = 20
        bins,counts = np.histogram(cl_mbh,bins=bins,density=density)
        # print(bins.shape,counts.shape)
        plt.step(counts[:-1],bins,label=cl)
    plt.xlabel(r'M$_{\text{BH}}$ [$M_{\odot}]$')
    if density:
        plt.ylabel('Density')
    else:
        plt.ylabel('Counts')
    plt.legend()
    if not save:
        plt.show()
    else:
        if density:
            savename = os.path.join(dpath,'BH_histogram_density.png')
        else:
            savename = os.path.join(dpath,'BH_histogram_counts.png')
        plt.savefig(savename)
        plt.close()

# mbh_histogram(False)
# mbh_histogram(True)

class_info = ascii.read(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\class_info.dat')
#sort the class_info by ztf_id
perm_indx = class_info['ztf_id'].argsort()
class_info = class_info[perm_indx]
#set empty values to Unknown instead of 0
class_info['classification'] = class_info['classification'].filled('Unknown')
class_info['classification'][0] = 'Unknown'
class_info = pd.DataFrame(np.array(class_info))

def mbh_cumul_hist(density=True,save = True,catal=class_info) -> None:
    mbh_indx = np.nonzero(catal['MBH'].values)
    cat = catal.copy() #keep the original
    cat = cat.iloc[mbh_indx] #filter out only the ones with known mbh data
    agnmask = (cat['classification'].values == 'AGN') 
    fullcat = fullcatalogue.copy()
    fullcat = fullcat.iloc[np.nonzero(fullcatalogue['MBH'].values)]
    strongmask = (fullcat['classification'].values == 'AGN') * fullcat['in_selection'].values

    mbh_strong = fullcat['MBH'][strongmask].values
    mbh_other = cat['MBH'][agnmask].values

    n_bins = np.histogram_bin_edges(cat['MBH'].values,bins=30)

    fig,ax = plt.subplots()
    n, bins, patches = ax.hist(mbh_strong, n_bins, density=density, histtype='step',
                           cumulative=True, label='Extreme AGN flare')
    ax.hist(mbh_other, n_bins, density=density, histtype='step',
                           cumulative=True, label='Normal AGN flare')

    ax.set_xlabel(r'$\log_{10}\left(\text{M}_{\text{BH}}/M_{\odot}\right)$')
    if density: ylabel = "CDF"
    else: ylabel = 'Cumulative sum'
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    plt.show()

mbh_cumul_hist(catal=class_info)
mbh_cumul_hist(False,catal=class_info)