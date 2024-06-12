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

#Physical functions
def BB(nu,T):
    #Blackbody spectrum for a certain frequency given in Hz, not an array of values
    factor = 2*h*np.power(nu,3)/(c**2)
    exponent = (h*nu)/(k*T)
    return factor /(np.exp(exponent)-1)

def BB_ratio(T,v,v_0):
    #Ratio for two blackbodies at different frequencies given a temperature.
    return BB(v,T)/BB(v_0,T)
#==

#Bandpass filters central frequencies
#Source: http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=Palomar&gname2=ZTF&asttype=
g_center = c/ (4746.48 * 1e-10)
r_center = c / (6366.38 * 1e-10)
i_center = c / (7867.41 * 1e-10)

#==
#load in data

dpath = r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data'

param_path = os.path.join(dpath,'params_good.txt')
param_df = pd.read_csv(param_path, sep=",")

refmag_path = os.path.join(dpath,'refmags_good.txt')
refmag_df = pd.read_csv(refmag_path, sep=",")

class_file_path =os.path.join(dpath,'class_info.dat')
class_info = ascii.read(class_file_path)

#function to cut out bad fits
def quality_cuts(params: pd.DataFrame = param_df) -> np.array:
    bad_log10sigma_error_mask = params['log10_sigma_rise_err'] < 0.2 
    bad_log10tau_error_mask = params['log10_tau_dec_err'] <  0.2
    bad_log10fp_mask = params['log10_F_p_err'] < 0.2

    return np.array(bad_log10sigma_error_mask * bad_log10tau_error_mask * bad_log10fp_mask) 

def temp_correct(params: pd.DataFrame = param_df) -> tuple:
    nu_0 = params['nu_0'].values
    T = params['T'].values
    green_correction = BB_ratio(T,g_center,nu_0)
    red_correction = BB_ratio(T,r_center,nu_0)
    return green_correction,red_correction

#function to select region of interesting flares
def pop_cuts(params: pd.DataFrame = param_df) -> tuple[np.ndarray]:
    dmag_mask = np.logical_or((params['dmg'].values <= 1),params['dmr'].values <= 1) #create the delta m selection based on if either r or g has dmag < 1
    sigma_rise_mask = params['sigma_rise'].values <= 100 #changed from 150
    tau_dec_mask = params['tau_dec'].values <= 500
    
    fullmask = dmag_mask * sigma_rise_mask * tau_dec_mask
    return fullmask, dmag_mask #return both the full selection as only the strong flare selection

if __name__ == '__main__':
    #perform temperature correction
    gcorrect,rcorrect = temp_correct(param_df)
    Fp = param_df['F_p'].values #in micro Jansky!
    dmg = flux2mag(Fp *1e-6 * gcorrect) - refmag_df['refmag_g'].values #calculate delta m in the g band
    dmr = flux2mag(Fp *1e-6 * rcorrect) - refmag_df['refmag_r'].values #calculate delta m in the r band
    #add to DataFrame
    param_df['dmg'] = dmg
    param_df['dmr'] = dmr

    #merge param_df and refmag_df into one catalogue, using only 1 ZTF_ID column
    catalogue = pd.concat([param_df,refmag_df.iloc[:,1:]],axis=1)
    catalogue.reset_index(inplace=True,drop=True)

    #sort the class_info by ztf_id
    perm_indx = class_info['ztf_id'].argsort()
    class_info = class_info[perm_indx]
    #set empty values to Unknown instead of 0
    class_info['classification'] = class_info['classification'].filled('Unknown')
    class_info['classification'][0] = 'Unknown' #test instance of classification, also unknown

    #sort the catalogue by ZTF_ID as well
    catalogue.sort_values(by='ZTF_ID',inplace=True,ignore_index=True)
    #filter the ZTFs that aren't in the classification file
    #add the mask as a separate column
    ztfmask = np.isin(np.array(catalogue['ZTF_ID']),np.array(class_info['ztf_id']))
    catalogue['classified'] = ztfmask

    #do the same but for adding to the class file
    ztfmask2 = np.isin(np.array(class_info['ztf_id']),np.array(catalogue['ZTF_ID']))

    #classification as a dataframe 
    class_df = pd.DataFrame(np.array(class_info))
    class_df['fit_exists'] = ztfmask2
    # catalogue = pd.concat([catalogue,class_df.iloc[:,3:]],axis=1) #:,3: because we don't want dupe ztf_id and don't care about RA,DEC
    # catalogue.reset_index(inplace=True,drop=True)

    #filter the bad fits out and add as a column
    qcuts = quality_cuts(catalogue)
    catalogue['fit_quality_good'] = qcuts

    #generate masks for the population selection and strong flare selection
    areamask, strong_flare_mask = pop_cuts(catalogue)
    
    #add the masks as columns for later usage
    catalogue['strong'] = strong_flare_mask
    catalogue['in_selection'] = areamask

    #write catalogue and class info to txt file
    #catalogue will have 4 columns corresponding to masks
    catalogue.to_csv(os.path.join(dpath,'transient_catalogue.txt'),index=False)
    class_df.to_csv(os.path.join(dpath,'transient_classinfo.txt'),index=False)
