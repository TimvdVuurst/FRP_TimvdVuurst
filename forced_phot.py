import numpy as np
import sjoert.stellar
import matplotlib.pyplot as plt
import pandas as pd
import os

'''
List of functions that I can just import throughout my FRP since I'll need them often.
Last update: 26/01/2024
Tim van der Vuurst, Bsc (Leiden University)
'''

def flux_jy(data):
    """Get the flux of a certain data pandas.DataFrame in uJy, with subsequent error as well. Works on clean data structure.

    Args:
        data (DataFrame): Pandas Dataframe containing the ZTF data.

    Returns:
        tuple: Returns tuple of flux and error ordered on time both in uJy.
    """
    zp = sjoert.stellar.mag2flux(data['zeropoint'])
    flux = zp * data['flux'] * 1e6
    err = zp * data['flux_unc'] * 1e6
    return flux,err

def chi2(flux,flux_err,model_fit):
    """Chi^2 function for least square fitting.

    Args:
        flux (array): Flux data.
        flux_err (array): Error corresponding to flux data.
        time (array): Time data corersponding to flux data.
        model_fit (array): Fitted model to compare.

    Returns:
        float: Value of chi^2 given the model.
    """
    return np.sum(np.square(flux - model_fit)/np.square(flux_err))

def gaussian(t,mu,amp=1.,sigma=10.):
    """Normalized Gaussian for peak finding purposes.

    Args:
        t (array): Time data of the TDE
        mu (float): The point at which we say the peak is
        amp (float, optional): Amplitude of the Gaussian. Defaults to 1.
        sigma (float, optional): Standard deviation, characteristic of the width. Defaults to 10, which means a rise and decay of ~20 days.

    Returns:
        array: Gaussian model as per the given parameters and timesteps.
    """
    gauss = np.exp(-(np.square(t-mu)/(2*sigma**2)))
    return gauss/np.sum(gauss)

def chi2_peak_finder(flux,flux_err,time,time_zeropoint):
    """Finds an estimate of TDE peak using naive least chi-square fitting of a gaussian of ~20 days width
       for the purposes of fitting a rise-decay function later. A new time array is initialized that takes
       steps of 5 days for the purpose of reducing straing and not re-using exisiting data points for peak
       candidates. The closest time from the actual data is then found and the corresponding flux is assigned
       as the peak guess.

    Args:
        flux (array): Flux data.
        flux_err (array): Error corresponding to flux data.
        time (array): Time data corersponding to flux data.
        normalized (bool): If False, the data was not normalized and the function does this for you. Otherwise, the data isn't touched.

    Returns:
       tuple: Tuple of the chi2 results per the chi2 function above and the index at which these values reaches a minimum.
    """
    latest_peak =  2459783.499988 #jd of 31-07-2022 23:59:59.000
    latest_peak -= time_zeropoint

    timesteps_for_gauss = np.arange(np.min(time[1:]),latest_peak,step=5) #key array that deduces the times which will become candidates

    flux_norm = np.copy(flux) / np.max(flux)
    flux_err_norm = np.copy(flux_err) / np.max(flux)
    chi2_results = np.full(timesteps_for_gauss.shape,np.nan)

    for i,t in enumerate(timesteps_for_gauss):
        # fit = gaussian(flux,mu)
        if t < latest_peak:
            chi2_results[i] = chi2(flux_norm,flux_err_norm,gaussian(time,amp=1,mu=t))
    peak = np.nanargmin(chi2_results) #ignore the nans, find the least chi2

    #finding the flux of the closest corresponding time in the actual data as the corresponding flux to the peak guess
    best_time_insteps = timesteps_for_gauss[peak]
    best_fit_arg = np.argmin(np.abs(time - best_time_insteps))
    best_flux = flux[best_fit_arg]

    return chi2_results, peak, timesteps_for_gauss[peak], best_flux


def preprocess_clean_data(datapath):
    """Preprocess a cleaned json file given its path. The full clean data will be read in. A mask that can filter out all
       ZTF_i measurements is created. Flux measurements and its errors are converted to uJy and the DataFrame is appended to only retain this
       information. Time is normalized to the minimal Julian Date of a measurement in ZTF_g or ZTF_r for the purposes of lessening the strain
       on the fitting procedure later on (smaller numbers). 

    Args:
        datapath (string): Path to clean data.

    Returns:
        data (DataFrame): pandas DataFrame of full, transformed clean data.
        ztf_name (string): string containing the ZTF identifier of the transient.
        no_i_mask (array_like): mask array that can filter out all ZTF_i filter measurements.
        time_zeropoint (float): julian date of the zeropoint used in this instance, saved since it will differ for each clean data file. 
    """
    data = pd.read_json(datapath) #read in the clean data
    data.sort_values('time',inplace=True) #sort the data by time for plotting reasons later on
    ztf_name = os.path.split(datapath)[-1].split('_')[0] #save the ZTF name using the clean data naming convention
    no_i_mask = data['filter'] != 'ZTF_i' #create the ZTf_i mask
    flux,err = flux_jy(data) #convert the flux and its errors to uJY
    err = np.clip(err,0.01*flux,np.inf) # clip the errors to be at least 1% of the flux
    data['flux'] = flux.values
    data['flux_unc'] = err.values
    time_zeropoint = np.min(data['time'][no_i_mask]) #find the zeropoint for time to be used here
    time_mjd = data['time'] - time_zeropoint
    data['time'] = time_mjd.values #transform time column

    return data, ztf_name, no_i_mask, time_zeropoint





#Works best if it's in a class, otherwise we'll have redefine/re-initialize things like ztf_name and time_zeropoint all the time.

# def plot_clean_unclean_data(clean_data,unclean_data):
#     unclean_data = unclean_data[unclean_data['forcediffimfluxunc'] > 0]
#     clean_flux, clean_err, clean_time = clean_data[['flux','flux_unc','time']].T.to_numpy(dtype=np.float64)
#     raw_flux, raw_err, raw_time = unclean_data[['forcediffimflux','forcediffimfluxunc','jd']].T.to_numpy(dtype=np.float64)
#     raw_flux = sjoert.stellar.mag2flux(unclean_data['zpdiff']) * 1e6 * raw_flux
#     raw_err = sjoert.stellar.mag2flux(unclean_data['zpdiff']) * 1e6 * raw_err
#     raw_flux,raw_err,raw_time = raw_flux,raw_err,raw_time

#     clean_filtermasks = [(clean_data['filter'] == 'ZTF_g'), (clean_data['filter'] == 'ZTF_r'), (clean_data['filter'] == 'ZTF_i')]
#     raw_filtersmasks = [(unclean_data['filter'] == 'ZTF_g'), (unclean_data['filter'] == 'ZTF_r'), (unclean_data['filter'] == 'ZTF_i')]
#     colors = ['green','red','brown']
#     names = ['ZTF_g','ZTF_r','ZTF_i']

#     fig,axes = plt.subplots(nrows=3,sharex=True,figsize=(6,8))
#     plt.suptitle(ztf_name,fontsize=14)
#     axes[0].xaxis.set_tick_params(which='both', labelbottom=True)
#     axes[1].xaxis.set_tick_params(which='both', labelbottom=True)
#     for i,ax in enumerate(axes):
#         ax.set_title(names[i])
#         ax.errorbar(raw_time[raw_filtersmasks[i]],raw_flux[raw_filtersmasks[i]],raw_err[raw_filtersmasks[i]],fmt=',',alpha=0.5,c='gray',label='Raw data')
#         ax.errorbar(clean_time[clean_filtermasks[i]],clean_flux[clean_filtermasks[i]],clean_err[clean_filtermasks[i]],fmt=',',c=colors[i],label='Cleaned data')
#         ax.legend(loc='lower right')
#         ax.set_ylabel(r'Flux [$\mu$Jy]',fontsize=12)
    
#     plt.xlabel(f'Time [mjd] w.r.t. JD {time_zeropoint}',fontsize=12)
#     fig.tight_layout()
#     plt.show()

# def BB(nu,T):
#     #Blackbody spectrum for a certain frequency given in Hz, not an array of values
#     factor = 2*h*np.power(nu,3)/(c**2)
#     exponent = (h*nu)/(k*T)
#     return factor /(np.exp(exponent)-1)

# def BB_ratio(T):
#     return BB(v1,T)/BB(v0,T)

# #With BB temperature correction, used for fitting to ZTF_g and ZTF_r data
# def gauss_exp_fit(t,*p):
#     Fp = 10**p[0]
#     peak_position = p[1]
#     sigma_rise = 10**p[2]
#     tau_dec = 10**p[3]
#     F0 = p[4]
#     T = 10**p[5]

#     trel = t - peak_position
#     gaussian = lambda t: Fp * np.exp(-(np.square(t-peak_position)/(2*sigma_rise**2))) + F0
#     exp_decay = lambda t: Fp * np.exp(-(t-peak_position)/tau_dec) + F0

#     function = np.piecewise(t,[trel <= 0,trel>0],[gaussian, exp_decay])
#     return function * BB_ratio(T)

# #without the BB correction, used for plotting. 
# def gauss_exp(t,*p):
#     Fp = 10**p[0]
#     peak_position = p[1]
#     sigma_rise = 10**p[2]
#     tau_dec = 10**p[3]
#     F0 = p[4]
#     T = 10**p[5]

#     trel = t - peak_position
#     gaussian = lambda t: Fp * np.exp(-(np.square(t-peak_position)/(2*sigma_rise**2))) + F0
#     exp_decay = lambda t: Fp * np.exp(-(t-peak_position)/tau_dec) + F0

#     # trel2 = t - t[np.argmin(gaussian(t)-Fp)]
#     # print(trel2)
#     function = np.piecewise(t,[trel <= 0,trel>0],[gaussian, exp_decay])
#     return function 
