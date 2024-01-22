import numpy as np
import sjoert.stellar

'''
List of functions that I can just import throughout my FRP since I'll need them often.
Last update: 18/12/2023
Tim van der Vuurst, Bsc (Leiden University)
'''

def flux_jy(data):
    """Get the flux of a certain data pandas.DataFrame in uJy, with subsequent error as well.

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

def chi2_peak_finder(flux,flux_err,time,time_zeropoint,normalized=False):
    """Finds an estimate of TDE peak using naive least chi-square fitting of a gaussian of ~20 days width
       for the purposes of fitting a rise-decay function later.

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

    flux = np.copy(flux)
    flux_err = np.copy(flux_err)
    chi2_results = np.full(flux.shape,np.nan)
    if not normalized:
        flux /= np.max(flux)
        flux_err /= np.max(flux)
    for i,t in enumerate(time):
        # fit = gaussian(flux,mu)
        if t<= latest_peak:
            chi2_results[i] = chi2(flux,flux_err,gaussian(time,amp=1,mu=t))
    peak = np.nanargmin(chi2_results)
    return chi2_results, peak