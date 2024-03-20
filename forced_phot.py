import numpy as np
import sjoert.stellar
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit
import json
from scipy.constants import h,c,k
from numpyencoder import NumpyEncoder

'''
List of functions that I can just import throughout my FRP since I'll need them often.
Last update: 20/03/2024
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
        time (array): Time data corersponding to flux data. Assumes this has already been transformed to mjd given the time zeropoint.
        normalized (bool): If False, the data was not normalized and the function does this for you. Otherwise, the data isn't touched.

    Returns:
       tuple: Tuple of the chi2 results per the chi2 function above and the index at which these values reaches a minimum.
    """
    latest_peak =  2459945.49999  #jd of 31-12-2022 23:59:59.000
    # latest_peak =  2459783.499988  #jd of 31-07-2022 23:59:59.000
    # latest_peak =  2460157.49999  #jd of 31-07-2023 23:59:59.000

    #transform to mjd
    latest_peak -= time_zeropoint
    #it may be the case we don't have nearly data beyond the latest peak, in which case we take the latest datapoint as the latest peak

    if latest_peak > np.max(time):
        latest_peak = np.max(time)

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


def cross_correlation(flux,time,time_zeropoint=2458484.5,full_output=True):
    latest_peak =  2459945.49999  #jd of 31-12-2022 23:59:59.000
    latest_peak -= time_zeropoint #in mjd

    #it may be the case we don't have data beyond the latest peak, 
    #in which case we take the latest datapoint as the latest peak
    if latest_peak > np.max(time):
        latest_peak = np.max(time)

    timesteps_for_gauss = np.arange(np.min(time[1:]),latest_peak,step=5) #key array that deduces the times which will become candidates
    cross_corr = np.zeros_like(timesteps_for_gauss)

    #interpolate the data at the timesteps we use so that the arrays have the same shape.
    # interp_data = np.interp(timesteps_for_gauss,time,flux)#/np.max(flux)

    for i,t in enumerate(timesteps_for_gauss):
        #we evaluate the gaussian at the defined timesteps and use the interpolated data to compare to. This usually reduces 
        #runtime greatly compared to evaluating the gaussian at all times in "time" and using the actual flux data
        #and gives comparable results.
        interp_gauss = np.interp(time,timesteps_for_gauss,gaussian(timesteps_for_gauss,amp=1,mu=t))
        cross_corr[i] = np.dot(interp_gauss,flux)
    
    #best fit is the highest cross correlation
    
    peak = np.argmax(cross_corr)

    #finding the flux of the closest corresponding time in the actual data as the corresponding flux to the peak guess
    best_time_insteps = timesteps_for_gauss[peak]
    best_fit_arg = np.argmin(np.abs(time - best_time_insteps))
    best_flux = flux[best_fit_arg]

    if full_output:
        return cross_corr, peak, timesteps_for_gauss, best_flux
    return peak,timesteps_for_gauss[peak],best_flux



##NEEDS UPDATING
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

class ZTF_forced_phot:
    def __init__(self,ztf_dir,ztf_name=None): 
        """Class: ZTF_forced_phot. Given a directory and optionally a name (if the name is not the end of the directory), perform forced photometry
        on the object. Both a plot of the raw and clean data together as well as a plot with fits to the data may be generated. After fitting the 
        subsequent parameters may be saved to a json file as well. Saving happens in the specified directory. The __init__() function loads in the 
        data and does some necessary preprocessing.
        The class runs on pandas to create dataframes, both for oversight in the code and ease in calling columns. This is of course not necessary
        and the code may be relatively easy rewritten to contain only NumPy arrays.

        Args:
            ztf_dir (str): Path to the directory where the clean data, raw data and log file of the ZTF data is stored.
            ztf_name (str, optional): Name of the ZTF we are investigating here. Preferably, the data directory of each object is named after
            the object itself and thus the ZTF identifier may be taken from the ztf_dir argument. If this is not the case, use this argument.
            Defaults to None.
        """
        if ztf_name == None:
            ztf_name = os.path.split(ztf_dir)[-1]


        #gather the log and check if there even is any viable g and r data.
        #if there is not, turn on a flag to prevent errors later on and write the 
        #case to a txt file in the parent directory.
        with open(os.path.join(ztf_dir,f"{ztf_name}_clean_log.json")) as f:
            logfile = json.load(f)

        self.no_gr_flag = False
        if logfile['ZTF_g']['no_viable_data'] or logfile['ZTF_r']['no_viable_data']:
            parent_dir = os.path.split(ztf_dir)[0]
            with open(os.path.join(parent_dir,'no_viable_g_or_r.txt'),'a+') as no_gr_file:
                no_gr_file.seek(0)
                lines = no_gr_file.readlines()
                if ztf_name+'\n' not in lines:
                    no_gr_file.write(ztf_name+'\n')
            
            self.no_gr_flag = True
            print(f'No viable g or r data in {ztf_name}. Skipping this instance.')
            return 


        clean_data = pd.read_csv(os.path.join(ztf_dir,f"{ztf_name}_clean_data.txt"),
                                 sep='\t',comment='#',
                                 names=['time','flux','flux_unc','zeropoint','filter'])
        clean_data.sort_values('time',inplace=True) #sort the data by time for plotting reasons later on
        no_i_mask = clean_data['filter'] != 'ZTF_i' #create the ZTf_i mask to filter out ZTF-i-band data.
        flux,err = flux_jy(clean_data) #convert the flux and its errors to uJY
        err = np.clip(err,0.01*flux,np.inf) # clip the errors to be at least 1% of the flux
        clean_data['flux'] = flux.values #overwrite with new data
        clean_data['flux_unc'] = err.values 

        #create numpy arrays for ease and efficiency
        flux, err, time = clean_data[['flux','flux_unc','time']].T.to_numpy(dtype=np.float64)

        time_zeropoint = 2458484.5 #JD of 01-01-2019 @ 00:00:00.000
        #transform time column
        time_mjd = clean_data['time'] - time_zeropoint
        clean_data['time'] = time_mjd.values 

        columns = ['sindex', 'field', 'ccdid', 'qid', 'filter', 'pid', 'infobitssci', 'sciinpseeing', 'scibckgnd', 'scisigpix', 'zpmaginpsci', 'zpmaginpsciunc', 'zpmaginpscirms', 'clrcoeff', 'clrcoeffunc', 'ncalmatches', 'exptime', 'adpctdif1', 'adpctdif2', 'diffmaglim', 'zpdiff', 'programid', 'jd', 'rfid', 'forcediffimflux', 'forcediffimfluxunc', 'forcediffimsnr', 'forcediffimchisq', 'forcediffimfluxap', 'forcediffimfluxuncap', 'forcediffimsnrap', 'aperturecorr', 'dnearestrefsrc', 'nearestrefmag', 'nearestrefmagunc', 'nearestrefchi', 'nearestrefsharp', 'refjdstart', 'refjdend', 'procstatus']
        dtypes = [(columns[x],float) for x in range(len(columns))]
        dtypes[4] = ('filter',r'U8')
        for file in os.listdir(ztf_dir):
            if 'batchfp' in file:
                batchrq_string = file
        unclean_data = pd.DataFrame(np.genfromtxt(os.path.join(ztf_dir,batchrq_string),skip_header=53,dtype=dtypes))
        unclean_data = unclean_data[unclean_data['forcediffimfluxunc'] > 0] #these need to be removed or plotting doesn't work, usually not a lot of datapoints.


        #Source: http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=Palomar&gname2=ZTF&asttype=
        g_center = c/ (4746.48 * 1e-10)
        r_center = c / (6366.38 * 1e-10)
        i_center = c / (7867.41 * 1e-10)

        self.clean_filtermasks = [(clean_data['filter'] == 'ZTF_g'), (clean_data['filter'] == 'ZTF_r'), (clean_data['filter'] == 'ZTF_i')]

        # chi2_results, peak_ind, t_0_guess, peak_guess = chi2_peak_finder(flux,err,clean_data['time'],time_zeropoint)
        #cross correlation with a simple Gaussian to get a guess for the peak.
        peak_ind,t_0_guess,peak_guess = cross_correlation(flux,clean_data['time'].values,time_zeropoint,full_output=False)

        time_mask_pure = (clean_data['time'] > (t_0_guess - 365)) & (clean_data['time'] < (t_0_guess+365*2)) #these are the times we will be fitting on
        time_mask = time_mask_pure * no_i_mask # also filter out ZTF_i measurements

        #values for fitting
        flux_fit,err_fit,time_fit,filters_fit = flux[time_mask],err[time_mask],clean_data['time'][time_mask].to_numpy(),clean_data['filter'][time_mask].to_numpy()

        #frequency array and "central" frequency for blackbody ratio correction
        nu_1 = [g_center if f == 'ZTF_g' else r_center for f in filters_fit]
        nu_1 = np.array(nu_1).astype(np.float64)
        nu_0 = np.average([g_center,r_center],weights=[np.sum(filters_fit=='ZTF_g'),np.sum(filters_fit=='ZTF_r')])

        #Initial guesses and boundings for fitting in order: Fp, peak_pos, sigma, tau_dec, F0, T
        #There are seperate initial guesses for g and r only fitting, namely in the baseline
        
        guesses = [np.log10(np.max(flux_fit)),t_0_guess,1,2.5,0,4]
        #365 used to be 100 keep that in mind
        boundings = ([1,t_0_guess-365,0,0,np.min(flux[no_i_mask]),3],[np.log10(np.max(flux_fit*2)),t_0_guess+365,4,4,.5*np.max(flux_fit),5]) 
        
        #For g and r seperate we need to be more careful
        #these may be very low
        gfilt_time_mask = np.invert(time_mask)&self.clean_filtermasks[0] 
        rfilt_time_mask = np.invert(time_mask)&self.clean_filtermasks[1]

        #It may happen that the flux outside of the time_mask range for a given filter is only a single point, meaning the lower and 
        #upper bounds for the baselines will be the same value. If this is the case, take the percentile of the entire filter data
        #start with all data, see if we can reduce that to only the data outside the fit range
        lower_baseline_g = np.percentile(flux[self.clean_filtermasks[0]],5)
        upper_baseline_g = np.percentile(flux[self.clean_filtermasks[0]],95)
        median_baseline_g = np.median(flux[self.clean_filtermasks[0]])
        if np.sum(gfilt_time_mask) > 2:
            lower_baseline_g = np.percentile(flux[gfilt_time_mask],5)
            upper_baseline_g = np.percentile(flux[gfilt_time_mask],95)
            median_baseline_g = np.median(flux[gfilt_time_mask])

        if lower_baseline_g == upper_baseline_g:
            lower_baseline_g = np.percentile(flux[self.clean_filtermasks[0]],5)
            upper_baseline_g = np.percentile(flux[self.clean_filtermasks[0]],95)
        
        guesses_g = [np.log10(np.max(flux_fit)),t_0_guess,1,2.5,median_baseline_g,4]
        boundings_g = ([1,t_0_guess-100,0,0,lower_baseline_g,3], #lower bounds
                    [np.log10(np.max(flux_fit*2)),t_0_guess+100,4,4,upper_baseline_g,5]) #upper boundings
        
        #start with all data, see if we can reduce that to only the data outside the fit range
        lower_baseline_r = np.percentile(flux[self.clean_filtermasks[1]],5)
        upper_baseline_r = np.percentile(flux[self.clean_filtermasks[1]],95)
        median_baseline_r = np.median(flux[self.clean_filtermasks[1]])
        if np.sum(rfilt_time_mask) > 2:
            lower_baseline_r = np.percentile(flux[rfilt_time_mask],5)
            upper_baseline_r = np.percentile(flux[rfilt_time_mask],95)
            median_baseline_r = np.median(flux[rfilt_time_mask])            

        if lower_baseline_r == upper_baseline_g:
            lower_baseline_r = np.percentile(flux[self.clean_filtermasks[1]],5)
            upper_baseline_r = np.percentile(flux[self.clean_filtermasks[1]],95)
        
        guesses_r = [np.log10(np.max(flux_fit)),t_0_guess,1,2.5,median_baseline_r,4]
        boundings_r = ([1,t_0_guess-100,0,0,lower_baseline_r,3], 
                       [np.log10(np.max(flux_fit*2)),t_0_guess+100,4,4,upper_baseline_r,5]) #upper boundings

        #initializing all the variables that need to be used later on with self.
        self.ztf_name = ztf_name
        self.ztf_dir = ztf_dir

        self.no_i_mask = no_i_mask
        self.time_zeropoint = time_zeropoint
        self.time_mask = time_mask

        self.clean_data = clean_data
        self.flux, self.err, self.time = self.clean_data[['flux','flux_unc','time']].T.to_numpy(dtype=np.float64)
        self.flux_fit,self.err_fit,self.time_fit,self.filters_fit = flux_fit,err_fit,time_fit,filters_fit

        if sum(self.clean_filtermasks[-1]) == 0:
            self.no_i_data = True
        else:
            self.no_i_data = False

        self.logfile = logfile
        self.unclean_data = unclean_data
        self.g_center = g_center
        self.r_center = r_center
        self.i_center = i_center
        self.nu_0, self.nu_1 = nu_0, nu_1
        self.guesses, self.boundings= guesses, boundings
        self.guesses_g, self.guesses_r = guesses_g, guesses_r
        self.boundings_g, self.boundings_r = boundings_g, boundings_r
        self.peak_ind, self.t_0_guess, self.peak_guess = peak_ind, t_0_guess, peak_guess
        self.time_mask_pure = time_mask_pure


    def plot_clean_unclean_data(self,clean_ylim=True):
        if self.no_gr_flag:
            return

        clean_flux, clean_err, clean_time = self.flux,self.err,self.time
        raw_flux, raw_err, raw_time = self.unclean_data[['forcediffimflux','forcediffimfluxunc','jd']].T.to_numpy(dtype=np.float64)
        raw_flux = sjoert.stellar.mag2flux(self.unclean_data['zpdiff']) * 1e6 * raw_flux
        raw_err = sjoert.stellar.mag2flux(self.unclean_data['zpdiff']) * 1e6 * raw_err
        raw_time -= self.time_zeropoint

        raw_filtersmasks = [(self.unclean_data['filter'] == 'ZTF_g'), (self.unclean_data['filter'] == 'ZTF_r'), (self.unclean_data['filter'] == 'ZTF_i')]
        colors = ['green','red','brown']
        names = ['ZTF g-band','ZTF r-band','ZTF i-band']
        rawlabels = ['Raw data',None,None]

        num_rows = np.sum([ 1 if np.sum(fmask) > 0 else 0 for fmask in raw_filtersmasks])
        fig,axes = plt.subplots(nrows=num_rows,sharex=True,figsize=(8,8))
        plt.suptitle(self.ztf_name,fontsize=14)
        axes[0].xaxis.set_tick_params(which='both', labelbottom=True)
        axes[1].xaxis.set_tick_params(which='both', labelbottom=True)

        lines = []
        labels = []

        for i,ax in enumerate(axes):
            ax.set_title(names[i])
            ax.errorbar(raw_time[raw_filtersmasks[i]],raw_flux[raw_filtersmasks[i]],raw_err[raw_filtersmasks[i]],fmt=',',alpha=0.5,c='gray',label=rawlabels[i],capsize=2)
            if i == 2:
                if self.no_i_data:
                    print("There is no clean ZTF i-band data.")
                else:
                    ax.errorbar(clean_time[self.clean_filtermasks[i]],clean_flux[self.clean_filtermasks[i]],clean_err[self.clean_filtermasks[i]],fmt=',',c=colors[i],label=names[i],capsize=2)
            else:
                ax.errorbar(clean_time[self.clean_filtermasks[i]],clean_flux[self.clean_filtermasks[i]],clean_err[self.clean_filtermasks[i]],fmt=',',c=colors[i],label=names[i],capsize=2)
            
            ax.set_ylabel(r'Flux [$\mu$Jy]',fontsize=12)
            Line, Label = ax.get_legend_handles_labels() 
            lines.extend(Line) 
            labels.extend(Label)
            if clean_ylim:
                if i == 2 and self.no_i_data:
                    pass
                else:
                    ax.set_ylim(1.25*np.min(clean_flux[self.clean_filtermasks[i]]),1.25*np.max(clean_flux[self.clean_filtermasks[i]]))
        
        
        plt.xlabel(f'Time [mjd] w.r.t. JD {self.time_zeropoint}',fontsize=12)
        fig.tight_layout()
        fig.legend(lines,labels,bbox_to_anchor=[1.2,0.6],fontsize=12)
        plt.show()
    
    def BB(self,nu,T):
        #Blackbody spectrum for a certain frequency given in Hz, not an array of values
        factor = 2*h*np.power(nu,3)/(c**2)
        exponent = (h*nu)/(k*T)
        return factor /(np.exp(exponent)-1)
    
    def BB_ratio(self,T,v,v_0):
        return self.BB(v,T)/self.BB(v_0,T)

    def gauss_exp(self,t,nu,nu_0,no_baseline,*p):

        Fp = 10**p[0]
        t_0 = p[1]
        sigma_rise = 10**p[2]
        tau_dec = 10**p[3]
        no_temp = False
        try:
            T = 10**p[5]
        except:
            no_temp = True
        if no_baseline:
            T = 10**p[4]
            no_temp = False
            F_0 = 0
        else:
            F_0 = p[4]

        trel = t - t_0
        gaussian = lambda t1: Fp * np.exp(-(np.square(t1-t_0)/(2*sigma_rise**2))) + F_0
        exp_decay = lambda t1: Fp * np.exp(-(t1-t_0)/tau_dec) + F_0

        function = np.piecewise(t,[trel <= 0,trel>0],[gaussian, exp_decay])
        # print((exp_decay(t[np.abs(t - t_0).argmin()]) - F_0 )/ Fp)
        if no_temp:
            return function
        else:
            return function* self.BB_ratio(T,nu,nu_0) #for plotting it should be specified at what frequency we are looking.
    
        
    #With BB temperature correction, used for comparing to ZTF_g and ZTF_r data
    def gauss_exp_fit(self,t,*p):
        return self.gauss_exp(t,self.nu_1,self.nu_0,False,*p)
    
    def gauss_exp_fit_no_baseline(self,t,*p):
        return self.gauss_exp(t,self.nu_1,self.nu_0,True,*p)

    def gauss_exp_fit_g(self,t,*p):
        return self.gauss_exp(t,self.g_center,self.nu_0,False,*p)
    
    def gauss_exp_fit_r(self,t,*p):
        return self.gauss_exp(t,self.r_center,self.nu_0,False,*p)

    def gauss_exp_fit_for_plot(self,t,v1,v0,*p):
        return self.gauss_exp(t,v1,v0,True,*p)
    
    def gauss_exp_baseline_peak(self,t,t_0,sigma_rise,tau_dec,T,v0,v1,*p):
        #only baseline and peak are fittable, used for i band
        #v0 is the same as before, v1 is the i band central frequency
        Fp = 10**p[0]
        F_0 = p[1]

        trel = t - t_0
        gaussian = lambda t: Fp * np.exp(-(np.square(t-t_0)/(2*sigma_rise**2))) + F_0
        exp_decay = lambda t: Fp * np.exp(-(t-t_0)/tau_dec) + F_0

        function = np.piecewise(t,[trel <= 0,trel>0],[gaussian, exp_decay])
        return function #* self.BB_ratio(T,v1,v0)


    def fit(self,plot=True,fit_i=True):
        #If this is an instance with no viable g or r data, return immediately to prevent errors.
        if self.no_gr_flag:
            return

        # First fit g and r seperately to find the baseline correction.
        popt_g, pcov_g = curve_fit(self.gauss_exp_fit_g,self.time_fit[self.clean_filtermasks[0][self.time_mask]],
                                   self.flux_fit[self.clean_filtermasks[0][self.time_mask]],
                                    p0=self.guesses_g[:-1],bounds=[b[:-1] for b in self.boundings_g],
                                    sigma=self.err_fit[self.clean_filtermasks[0][self.time_mask]],
                                    full_output=False,
                                    absolute_sigma=True) 

        popt_r, pcov_r = curve_fit(self.gauss_exp_fit_r,self.time_fit[self.clean_filtermasks[1][self.time_mask]],
                                self.flux_fit[self.clean_filtermasks[1][self.time_mask]],
                                p0=self.guesses_r[:-1],bounds=[b[:-1] for b in self.boundings_r],
                                sigma=self.err_fit[self.clean_filtermasks[1][self.time_mask]],
                                full_output=False,
                                absolute_sigma=True)
        
        #Get the filter dependent baseline and its error from the fit.
        F0_g, F0_g_err = popt_g[-1], np.sqrt(np.diag(pcov_g)[-1])
        F0_r, F0_r_err = popt_r[-1], np.sqrt(np.diag(pcov_r)[-1])

        #Create an array with varying values, either the g-band baseline or the r-band baseline, used for fitting both filters together
        #much in the same way as we did for nu_1
        baseline = [F0_g if self.filters_fit[i] == 'ZTF_g' else F0_r for i in range(len(self.filters_fit))]
        baseline_err = [F0_g_err if self.filters_fit[i] == 'ZTF_g' else F0_r_err for i in range(len(self.filters_fit))]


        #fitting g and r together to get the other parameter values. The data is fitted on baseline transposed data 
        #(with subsequent error propagation) but this is not the shown data in the plot.
        popt,pcov = curve_fit(self.gauss_exp_fit_no_baseline,self.time_fit,self.flux_fit - baseline,
                                            p0=self.guesses[:-2]+[self.guesses[-1]],bounds=[b[:-2]+[b[-1]] for b in self.boundings],
                                            sigma=np.sqrt(np.square(self.err_fit) + np.square(baseline_err)),
                                            full_output=False,
                                            absolute_sigma=True)

        #Get the error estimates from the covariance matrix
        perr = np.sqrt(np.diag(pcov))

        #No i-data? Then don't try to fit a baseline and peak to i 
        if np.sum(np.invert(self.no_i_mask)) == 0:
            fit_i = False
        if fit_i:
            #Fitting on i-band data. We fit here only a baseline and a peak keeping the other found parameters equal.
            i_func_to_fit = lambda t,*p: self.gauss_exp_baseline_peak(t,popt[1],popt[2],popt[3],popt[4],self.nu_0,self.i_center,*p)
            only_i_mask = np.invert(self.no_i_mask)
            i_time = self.clean_data['time'][only_i_mask].values
            i_flux,i_flux_err = self.clean_data['flux'][only_i_mask].values, self.clean_data['flux_unc'][only_i_mask].values

        #Initial guesses and boundings for fitting in order: Fp, peak_pos, sigma, tau_dec, F0, T
        #check if there is even any data within the time-frame in which we fit the data
        #if there isn't we can just use the i-data as is since it always exists outside our time-frame
            if np.sum(i_flux > 0) > 2: #if there are more than 2 positively valued points, meaning we can do percentiles and medians:
                if len(i_flux[np.invert(self.time_mask_pure)[only_i_mask]]) < 2:
                    i_guesses = [np.log10(np.max(i_flux)),np.median(i_flux)]
                    i_bounds = ([np.log10(0.5*np.max(i_flux)),np.percentile(i_flux,5)],
                            [np.log10(np.max(i_flux*2)),np.percentile(i_flux,95)])
                #otherwise, we must do some calculations to make sure we do our estimations on the data outside the timeframe
                else:
                    f_guess_bounds = i_flux[np.invert(self.time_mask_pure)[only_i_mask]] #Get only the i data that exists outside our time-frame
                    if len(f_guess_bounds) <= 2:
                        f_guess_bounds = i_flux

                    i_guesses = [np.log10(np.max(i_flux)),np.median(f_guess_bounds)]
                    i_bounds = ([np.log10(0.5*np.max(i_flux)),np.percentile(f_guess_bounds,5)],
                                [np.log10(np.max(i_flux*2)),np.percentile(f_guess_bounds,95)])

            else:
                fit_i = False
 
            # print(f_guess_bounds)
            # print(i_guesses)
            # print(i_bounds)
            if fit_i:
                popt_i,pcov_i = curve_fit(i_func_to_fit,i_time,i_flux,
                                                p0=i_guesses,
                                                bounds=i_bounds,
                                                sigma=i_flux_err,
                                                full_output=False,
                                                absolute_sigma=True)
                

        #calculate the chi2/dof for the g and r data combined
        dof = len(self.filters_fit) - (len(popt) + 2) #number of points in g and r - the amount of parameters (popt and the 2 baselines)
        chi2_val = chi2(self.flux_fit - baseline,np.sqrt(np.square(self.err_fit) + np.square(baseline_err)),
                       self.gauss_exp_fit_no_baseline(self.time_fit,*popt))
        
        # print(f'Found chi2/dof = {chi2_val/dof:.6f}')


        if plot:
            #array to smoothen the line in the figure
            moretimes = np.linspace(min(self.time_fit),max(self.time_fit),1000)

            centers = [self.g_center,self.r_center,self.i_center]
            
            baseline_corr = [F0_g,F0_r,0]

            #Get the fits for the three filters, undoing the transposition of the baseline correction from before.
            fits_plot = [self.gauss_exp_fit_for_plot(moretimes,c,self.nu_0,*popt) + baseline_corr[i] for i,c in enumerate(centers)]
            
            #If we fitted i, the third entry in fits_plot is now inaccurate since we have a better baseline
            #and peak, so re-do this.
            if fit_i:
                #order: Fp, peak_pos, sigma, tau_dec, F0, T
                new_popt = [popt_i[0],*popt[1:4],popt_i[1]]#,popt[-1]]
                fits_plot[2] = self.gauss_exp(moretimes,self.i_center,self.nu_0,False,*new_popt)

            #Create a string for presenting the parameters in the figure
            paramstr = ''
            newnames = [r'log$_{10}$(F$_\mathrm{p}$)',r't$_0$',r'log$_{10}(\sigma_\mathrm{rise}$)',r'log$_{10}(\tau_\mathrm{dec}$)',r'log$_{10}$(T)']
            for i,n in enumerate(newnames):
                paramstr += f'{n} = {popt[i]:.2f} ± {perr[i]:.3f}'
                paramstr += '\n'

            colors = ['green','red','brown']
            labels_ebar = ['ZTF: g-band','ZTF: r-band','ZTF: i-band']
            lines = []
            labels = []

            fig,axes = plt.subplots(nrows=3,sharex=True,figsize=(8,8))
            #so that ticks are visible on each x-axis but (because of sharex = True) no label except for the bottom axis.
            axes[0].xaxis.set_tick_params(which='both', labelbottom=True)
            axes[1].xaxis.set_tick_params(which='both', labelbottom=True)
            #plot the data and fits
            for i,ax in enumerate(axes):
                ax.errorbar(self.clean_data['time'][self.clean_filtermasks[i]],self.clean_data['flux'][self.clean_filtermasks[i]],
                            self.clean_data['flux_unc'][self.clean_filtermasks[i]],
                            fmt='.',c=colors[i],label=labels_ebar[i],capsize=2)
                if i == 0:
                    ax.plot(moretimes,fits_plot[i],c='black',zorder=10,label='Fit')
                else:
                    ax.plot(moretimes,fits_plot[i],c='black',zorder=10)

                if i < 2:#used to be <3. <2 excludes i but < 3 is always true here
                    if i == 0:
                        ax.scatter(self.t_0_guess,self.peak_guess,s=50,marker='x',c='blue',zorder=10,label='Starting point') 
                    else:
                        ax.scatter(self.t_0_guess,self.peak_guess,s=50,marker='x',c='blue',zorder=10) 

                ax.set_ylabel(r"Flux ($\mu$Jy)",fontsize=12)

                #For the legend
                Line, Label = ax.get_legend_handles_labels() 
                lines.extend(Line) 
                labels.extend(Label)
                
            #Add in the first baseline and add it to the legend lists, since every baseline looks the same
            axes[0].hlines(F0_g,min(self.time),max(self.time),linestyles='dashed',colors='black',label='Baseline')#,label=r'F$_{0,g} = $' + f'{F0_g:.3f} ± {F0_g_err:.3f}')
            lin,lab = axes[0].get_legend_handles_labels()
            idx = np.where(np.array(lab) == 'Baseline')[0][0]
            lines.append(lin[idx])
            labels.append(lab[idx])

            axes[1].hlines(F0_r,min(self.time),max(self.time),linestyles='dashed',colors='black')#,label=r'F$_{0,r} = $' + f'{F0_r:.3f} ± {F0_r_err:.3f}')
            if fit_i:
                axes[2].hlines(popt_i[1],min(self.time),max(self.time),linestyles='dashed',color='black')
        
            #Final parameters in the text
            paramstr += r'F$_{0,g} = $' + f'{F0_g:.2f} ± {F0_g_err:.3f}\n'
            paramstr += r'F$_{0,r} = $' + f'{F0_r:.2f} ± {F0_r_err:.3f}\n'
            paramstr += r'$\chi ^2_{\nu}$ = ' + f'{chi2_val/dof:.3f}'


            axes[-1].set_xlabel(f"Time (mjd - 2458484.5)",fontsize=12)

            plt.text(0.835,0.125,paramstr,fontsize=10,backgroundcolor='lightgray',zorder=-1,transform=plt.gcf().transFigure)

            plt.suptitle(self.ztf_name,fontsize=14)
            fig.legend(lines,labels,bbox_to_anchor=[1.055,0.6],fontsize=12)
            fig.tight_layout()
            plt.show()
        


        #So that parameters may be saved in the save_params function.
        #Add the baselines to the parameters as well.
        self.popt = np.concatenate([popt,[F0_g,F0_r]])
        self.perr = np.concatenate([np.sqrt(np.diag(pcov)),[F0_g_err,F0_r_err]])
        self.popt_r,self.perr_r = popt_r,np.sqrt(np.diag(pcov_r))
        self.popt_g,self.perr_g = popt_g, np.sqrt(np.diag(pcov_g))
        self.chi_nu = chi2_val/dof
        if plot:
            self.fits_plot = fits_plot


    def save_params(self,savepath=None,save=True):
        if self.no_gr_flag:
            return
        
        names = ['log10(Fp)', 't_0', 'log10(sigma_rise)', 'log10(tau_dec)', 'log10(T)','F_0g','F_0r']
        params_names = ['F_p', 't_0', 'sigma_rise','tau_dec', 'T','F_0g','F_0r','nu_0']
        log_names = ['log10_'+params_names[i] for i,name in enumerate(names) if 'log10' in name]
        units = ['uJy','mjd','days','days','K','uJy','uJy','Hz']
        # filter_order = ['ZTF_g','ZTF_r']


        params = [10**p if 'log10' in names[i] else p for i,p in enumerate(self.popt)]
        param_errs = [np.log(10)*err*params[i] if 'log10' in names[i] else err for i,err in enumerate(self.perr)]

        log_params = [p for i,p in enumerate(self.popt) if 'log10' in names[i]]
        log_param_errs = [e for i,e in enumerate(self.perr) if 'log10' in names[i]]

        params.append(np.float32(self.nu_0)) #since nu_0 doesn't have an error, add it only now. 
        param_errs.append(np.float32(0))

        params.extend(log_params)
        param_errs.extend(log_param_errs)
        # print(np.shape(log_params))

        dict_keys = params_names
        dict_keys.extend(log_names)
        dict_keys.extend(['chi2_dof','units'])
        ordered_params = [np.array([np.float64(params[i]),np.float64(param_errs[i])]) for i in range(len(params))]
        ordered_params.extend([self.chi_nu,units])

        param_dict = dict(zip(dict_keys,ordered_params))
        # print(param_dict)

        #Now doing the pretty much the same again but for the filter dependent parameters
        names_g = ['log10(Fp)', 't_0', 'log10(sigma_rise)', 'log10(tau_dec)','F_0g']
        params_names_g = ['F_p', 't_0', 'sigma_rise','tau_dec','F_0g']
        log_names_g = ['log10_'+params_names_g[i] for i,name in enumerate(names_g) if 'log10' in name]

        names_r = ['log10(Fp)', 't_0', 'log10(sigma_rise)', 'log10(tau_dec)','F_0r']
        params_names_r = ['F_p', 't_0', 'sigma_rise','tau_dec','F_0r']
        log_names_r = ['log10_'+params_names_r[i] for i,name in enumerate(names_r) if 'log10' in name]

        units_filt = ['uJy','mjd','days','days','uJy']

        params_g = [10**p if 'log10' in names_g[i] else p for i,p in enumerate(self.popt_g)]
        param_errs_g = [np.log(10)*err*params_g[i] if 'log10' in names_g[i] else err for i,err in enumerate(self.perr_g)]
        log_params_g = [p for i,p in enumerate(self.popt_g) if 'log10' in names_g[i]]
        log_param_errs_g = [e for i,e in enumerate(self.perr_g) if 'log10' in names_g[i]]
        params_g.extend(log_params_g), param_errs_g.extend(log_param_errs_g)

        params_r = [10**p if 'log10' in names_r[i] else p for i,p in enumerate(self.popt_r)]
        param_errs_r = [np.log(10)*err*params_r[i] if 'log10' in names_r[i] else err for i,err in enumerate(self.perr_r)]
        log_params_r = [p for i,p in enumerate(self.popt_r) if 'log10' in names_r[i]]
        log_param_errs_r = [e for i,e in enumerate(self.perr_r) if 'log10' in names_r[i]]
        params_r.extend(log_params_r), param_errs_r.extend(log_param_errs_r)

        dict_keys_g = params_names_g
        dict_keys_g.extend(log_names_g)
        dict_keys_g.extend(['units'])
        ordered_params_g = [np.array([np.float64(params_g[i]),np.float64(param_errs_g[i])]) for i in range(len(params_g))]
        ordered_params_g.append(units_filt)
        param_dict_g = dict(zip(dict_keys_g,ordered_params_g))

        dict_keys_r = params_names_r
        dict_keys_r.extend(log_names_r)
        dict_keys_r.extend(['units'])
        ordered_params_r = [np.array([np.float64(params_r[i]),np.float64(param_errs_r[i])]) for i in range(len(params_r))]
        ordered_params_r.append(units_filt)
        param_dict_r = dict(zip(dict_keys_r,ordered_params_r))


        param_dicts = [param_dict,param_dict_g,param_dict_r]
        if save:
            if savepath == None:
                savepath = self.ztf_dir
            
            filename = self.ztf_name + '_parameters'
            all_filenames = [filename,filename+'_g',filename+'_r']
            for i,fn in enumerate(all_filenames):
                with open(os.path.join(savepath,fn),'w') as file:
                    json.dump(param_dicts[i],file,indent = 4,cls=NumpyEncoder)

        return param_dicts


        


