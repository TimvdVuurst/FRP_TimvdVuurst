import numpy as np
import matplotlib.pyplot as plt
import sjoert.stellar
import pandas as pd
import os
from tqdm import tqdm
from astropy import coordinates as coord
import json
from numpyencoder import NumpyEncoder


#From Sjoert, slightly modified
"""Define the quality cuts."""
def q_cuts(ZTF, softcuts=True,output=False):
    '''
    update for Jul 2023
    main difference:  forcediffimfluxap/forcediffimflux less strict
    '''
    
    DC2Jy = sjoert.stellar.mag2flux(ZTF['zpdiff'])

    app_flux_ratio = ZTF['forcediffimfluxap'] / ZTF['forcediffimflux']
    app_psf_diff = (ZTF['forcediffimfluxap'] - ZTF['forcediffimflux']) #/ np.abs(ZTF['forcediffimfluxap'])

    #(np.abs(np.log10(app_flux_ratio)) <1.5) *\

    #final three masks added by Tim from section 6.1 of "A New Forced Photometry Service for the Zwicky Transient Facility" 
    iok =   (ZTF['procstatus'] != '56') * \
            (ZTF['scisigpix'] < 20) * \
            (ZTF['sciinpseeing'] < 4.0) * \
            (ZTF['zpmaginpscirms'] < 0.05)  * \
            (ZTF['infobitssci'] < 33554432) * \
            (ZTF['scisigpix'] <= 25) * \
            (ZTF['sciinpseeing'] <= 4) * \
            (ZTF['procstatus'] == 0) #this one added by Tim because error code should be 0 for an epoch, else there will probably still be a faulty epoch in there.
                
          
    if softcuts==False: 
        iok *=  (ZTF['adpctdif1'] < 0.2)  *\
                (np.abs(app_psf_diff)<200)*\
                (DC2Jy*ZTF['forcediffimflux'] > -50e-6) *\
                (DC2Jy*np.abs(ZTF['forcediffimfluxunc']) < 30e-6)    

    # new in 2022 check for zero-point outliers, per filter
    for flt in np.unique(ZTF['filter']):
        iflt = ZTF['filter']==flt

        if softcuts:
            iok.loc[iflt] *= np.abs(np.log10(DC2Jy.loc[iflt]/np.median(DC2Jy.loc[iflt])))<0.4 
        else:
            iok.loc[iflt] *= np.abs(np.log10(DC2Jy.loc[iflt]/np.median(DC2Jy.loc[iflt])))<0.1 # new 2023 (was 0.4)
    if output:
        # hack for checking cuts
        plt.hist(app_psf_diff,range=[-300,300],bins=30)
        plt.pause(0.1)
        # key = input()

        print ('# of raw points     :', len(ZTF))    
        print ('# of points rejected:', len(ZTF)-sum(iok))

    return iok

def field_check(ZTF,output=False):
    fields,counts = np.unique(ZTF['field'],return_counts=True)
    imax = np.argmax(counts)
    if output:
        print(f"Field {fields[imax]} occurs most with {counts[imax]} / {len(ZTF)} instances.")

    to_keep = ZTF['field'] == fields[imax]

    return to_keep

def filter_split(ZTF,asmasks=False,aslist=False):
    filters = np.unique(ZTF['filter'])
    if asmasks:
        return {x:(ZTF['filter'] == x) for x in filters}
    
    elif aslist:
        return [ZTF[(ZTF['filter'] == x)] for x in filters]
    
    return {x:ZTF[(ZTF['filter'] == x)] for x in filters}

def flux_unc_val(ZTF,output=False):
    chisq = ZTF['forcediffimchisq']
    median_chi = np.median(chisq) #instead of mean, more robust

    unc = np.array(ZTF['forcediffimfluxunc']) 
    unc *= np.sqrt(median_chi)

    if output:
        print(median_chi)

        fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(8,5))
        plt.suptitle(f"In {ZTF['filter'].iloc[0]} filter ({len(ZTF)} points)",fontsize=16)
        ax1.set_ylabel(r"$\chi^2$")
        ax1.set_xlabel("Flux")
        ax2.set_xlabel("Counts")
        ax1.scatter(ZTF['forcediffimflux'],chisq)
        ax2.hist(chisq,bins=30,orientation='horizontal')
        plt.show()

    return unc

##OLD
# def clean_data(data,filename=None,savepath=None,split_filters=True):
#     iok = q_cuts(data) * field_check(data)
#     idx_bad = np.where(np.invert(iok))[0]
#     data_ok = data[iok]
#     if savepath != None:
#         pd.DataFrame({"bad_idx":[idx_bad],"median_chi_before":np.median(data['forcediffimchisq']),"median_chi_after":np.median(data_ok['forcediffimchisq'])}).to_csv(\
#             os.path.join(savepath,'clean_log.txt'),sep=' ',index=False)
#     if split_filters:
#         for filter,filter_data in filter_split(data_ok).items():
#             new_unc = np.array(flux_unc_val(filter_data))
#             norm_jd = np.array(filter_data['jd'] - filter_data['jd'].iloc[0])
#             clean_data = pd.DataFrame({"time":norm_jd,"flux":filter_data['forcediffimflux'].values,"flux_unc":new_unc,
#                                     "zeropoint":filter_data['zpdiff'].values})
#             if savepath != None:
#                 assert filename != None
#                 savestring = os.path.join(savepath,"clean_"+filter+"_"+filename)
#                 clean_data.to_csv(savestring,sep=' ',index=False)
#             else:
#                 print(clean_data)
#                 print(20*'--')

def clean_data(datapath,ZTF_ID,savepath=None,verbose=False):
    """Cleans ZTF batch request data using the qcuts function. The removed data is neatly logged on a per filter - per field basis. The cleaning
    now works in such a way that only the data from the primary field (defined as the field in which most measurements were made) is used.
    The uncertainty on the flux measurements is validated and subsequently updated following the method of the ZFPS user guide: 
    https://web.ipac.caltech.edu/staff/fmasci/ztf/zfps_userguide.pdf. The cleaned data contains: time in jd, forced difference image PSF-fit flux [DN],
      the 1-sigma uncertainty in the forced difference image PSF-fit flux [DN], the photometric zeropoint for difference image [mag] and the filter 
      (one of ZTF_g, ZTF_r and ZTF_i). The cleaning log contains, for every field in every filter, whether there are even viable measurements and if
      there are if the field in question is the primary field, what the median zeropoint is, what the standard deviation of the zeropoints is, how
      many of the data points were removed in cleaning and the median chi-square of the datapoints before cleaning. For every filter the median 
      chi-square after cleaning is saved only for the primary field - this will differ only slightly from the median chi-square before cleaning. 
      Both cleaned data and cleaning log are saved as json files in the form "(ZTF_ID)_clean_data.json" and "(ZTF_ID)_clean_log.json".
      
      IMPORTANT: dependencies are the numpy, pandas, json and os packages as well as a special json NumpyEncoder by Hunter M. Allen (https://pypi.org/project/numpyencoder/).

    Args:
        datapath (str): Path to the raw data in the form path\to\data\batchf_reqxxxxxxxxxxxxxxxxx_lc.txt.
        ZTF_ID (str): ZTF identifier of the transient.
        savepath (str, optional): Path to folder in which clean data and cleaning log will be saved. Defaults to None, in which case data is printed if verbose is True, otherwise it is lost.
        verbose (bool, optional): Controls the (amount of) print statements in the function. Defaults to False.
    """
    #Read in the raw data from the data path as a Pandas DataFrame. 
    columns = ['sindex', 'field', 'ccdid', 'qid', 'filter', 'pid', 'infobitssci', 'sciinpseeing', 'scibckgnd', 'scisigpix',
                'zpmaginpsci', 'zpmaginpsciunc', 'zpmaginpscirms', 'clrcoeff', 'clrcoeffunc', 'ncalmatches', 'exptime',
                'adpctdif1', 'adpctdif2', 'diffmaglim', 'zpdiff', 'programid', 'jd', 'rfid', 'forcediffimflux', 'forcediffimfluxunc',
                'forcediffimsnr', 'forcediffimchisq', 'forcediffimfluxap', 'forcediffimfluxuncap', 'forcediffimsnrap', 'aperturecorr',
                'dnearestrefsrc', 'nearestrefmag', 'nearestrefmagunc', 'nearestrefchi', 'nearestrefsharp', 'refjdstart', 'refjdend',
                'procstatus']
    dtypes = [(columns[x],float) for x in range(len(columns))]
    dtypes[4] = ('filter',r'U8')
    data = pd.DataFrame(np.genfromtxt(datapath,skip_header=53,dtype=dtypes))

    clean_data_full = pd.DataFrame() #an empty frame on which the data from every filter will be vertically stacked.
    iok = q_cuts(data) #very first quality check, mask array of good data points.
    data_ok = data[iok]
    logdict = {} #dictionary that will form the log.json file

    filters = np.unique(data['filter'])
    filtermasks = [data['filter'] == f for f in filters]
    # fields,field_counts = np.unique(data['field'],return_counts=True) #return_counts for picking the primary field
    fields,field_counts = np.unique(data_ok['field'],return_counts=True) #Do we want the primary field on data or on data_ok?
    fieldmasks = [data['field'] == fid for fid in fields]
    primary_field = [i == np.max(field_counts) for i in field_counts] #should be this one if primary field is for the whole data

    if iok.sum() == 0: #this might occur, this prevents an error
        print(f"{ZTF_ID}: no viable data found in the batch request. Proceeding to next file.")
        return 
    
    for i,filter in enumerate(filters):
            logdict[filter] = {}
            logdict[filter]["no_viable_data"] = 0 #can be used for a check when loading in the data; if this is True then the data is useless in this particular filter 
            filtermask = filtermasks[i]
            iok_filter = (iok * filtermask) #this checks if something is ok according to qcuts and is in a certain filter. Has the same len as data.
            
            if iok_filter.sum() == 0: #this might occur, this prevents an error
                print(f"{ZTF_ID}: no viable data found in {filter}. Proceeding to next filter.")
                logdict[filter]["no_viable_data"] = 1
                continue

            ##IF we take the primary field on a per filter basis use three lines below.
            # data_ok_filter = data[iok_filter]
            # filter_field_counts = [np.sum(data_ok_filter['field'] == fid) for fid in fields] #count for each field we know to have in the uncleaned data how often it appears in this filter. Might yield 0's! 
            # primary_field = [c == np.max(filter_field_counts) for c in filter_field_counts] #should be this one if we want to pick the primary on a per filter basis

            for j,fid in enumerate(fields):
                field_mask = fieldmasks[j]
                iok_filter_field = iok_filter * field_mask #this checks if something is ok according to qcuts, is in a certain filter and is in a certain field. Has the same len as data.

                logdict[filter][fid] = {}
                if iok_filter_field.sum() == 0: #this might occur, this prevents an error
                    print(f"{ZTF_ID}: no viable data found in field {fid} of filter {filter}. Proceeding to next field for this filter.")
                    logdict[filter][fid]["no_viable_data"] = 1 #can be used for a check when loading in the data; if this is True then the data is useless in this particular field / filter combo
                    continue

                data_ok_filter_field = data[iok_filter_field]
                data_filter_field = data[filtermask*field_mask] #this is the uncleaned data of this field in this filter
                zeropoint = data_ok_filter_field['zpdiff'].values

                logdict[filter][fid] = {"primary_field":int(primary_field[j]),
                                        "median_zeropoint":np.median(zeropoint),'std_zeropoint':np.std(zeropoint),
                                        "removed_in_cleaning":np.sum(np.invert(iok_filter_field)),
                                        "median_chi2":np.median(data_filter_field['forcediffimchisq']),
                                            "no_viable_data":0}

                if primary_field[j]:
                    #correct the errors of the clean data in this filter (only on the primary field)
                    new_unc = np.array(flux_unc_val(data_ok_filter_field))
                    #the median chi squared after is that of the good data in the primary field of the respective filter
                    logdict[filter]["median_chi2_after"] = np.median(data_ok_filter_field['forcediffimchisq']) 
                    clean_data_filt = pd.DataFrame({'time':data_ok_filter_field['jd'],'flux':data_ok_filter_field['forcediffimflux'],
                                                   'flux_unc':new_unc,'zeropoint':data_ok_filter_field['zpdiff'],
                                                   'filter':data_ok_filter_field['filter']})
                    clean_data_full = pd.concat([clean_data_full,clean_data_filt],ignore_index=True)
        
    if savepath != None:
        clean_data_full.to_json(os.path.join(savepath,str(ZTF_ID)+'_clean_data.json'))
        with open(os.path.join(savepath,str(ZTF_ID)+'_clean_log.json'),'w') as outfile:
            json.dump(logdict,outfile,indent=4,ensure_ascii=False,separators=(',',':'),cls=NumpyEncoder)
    else:
        print('No savepath provided. Dumping results, shown if verbose set to True.')
        if verbose:
            print(clean_data_full.to_markdown())
            print()
            print(logdict)
                    



sjoertpath = r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\Sjoert_Flares'
sjoertflares = pd.read_csv(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\ZTF_neoWISE_flares_acflares.dat',delimiter=' ')
catalog_coords = coord.SkyCoord(np.array(sjoertflares['ra']),np.array(sjoertflares['dec']),unit='deg')

def get_year_sjoertflares(file,catalog_coords=catalog_coords):
    #get data from file into a list and close it again
    with  open(file,'r') as data:
        data_list = data.readlines()

    #get the relevant lines from the list
    ra_line = data_list[3]
    dec_line = data_list[4]

    ra = float(ra_line[25:][:-9])
    dec = float(dec_line[25:][:-9]) #the first 26 and the last 10 characters can be deleted from the line to get only the numerical value for both RA en Dec in all files
    matchcoord = coord.SkyCoord(ra,dec,frame='icrs',unit='deg')
    idx,_,_ = coord.match_coordinates_sky(matchcoord,catalog_coords)
    idx = int(idx)
    
    ID = sjoertflares._get_value(idx,'name') #get the ZTF id of the simeon data file entry with least angular separation
    year = int(ID[3:5]) #distill the year from the ID
    return year,ID