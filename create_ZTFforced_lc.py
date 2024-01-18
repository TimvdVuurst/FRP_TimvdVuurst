#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:45:35 2021

@author: ekhammer 
@author: sjoertvv

example: python3 create_ZTFforced_lc.py -name Varys -date dr12
"""
import numpy as np
import json, argparse, os, pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from astropy.time import Time

import warnings
warnings.filterwarnings("ignore")

import sjoert
#import sjoert.stellar

tde_sample0 = pickle.load(open('../collection/known_tde_sample.pickle', 'rb'))

plt.close()
#plt.style.use('ekhammer')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', action='store', nargs=1, dest='name',
                        help='Name of target to create LC for.')

    parser.add_argument('-skip_baseline', action='store_true', default=False,
                        dest='skip_baseline',
                        help='Dont do any baseline correction')


    parser.add_argument('-public', action='store_true', default=False,
                        dest='public',
                        help='Filters out only public data.')

    parser.add_argument('-interactive', action='store', default=True,
                        dest='interactive',
                        help='Show prompt to change baseline by hand')


    parser.add_argument('-wait', action='store_true', default=False,
                        dest='wait',
                        help='Wait after final plot')

    parser.add_argument('-testgals', action='store_true', default=False,
                        dest='do_testgals',
                        help='Run for comparison galaxies')

    
    parser.add_argument('-date', action='store', default='dr18_public',
                        dest='date_str',
                        help='local date/DR key (eg, Nov21, d15, dr18_public)')
    
    parser.add_argument('-dr', action='store', default='dr18_public',
                        dest='date_str',
                        help='local date/DR key (eg, Nov21, dr15, dr18_public)')


    parser.add_argument('-softcuts', action='store_true', default=False,
                        dest='softcuts',
                        help='Option to make less strict quality cuts (see code)')



    args = parser.parse_args()
    name = args.name[0]
    public = args.public
    date_str = args.date_str 
    skip_baseline = args.skip_baseline
    interactive = int(args.interactive)
    do_testgals = args.do_testgals
    softcuts = args.softcuts
    print ('interactive?', interactive)
    print ('test gals?', do_testgals)

# resolve the name (autocomplete)
idx = [i for i,x in enumerate(tde_sample0) if name  in x['name']]
name = tde_sample0[idx]['name'][0]

# find the peak from the stored time of peak
peak_year = tde_sample0[idx]['peak_year'][0]
peak_jd = sjoert.simtime.mjdtojd(sjoert.simtime.yeartomjd(peak_year))


# ---
# here follow some hardcoded exceptions
# manual expections for width
widths = {}
widths['SansaStark'] = 100      # not much baseline, just enough
widths['AryaStark'] = 120       # not much baseline, just enough
widths['Xian'] = 500            # very slow rise
widths['JorahMormont'] = 190
widths['Osha'] = 600            # slow rise
widths['Podrick'] = 500         # slow rise
widths['JonSnow'] = 200         # more (post-peak) baseline
widths['TrapperWolf'] = 400     # less (pre-peak) baseline, slow rise
widths['ATLAS22kjn'] = 500      # less (pre-peak) baseline, slow rise

alllow_postpeak_baseline = ['NedStark', 'JonSnow', 'GendryBaratheon']

# the baseline images for Gendry are claimed to be from 2018, pre-peak
if 'Gendry' in name:
    skip_baseline = True


"""Define the quality cuts."""
def q_cuts(ZTF, softcuts=False):
    '''
    update for Jul 2023
    main difference:  forcediffimfluxap/forcediffimflux less strict
    '''
    
    DC2Jy = sjoert.stellar.mag2flux(ZTF['zpdiff'])

    app_flux_ratio = ZTF['forcediffimfluxap'] / ZTF['forcediffimflux']
    app_psf_diff = (ZTF['forcediffimfluxap'] - ZTF['forcediffimflux']) #/ np.abs(ZTF['forcediffimfluxap'])

    #(np.abs(np.log10(app_flux_ratio)) <1.5) *\

    iok = \
          (ZTF['procstatus'] != '56') *\
          (ZTF['scisigpix'] < 20) *\
          (ZTF['sciinpseeing'] < 4.0) * (ZTF['zpmaginpscirms'] < 0.05) 
                
          
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

    # hack for checking cuts
    plt.hist(app_psf_diff,range=[-300,300],bins=30)
    plt.pause(0.1)
    key = input()

    print ('# of raw points     :', len(ZTF))    
    print ('# of points rejected:', len(ZTF)-sum(iok))

    return iok

print('\nCreating LC for', name)

"""Create master legend."""
legend_elements = [Line2D([0], [0], marker='o', color='w', label='ZTF-g',
                          markerfacecolor='g', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='ZTF-r',
                          markerfacecolor='r', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='ZTF-i',
                          markerfacecolor='gold', markersize=10)]

"""Set current time for x-axis labels."""
now = Time.now().jd


colors = {'ZTF_g': 'g', 'ZTF_r': 'r', 'ZTF_i': 'brown'}

"""Read in forced photometry table from IPAC."""
ZTF0 = pd.read_csv("./sources/"+name+"/"+name+"_forcedphot_"+date_str+".txt",
                    engine='python', skiprows=56,
                    comment='#', skipinitialspace=True,
                    sep='\s', index_col=0,
                    names=('index', 'field', 'ccdid', 'qid', 'filter',
                           'pid', 'infobitssci', 'sciinpseeing',
                           'scibckgnd', 'scisigpix', 'zpmaginpsci',
                           'zpmaginpsciunc', 'zpmaginpscirms',
                           'clrcoeff', 'clrcoeffunc', 'ncalmatches',
                           'exptime', 'adpctdif1', 'adpctdif2',
                           'diffmaglim', 'zpdiff', 'programid', 'jd',
                           'rfid', 'forcediffimflux',
                           'forcediffimfluxunc', 'forcediffimsnr',
                           'forcediffimchisq', 'forcediffimfluxap',
                           'forcediffimfluxuncap', 'forcediffimsnrap',
                           'aperturecorr', 'dnearestrefsrc',
                           'nearestrefmag', 'nearestrefmagunc',
                           'nearestrefchi', 'nearestrefsharp',
                           'refjdstart', 'refjdend', 'procstatus'))

ZTF0['procstatus'] = ZTF0['procstatus'].astype('str')

ZTF0 = ZTF0.reset_index(drop=True)

# apply cuts (before finding the baseline # new in 2022)

do_local_clean = True 

if do_local_clean:
    iok = q_cuts(ZTF0, softcuts=softcuts)

# this doesnt work yet because not the same object
else:
    import clean_lc
    ZTF = clean_lc.clean_lc(ZTF0)

ZTF = ZTF0[iok]
ZTF = ZTF.reset_index(drop=True)

# conver to micro Janksy; add new column
DC2Jy = sjoert.stellar.mag2flux(ZTF['zpdiff'])
ZTF['forcediffimflux_muJy'] = ZTF['forcediffimflux']*DC2Jy*1e6
ZTF['forcediffimfluxunc_muJy'] = ZTF['forcediffimfluxunc']*DC2Jy*1e6

# select only MSIP
# at some point the Partnership data also becomes public, 
# so it's better to do a seperate call to IPC with a non-ZTF username (see requestZTFforced_Sjoert.py)
if public:
    ipub = ZTF['programid'] == 1
    ZTF = ZTF[ipub]
    ZTF = ZTF.reset_index(drop=True)    




"""Save unbinned, pre-baseline corrected LC."""
prelc = plt.figure()
ax = prelc.add_subplot(111)

# Separate filters and convert to muJy
ztfg = ZTF[ZTF['filter'] == 'ZTF_g']

ztfr = ZTF[ZTF['filter'] == 'ZTF_r']

ztfi = ZTF[ZTF['filter'] == 'ZTF_i']

# Now to plot
ax.scatter(now-ztfg['jd'], ztfg['forcediffimflux_muJy'], c='g', alpha=0.75, s=10)
ax.scatter(now-ztfr['jd'], ztfr['forcediffimflux_muJy'], c='r', alpha=0.75, s=10)
ax.scatter(now-ztfi['jd'], ztfi['forcediffimflux_muJy'], c='gold', alpha=0.75, s=10)
ax.axhline(0.0, c='k', ls=':')
ax.set_xlabel('Days ago (since {0:0.1f})'.format(Time.now().decimalyear))
ax.set_ylabel(r'Unbinned flux ($\mathrm{\mu}$Jy)')
ax.set_title(str(name)+' raw light curve')
ax.legend(handles=legend_elements, loc='best')
ax.invert_xaxis()
plt.savefig('./lc_figs/' + date_str+'/'+
            name + '_raw_muJy.pdf')
#plt.show()


fields = np.unique(ZTF['field'])  # unique fields for this target

"""Separate the fields and get each filter observed in that field.
   Then calculate and apply the baseline correction for each filter/field
   combo and update it in the table."""
    

def do_basecorr(fieldno, ZTF, slog, peak_jd, width=None):
    """Separates fields into filters, checks if the uncertainties need to be
       rescaled, and applies a baseline correction if needed."""

    idx_field = ZTF['field'] == fieldno

    thisfield = ZTF.loc[idx_field] # get idxs of field
    filters = np.unique(thisfield['filter'])  # get unique filters for this field


    for j in filters:

        thisfield_flt = thisfield[thisfield['filter'] == j] # get all rows of specific filter
        print ('\nfilter={0}, {1} observations '.format(j, len(thisfield_flt) ))

        refjdstart =max(thisfield_flt['refjdstart'])
        refjdend =max(thisfield_flt['refjdend'])
        
        peak_min_ref = peak_jd-refjdend
        print ('Time difference between latest ref image and peak of light curve: {0:0} days'.format(peak_min_ref))

        # start with some general guess for the baseline
        startjd = min(ZTF['jd'])-1

        auto_remove = 0
        caught_in_ref = 0
        # to allow a post-peak baseline, we need to be at least two years after last ref
        
        if (peak_min_ref<60) and (max(thisfield_flt['jd'])-refjdend<365*2):
            print ('TDE entirely caught in reference; we should remove this field')
            auto_remove = 1

        elif (peak_min_ref<60) and not do_testgals:
            if name in alllow_postpeak_baseline:
                print ('TDE seems caught in the ref frame, lets try a post-peak window')
                startjd = max(thisfield_flt['jd']-180)
                caught_in_ref = 1
            else:
                print ('TDE seems caught in the ref frame, post-peak window not allowed for this source')
                auto_remove = 1

        # try to guess the width
        elif width is None:
            width = max(180, peak_jd-100-startjd)       

        if width is None: 
            width = 180

        # scale uncertainty by chi2 of PSF fit for the difference image 
        
        # this was the original approach, but the correction should per epoch        
        # avgchi = np.mean(chisq)
        # if np.abs(1-avgchi) > 0.1:
        #     print('Multiplying uncertainties with typical factor:', np.sqrt(avgchi)
        # slog[fieldno][j]['rescale_by_psf_chi2'] = np.sqrt(avgchi)        


        DC2muJy = sjoert.stellar.mag2flux(thisfield_flt['zpdiff'])*1e6
        forcediffimflux = thisfield_flt['forcediffimflux_muJy']
        forcediffimfluxunc = thisfield_flt['forcediffimfluxunc_muJy']
        chisq = thisfield_flt['forcediffimchisq']
        zpdiff = thisfield_flt['zpdiff']
        jd = thisfield_flt['jd']
        
        forcediffimfluxunc *= np.clip(np.sqrt(chisq),0.1,10)
        ZTF['forcediffimfluxunc_muJy'].loc[thisfield_flt.index] = forcediffimfluxunc         
        print ('chi2 of forced PSF fit [min, median, max]:', np.min(chisq), np.median(chisq), np.max(chisq))

        
        if j == 'ZTF_g':
            c = 'g'

        elif j == 'ZTF_r':
            c = 'r'
        elif j == 'ZTF_i':
            c = 'gold'


        
        # check we have enough data in this field
        ibase = (jd > startjd) * (jd < startjd + width)
        if len(jd)<10:
            print ('Not enough data in this field+filter:', len(jd))
            auto_remove = True
        
        # if not enough data in first guess of baseline, default is removal
        elif (sum(ibase)<5):
            print ('Not enough observations in baseline window:',sum(ibase))
            auto_remove = True
        
        # we also want enough postpeak data
        elif sum(jd>peak_jd)<3:
            print ('Not enough post-peak observations:', sum(jd>peak_jd))
            auto_remove = True
        


        cont = 0
        while cont == 0:

            ibase = (jd > startjd) * (jd < startjd + width)
            testflux = forcediffimflux[ibase]
            testfluxunc = forcediffimfluxunc[ibase]            
            
            basecorr = np.nanmedian(testflux)
            ibase_ok = np.abs((testflux-basecorr)/testfluxunc)<7
            
            basecorr_rms = np.nanstd(testflux[ibase_ok])
            basecorr_sigma = basecorr_rms/np.sqrt(len(testflux[ibase_ok]))
            
            if sum(ibase)>1:
                chi2_base = ( sum((testflux[ibase_ok]-basecorr)**2 / (testfluxunc[ibase_ok])**2))
                chi2dof_base = chi2_base/ (len(testflux[ibase_ok]-1))
            

            print('# obs in proposed baseline   :', len(testflux))
            print('# of 7-sig outliers in basel :', sum(ibase_ok==False))
            print('Proposed baseline flux       : {0:5.2f}'.format(basecorr))
            print('rms in baseline              : {0:5.2f}'.format(basecorr_rms))
            print('Typical statical uncertainty : {0:5.2f}'.format(np.median(testfluxunc[ibase_ok])))
            #print('rms/sqrt(N)                  : {0:5.1f}'.format(basecorr_sigma))
            print('significance                 : {0:5.1f}'.format(basecorr/basecorr_sigma))
            if sum(ibase)>1:
                print('chi2/dof of baseline obs     : {0:5.1f}'.format(chi2dof_base))

            plt.clf()
            plt.ion()
            plt.errorbar(jd, forcediffimflux,forcediffimfluxunc, fmt='o', c=c, ms=3, alpha=0.8)
            plt.axhline(np.nanmedian(testflux))
            plt.axvspan(startjd, startjd+width,ls='-', alpha=0.4, label='proposed baseline')
            plt.axvspan(refjdstart, refjdend, color='orange',alpha=0.25, ls='--', label='ref images')
            plt.axvline(peak_jd, color='grey', ls=':',alpha=0.8, label='est. peak (all bands)')
            plt.legend(loc=1)
            
            if auto_remove:
                plt.title('remove?')

            plt.show()
            plt.pause(0.2)

            run_prompt =True
            cont = 1 

            if skip_baseline:
                cont = 2 
                run_prompt = False
            
            if auto_remove:
                cont = 3 
                run_prompt = False
                if interactive:
                    key = input('remove field [y]/n')
                    if key == 'n':
                        run_prompt = True
            
            
            if run_prompt and interactive:

                key = (input('\nChange startjd, width of baseline? \
                    \npress 0: yes \
                    \npress 1 (or return): apply this baseline correction; \
                    \npress 2: no corrections at all; \
                    \npress 3: remove this field+filter entirely \
                    \n'))

                if key=='':
                    cont=1
                else:
                    cont=int(key)

            if cont == 0:
            
                print ('current width is: {0} (press return to keep this)'.format(width))
                key = input('Width of baseline: ')
                if key !='':
                    width = float(key)
                
                print ('current start is: {0:0.1f} (press return to keep this)'.format(startjd))
                key = input('Start of baseline: ') 
                if key!='':
                    startjd = float(key)


        # update the log with general info
        slog[fieldno][j]['N_field'] = len(thisfield_flt)
        slog[fieldno][j]['N_base'] = len(testflux)
        slog[fieldno][j]['caught_in_ref'] = int(caught_in_ref)

        if cont == 3:
            
            ZTF = ZTF.drop(thisfield_flt.index)
            print ('Removing {0} datapoints'.format(len(thisfield_flt)))
            
            slog[fieldno][j]['removed'] = 1

        elif cont == 2:
            print('No baseline correction performed.')            

            slog[fieldno][j]['no_correction'] = 1
        
        # We need at least 10 observations to define the baseline        
        # Make the baseline correction for this filter in this field
        elif len(testflux) >= 5:

            print ('Applying correction')            
            scale_factor_sys = np.clip(np.sqrt(chi2dof_base),0.5,10) 
            print ('Scaling factor for sytematic uncertainty :', scale_factor_sys)
            ZTF['forcediffimfluxunc_muJy'].loc[thisfield_flt.index] = forcediffimfluxunc * scale_factor_sys
            ZTF['forcediffimflux_muJy'].loc[thisfield_flt.index] -= basecorr
            
            # chi2_base = ( sum((testflux-basecorr)**2 / (forcediffimfluxunc[ibase]*scale_factor_sys)**2))
            # chi2dof_base = chi2_base/ (len(testflux-1))
            # print ('chi2/dof for baseline obs (after rescaling):', chi2dof_base)

            slog[fieldno][j]['postpeak_baseline'] = int(startjd>peak_jd)
            slog[fieldno][j]['base_start_jd'] = startjd
            slog[fieldno][j]['base_width_day'] = width
            slog[fieldno][j]['base_corr_flux_muJy'] = basecorr
            slog[fieldno][j]['base_corr_rms_muJy'] = basecorr_rms
            slog[fieldno][j]['base_corr_sig'] = basecorr/basecorr_sigma
            slog[fieldno][j]['base_chi2/dof'] = chi2dof_base
            slog[fieldno][j]['scale_factor_sys_unc'] = scale_factor_sys

            # flash the corrected flux on the screen
            plt.errorbar(ZTF['jd'].loc[thisfield_flt.index], 
                    ZTF['forcediffimflux_muJy'].loc[thisfield_flt.index],
                    ZTF['forcediffimfluxunc_muJy'].loc[thisfield_flt.index], 
                    fmt='x', lw=1)
            plt.pause(0.3)
            if args.wait:
                key = input('next?')

        else:
            print('No baseline correction performed because not enough baseline obs')        
            slog[fieldno][j]['no_correction'] = 1


    return ZTF, width


# defaults widths for all fields
width = 150 
if date_str=='Jul22':
    width = 150
elif date_str=='Jun22': 
    width = 75 
elif date_str=='Nov21':
    date_str = 30 

startjd = min(ZTF0['jd'])-1

# quick and dirty peak estimate based on forced photo light curve
if not do_testgals:
    peak_jd = ZTF['jd'].loc[np.argmax(ZTF['forcediffimflux'])] 


# make a log
slog = {int(i):{ 'ZTF_g':{},'ZTF_r':{},'ZTF_i':{} } for i in fields}




# loop over fields, do the baseline correction
# default is auto config the width
width = None 


# but for some source we alreayd know what we want
if name in widths:
    width = widths[name]

for i in fields:
    ZTF, width = do_basecorr(i, ZTF, peak_jd=peak_jd, slog=slog, width=width)

#plt.clf()
#
#plt.ion()
#plt.hist(DC2Jy*np.abs(ZTF0['forcediffimfluxunc']), bins=100)
#plt.show()
#
#unc = float(input('Uncertainty cutoff: '))

if len(ZTF)==0:
    print ('No data left...')


#time_peak = float(ZTF['jd'].loc[ZTF['forcediffimflux']==ZTF['forcediffimflux'].max()])
time_peak = peak_jd

dt = ZTF['jd']-time_peak

# itime = dt>-365 # only go back 1 year time
# ZTF = ZTF[iok*itime]

# keep only the relevant columns
full_lc = ZTF[['jd', 'filter', 'zpdiff', 'forcediffimflux_muJy',
               'forcediffimfluxunc_muJy']]

#convert to Jansky 
full_lc['forcediffimflux_muJy'] =  full_lc['forcediffimflux_muJy']*1e-6
full_lc['forcediffimfluxunc_muJy'] = full_lc['forcediffimfluxunc_muJy']*1e-6

# give better names
final_lc = full_lc.rename(columns={'forcediffimflux_muJy': 'flux_Jy',
                                   'forcediffimfluxunc_muJy': 'eflux_Jy',
                                   'zpdiff': 'zp_Jy'})


plt.clf()

for i in final_lc.index:
    plt.errorbar(final_lc['jd'][i]-time_peak, 
            final_lc['flux_Jy'][i]*1e6,
            final_lc['eflux_Jy'][i]*1e6,
            lw=0.7,
            fmt='.',
            color=colors[final_lc['filter'][i]], alpha=0.7)

plt.ylabel(r'Flux ($\mathrm{\mu}$Jy)')
plt.xlabel('Days since peak ({0:0.1f})'.format(Time(time_peak, format='jd').decimalyear ))
plt.axhline(0, alpha=0.3, color='k')

if public:
    pass
else:
    plt.savefig('./lc_figs/' + date_str+'/'+
                name + '_final_fullcadence.pdf')

plt.pause(0.5)
#plt.show()
if args.wait:
    key =input('done')

if not os.path.isdir('./lcs/'+date_str+'/logs'):
    os.mkdir('./lcs/'+date_str+'/logs')
if not os.path.isdir('./lcs/'+date_str+'/logs/'):
    os.mkdir('./lcs/'+date_str+'/logs/')

if public:
    final_lc.to_csv('./lcs_public/'+date_str+'/'+name+'_publiconly.txt')
    f = open('./lcs_public/'+date_str+'/logs/'+name+'_baselinecorr_log_json','w')
    json.dump(slog,f , indent=3)
    f.close()
else:
    final_lc.to_csv('./lcs/'+date_str+'/'+name+'_fullcadence.txt')
    f = open('./lcs/'+date_str+'/logs/'+name+'_baselinecorr_log.json','w')
    json.dump(slog, f, indent=3)
    f.close()
