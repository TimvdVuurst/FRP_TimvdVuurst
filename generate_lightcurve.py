from forced_phot import ZTF_forced_phot
import os
import numpy as np
from astropy.stats import bayesian_blocks


testztf = 'ZTF19aaejtoy'

def ztf_dir_finder(ztf: str,datapath: str = r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data') -> str:
    year = ztf[3:5]
    dpath = os.path.join(datapath,year,ztf)
    return dpath

Flag = True
while Flag:
    ztfname = str(input("Input ZTF name of lightcurve to be generated (X to quit): "))

    if ztfname in ['x','X']:
        Flag = False
        break

    figtype = str(input("Raw or fit? ")).lower()
    while figtype not in ['raw','fit']:
        figtype = str(input("Raw or fit? ")).lower()
    save = str(input('Save? (y/n) ')).lower()

    while save not in ['y','n']:
        save = str(input('Save? (y/n) ')).lower()
    if save == 'y':
        save = True
    else:
        save = False

    plottitle = str(input('Plot title? Type x for standard. '))
    if plottitle.lower() == 'x':
        plottitle = None
    
    print("Generating lightcurve...")
    ztfobj = ZTF_forced_phot(ztf_dir_finder(ztfname),onefilt_crosscor=False)
    # print(np.array(ztfobj.guesses),np.array(ztfobj.boundings[0]))
    # print(np.array(ztfobj.boundings[1]))
    # print('g')
    # print(np.array(ztfobj.guesses_g),np.array(ztfobj.boundings_g[0]))
    # print(np.array(ztfobj.guesses_g))
    # print('r')
    # print(np.array(ztfobj.guesses_r),np.array(ztfobj.boundings_r[0]))
    # print(np.array(ztfobj.guesses_r))

    if ztfname == testztf:
        cleandata = ztfobj.clean_data
        # print(cleandata)
        # break/
        BBlocks = bayesian_blocks(cleandata['time'].values,x=cleandata['flux'].values,sigma=cleandata['flux_unc'].values,fitness='measures',p0=0.1)
        print(BBlocks)
        

    if figtype == 'raw':
        ztfobj.plot_clean_unclean_data(save=save,title=plottitle)
    else:

        if not save:
            spath = None
        else:
            spath = r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Plots'
        ztfobj.fit(plot=True,savepath=spath,fit_i=True,title=plottitle) #performs fitting and plots the figure 