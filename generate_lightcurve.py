from forced_phot import ZTF_forced_phot
import os

testztf = 'ZTF20aaaaflr'

def ztf_dir_finder(ztfname: str,datapath: str = r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data') -> str:
    year = ztfname[3:5]
    dpath = os.path.join(datapath,year,ztfname)
    return dpath

Flag = True
while Flag:
    ztfname = str(input("Input ZTF name of lightcurve to be generated (X to quit): ")).lower()
    if ztfname == 'x':
        break
    print("Generating lightcurve...")
    ztfobj = ZTF_forced_phot(ztf_dir_finder(ztf_dir_finder(ztfname)))
    ztfobj.fit(plot=True,savepath=None) #performs fitting and plots the figure 