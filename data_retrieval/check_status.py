#!/usr/bin/env python3
import re
import requests

#added by Tim 03/11:
wr = False
write = input("Do you want to write the commands to the txt file? y/n ")
if write == 'y':
    wr = True
    wget = open(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\wget.txt','w')

# Script name: check_status.py

settings = {'email': 'vdvuurst@strw.leidenuniv.nl','userpass': 'lvvd367', 'option': 'All recent jobs', 'action': 'Query Database'}

r = requests.get('https://ztfweb.ipac.caltech.edu/cgi-bin/' +\
'getBatchForcedPhotometryRequests.cgi',
auth=('ztffps', 'dontgocrazy!'),params=settings)

#print(r.text)

if r.status_code == 200:
    print("Script executed normally and queried the ZTF Batch " +\
    "Forced Photometry database.\n")
    wget_prefix = 'wget --http-user=ztffps --http-passwd=dontgocrazy! -O '
    wget_url = 'https://ztfweb.ipac.caltech.edu'
    wget_suffix = '"'
    lightcurves = re.findall(r'/ztf/ops.+?lc.txt\b',r.text)
    if lightcurves is not None:
        for lc in lightcurves:
            p = re.match(r'.+/(.+)', lc)
            fileonly = p.group(1)
            if wr:
                print(wget_prefix + " " + fileonly + " \"" + wget_url + lc +\
                wget_suffix,file=wget)
            else:
                print(wget_prefix + " " + fileonly + " \"" + wget_url + lc +\
                wget_suffix)

else:
    print("Status_code=",r.status_code,"; Jobs either queued or abnormal execution.")