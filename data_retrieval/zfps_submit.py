#!/usr/bin/env python3
import requests
import json
import os

# Script name: zfps_submit.py

def submit_post(ra_list,dec_list):
    ra = json.dumps(ra_list)
    print(ra)
    dec = json.dumps(dec_list)
    print(dec)
    jds = 2458301.0000 # start JD for all input target positions. Set to 1st of July 2018
    jdstart = json.dumps(jds)
    print(jdstart)
    jde = 2460231.0000 # end JD for all input target positions. Set to 13-10-2023
    jdend = json.dumps(jde)
    print(jdend)
    email = 'vdvuurst@strw.leidenuniv.nl' # email you subscribed with.
    userpass = 'lvvd367' # password that was issued to you.
    payload = {'ra': ra, 'dec': dec,
    'jdstart': jdstart, 'jdend': jdend,
    'email': email, 'userpass': userpass}
    # fixed IP address/URL where requests are submitted:
    url = 'https://ztfweb.ipac.caltech.edu/cgi-bin/batchfp.py/submit'
    r = requests.post(url,auth=('ztffps', 'dontgocrazy!'), data=payload)
    print("Status_code=",r.status_code)

#--------------------------------------------------
# Main calling program. Ensure "List_of_RA_Dec.txt"
# contains your RA Dec positions.

#ADDED BY TIM: asks user for subfolder for better bookkeeping possibilities. 
#Make sure that the txt file is always called 'List_of_RA_Dec.txt', or just add another input that asks what the file is called that holds the ra,dec coordinates
cwd = os.getcwd()
fileloc = input("In which folder is the txt file stored? ")
file = os.path.join(cwd,fileloc,'List_of_RA_Dec.txt')
with open(file) as f:
    lines = f.readlines()
f.close()

print("Number of (ra,dec) pairs =", len(lines))

ralist = []
declist = []
i = 0
for line in lines:
    x = line.split()
    radbl = float(x[0])
    decdbl = float(x[1])

    raval = float('%.7f'%(radbl))
    decval = float('%.7f'%(decdbl))
                   
    ralist.append(raval)
    declist.append(decval)
    i = i + 1
    rem = i % 1500 # Limit submission to 1500 sky positions.
    if rem == 0:
        submit_post(ralist,declist)
        print(i)
        ralist = []
        declist = []

if len(ralist) > 0:
    submit_post(ralist,declist)
    print(i)
    pass

exit(0)