import numpy as np
import os
import pandas as pd
import astropy.coordinates as coord
from tqdm import tqdm

PATH = os.path.normpath(os.getcwd() + os.sep + os.pardir)
DATAPATHmaster = os.path.join(PATH,'Data')
CODEPATH = os.path.join(PATH,'Code')
DATAPATH = os.path.join(DATAPATHmaster,'Sjoert_Flares')
print(DATAPATH)
columns = ['sindex', 'field', 'ccdid', 'qid', 'filter', 'pid', 'infobitssci', 'sciinpseeing', 'scibckgnd', 'scisigpix', 'zpmaginpsci', 'zpmaginpsciunc', 'zpmaginpscirms', 'clrcoeff', 'clrcoeffunc', 'ncalmatches', 'exptime', 'adpctdif1', 'adpctdif2', 'diffmaglim', 'zpdiff', 'programid', 'jd', 'rfid', 'forcediffimflux', 'forcediffimfluxunc', 'forcediffimsnr', 'forcediffimchisq', 'forcediffimfluxap', 'forcediffimfluxuncap', 'forcediffimsnrap', 'aperturecorr', 'dnearestrefsrc', 'nearestrefmag', 'nearestrefmagunc', 'nearestrefchi', 'nearestrefsharp', 'refjdstart', 'refjdend', 'procstatus']
dtypes = [(columns[x],float) for x in range(len(columns))]
dtypes[4] = ('filter',r'U8')

test = list(open(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\18\ZTF18aasuray\batchfp_req0000317597_lc.txt','r'))

simeon_data = pd.read_csv(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\all_nuclear_transients.csv').dropna()


# #### Two cells below take the full ZTF queried wget command list and takes out the ones that are from Sjoert's paper - these are not in Simeon's list (I think)

with open(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\wget.txt','r') as wget:
    wget_commands = wget.readlines()

wget_commands_files = [line[55:][:28] for line in wget_commands]

lines_to_delete = []
for file in os.listdir(DATAPATH):
    if file in wget_commands_files:
        i = np.where(np.array(wget_commands_files) == file)[0][0]
        lines_to_delete.append(wget_commands[i].strip("\n"))


with open(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\wget.txt', "r") as f:
    lines = f.readlines()
with open(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\wget.txt', "w") as f:
    for line in lines:
        if line.strip("\n") not in lines_to_delete:
            f.write(line)

catalog_coords = coord.SkyCoord(np.array(simeon_data['RA']),np.array(simeon_data['Dec']),unit='deg')

def get_year(file,catalog_coords=catalog_coords):
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
    
    ID = simeon_data._get_value(idx,'ztf_id') #get the ZTF id of the simeon data file entry with least angular separation
    year = int(ID[3:5]) #distill the year from the ID
    return year,ID


#change this to wget.txt instead of wget_test.txt in order to run on ALL lightcurves except for Sjoert's.
wget = open(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\wget_test.txt','r')
# wget = open(r'C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\wget.txt','r')
downloaded = open(r"C:\Users\timvd\Documents\Uni 2023-2024\First Research Project\Data\downloaded_files.txt",'a+')

down_path = os.path.join(DATAPATHmaster,"DOWNLOADED")

def wget_move_lightcurve(wget_file,log_file,down_path):
    """Downloads lightcurves per wget commands that are in seperate lines in a .txt file to the "down_path" directory 
       and moves them to the correct directory for oversightful bookkeeping.
       Is able to handle duplicates; if a downloaded file already exists at the right location the new download is deleted.
       Logs the downloads and subsequent directories in the a log file ("log_file"). New downloads only logged only if the 
       file doesn't already exist in its right location.

    Args:
        wget_file (string): Full path to the .txt file that holds the wget commands to be executed.
        log_file (string): Full path to the log_file (.txt) which holds the filename and correct path of all downloaded lightcurves. This file should never be manually touched.
        down_path (string): Full path to the directory where the wget commands will be executed and thus the files be downloaded before being moved.
    """
    #proceed only if there's not already some files in DOWNLOADED, else there might be duplicates.
    if len(os.listdir(down_path)) == 0:
        for line in wget_file:
            #Boolean that tracks if the current wget is going to download a file that already exists according to the log file.
            skip_iter = False

            #extract the file name from the wget command and note its current path in the DOWNLOADED folder
            filename = line[55:][:28]
            filepath = os.path.join(down_path,filename)

            #if the file to be downloaded (as extracted from the wget command) is already in the download log_file, skip the iteration after verifying the log file 
            # and continue to the next wget command before downloading.
            log_file.seek(0)
            for l in log_file.readlines():
                if filename in l:
                    already_downloaded_path = l[29:].strip("\n")
                    print(f'Download log shows {filename} is already downloaded at {already_downloaded_path}. Continuing to next command.')
                    skip_iter = True
                    break

            if skip_iter:
                #check if the log file is even correct; see if the file to be downloaded actually exists in the path specified by the log file.
                #if it isn't continue the download and move it to the correct place. The log file will then be correct as well - it doesn't need manual correction.
                if filename in os.listdir(already_downloaded_path):
                    continue
                else:
                    print("Log file was incorrect, suspected manual deletion of lightcurve. Continuing download.")

            #change to the download directory
            os.chdir(down_path)
            os.system(line)

            #get year and transient (ZTF) ID from the lightcurve file
            year,ID = get_year(filepath)
            yearpath = os.path.join(DATAPATHmaster,str(year))
            savepath = os.path.join(yearpath,ID)

            #if a folder for the lightcurve does not already exist, make it.
            if ID not in os.listdir(yearpath): 
                os.mkdir(savepath)

            #move the file from the download folder to its rightful location
            #if the file already exists in its right location, let the user know and delete the newly downloaded file 
            # this is because the lightcurves shouldn't be any different or better.
            #Moreover, don't log the file in the log_file if the file already exists in its location - it should then already be in the log_file.
            log_string = f'{filename} {savepath}\n'
            try:
                os.rename(filepath,os.path.join(savepath,filename))
                #if skip_iter is set to True, then the log_file will already have an entry somewhere specifying the file in the current iteration which we can trust to be true. 
                # Then the log file needn't be updated.
                if not skip_iter:
                    #if the the exact log_string is not yet in the log file, write it to the log file.
                    #Be mindful: if files are manually deleted from a certain folder and the log file is not updated manually as well then the file is not reliable anymore.
                    if log_string not in log_file.readlines():
                        log_file.write(log_string)

            except FileExistsError:
                print(f"There already is a file with this exact name ({filename}) at the specified location. Deleting the new download and updating log file.")
                os.remove(filepath)
                #if it so happens that a file is not logged but it is found now that it does in fact exist, update the log file.
                if not skip_iter:
                    #if the the exact log_string is not yet in the log file, write it to the log file.
                    #Be mindful: if files are manually deleted from a certain folder and the log file is not updated manually as well then the file is not reliable anymore.
                    if log_string not in log_file.readlines():
                        log_file.write(log_string)

    else:
        print(f'There are already {len(os.listdir(down_path))} files in the "DOWNLOADED" folder, remove these before proceeding.')

    #close files and return the working directory to the one at the start of the code.        
    wget_file.close()
    downloaded.close()
    os.chdir(PATH)

wget_move_lightcurve(wget,downloaded,down_path)


