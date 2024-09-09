# FRP_TimvdVuurst
 All code used for the First Master's Research Project (FRP) created by Tim van der Vuurst in 2023-2024. 

Jupyter notebooks, in general, are for testing and messing around. At most, relevant plots may be made in some notebooks (such as "transient_plots.ipynb"). Some notebooks may also have been used for bookkeeping purposes (e.g. "deleted_in_cleaning.ipynb"). 
The biggest, and most important, workhorse of the code is "forced_phot.py". Most of all the class ZTF_forced_phot in this file is important. 
In this class you can give the ZTF identifier of the flare you want to fit the lightcurve of (this can be done as terminal input by running "generate_lightcurve.py") and the class will work through the file system as it was made in the cleaning and downloading processes (see maps data_retrieval and Cleaning). This means that the relative directories must be the same (or you have to change the code).

Sensitive information, such as username and password for the ZTF log-in for the purpose of automatically downloading the lightcurve data, have been removed - this is highlighted in the code. 

It might very well be that the directories of certain files are hardcoded (e.g. you see r"C://mynameandfilestorage"). Sorry about this, sometimes I made this lazily without keeping generalization in mind. You can replace these with whatever directory is relevant for you or be a better person than I was and use os or shutil. 

I tried my best to document relevant code, but you might find documentation in the form of comments lacking especially in the jupyter notebooks. Feel free to contact me at timvdvuurst@gmail.com or my phone number if you have it. I'll gladly help out. 