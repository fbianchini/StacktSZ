import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed

path_cats = {'DR7': '/Volumes/LACIE_SHARE/Data/SDSS/dr7qso.fit',
			 'DR12':'/Volumes/LACIE_SHARE/Data/SDSS/DR12Q.fits'
			 }

kws_cats = {'DR7': 
            ['SDSSJ',   # SDSS-DR7 designation hhmmss.ss+ddmmss.s (J2000)
		    'RA',       # Right Ascension in decimal degrees (J2000)
		    'DEC',      # Declination in decimal degrees (J2000)
		    'Z',        # Redshift 
		    'FIRSTMAG', # FIRST MATCHED FLAG (> 0 if QSO have FIRST counterparts)
		    'JMAG',	    # J magnitude 2MASS
		    'JMAGERR',  # Error J magnitude 2MASS
		    'HMAG',		# H magnitude 2MASS
		    'HMAGERR',	# Error H magnitude 2MASS
		    'KMAG',		# K magnitude 2MASS
		    'KMAGERR',  # Error K magnitude 2MASS
		    ]
		    ,

		    'DR12':
			['SDSS_NAME',    # SDSS-DR12 designation hhmmss.ss+ddmmss.s (J2000)
		    'RA',            # Right Ascension in decimal degrees (J2000)
		    'DEC',           # Declination in decimal degrees (J2000)
		    'Z_VI',          # Redshift from visual inspection
		    'Z_PIPE',        # Pipeline redshift estimate
		    'ERR_ZPIPE',     # Error on pipeline redshift estimate
		    'Z_PCA',         # PCA redshift estimate
		    'ERR_ZPCA',      # Error on PCA redshift estimate
		    'SDSS_DR7',      # DR7 matching flag (= 1 if the QSO was observed by DR7)
		    'FIRST_MATCHED', # FIRST MATCHED FLAG (= 1 if QSO have FIRST counterparts)
		    'JMAG',			 # J magnitude 2MASS
		    'ERR_JMAG',		 # Error J magnitude 2MASS
		    'HMAG',			 # H magnitude 2MASS
		    'ERR_HMAG',		 # Error H magnitude 2MASS
		    'KMAG',			 # K magnitude 2MASS
		    'ERR_KMAG',		 # Error K magnitude 2MASS
		    'W1MAG',		 # w1 magnitude WISE
		    'ERR_W1MAG',     # Error w1 magnitude WISE
		    'W2MAG',		 # w2 magnitude WISE
		    'ERR_W2MAG',     # Error w1 magnitude WISE
		    'W3MAG',		 # w3 magnitude WISE
		    'ERR_W3MAG',     # Error w1 magnitude WISE
		    'W4MAG',		 # w4 magnitude WISE
		    'ERR_W1MAG',     # Error w1 magnitude WISE
		    'CC_FLAGS'       # WISE contamination and confusion flag
		    ]}

def GetSDSSCat(cats=['DR7', 'DR12'], path_cats=path_cats, kws_cats=kws_cats, 
			   hdun=1, memmap=True, discard_FIRST=True, z_DR12='Z_PIPE'):
	"""
	Read in the SDSS-DRx QSO catalogs.
	Since these cats are usually massive, will only read the columns specified by kws_cats.
	By default, it will read and merge SDSS-DR7 and DR12, removing the duplicated sources
	(i.e. if a source is in both cats, will just keep the DR12 version) 

	Parameters
	----------
	cats : list of str
		Name of the cats release

	path_cats : dict
		Path to the fits file containing the cats (ex: path_cats = {'mycat': path_to_mycat})

	path_cats : dict
		Name of the fits columns to read in (ex: kws_cats = {'mycat': ['RA', 'DEC', 'FLUX']})	

	hdun : int
		Which HDUlist to read (default = 1)

	discard_FIRST : bool
		If True, remove all sources which have a counterpart in FIRST (default = True)

	z_DR12 : str
		Which type of redshift to use (look in kws_cats['DR12'] for more options)

	Returns
	-------


	"""
	# Loop over QSO cats
	df_cat = {}
	for cat in cats:
		tmp_cat  = fits.open(path_cats[cat], memmap=memmap)[hdun]
		tmp_dict = {}

		# Creating a temporary dictionary with the columns of interest
		for kws in kws_cats[cat]:
			tmp_dict[kws] = tmp_cat.data[kws].byteswap().newbyteorder() # Otherwise you get big-endian compiler crappy problem

		# Creating the data frame containing the cat
		df_cat[cat] = pd.DataFrame(tmp_dict)

		# Dropping FIRST counteparts
		if discard_FIRST:
			assert ( ('FIRSTMAG' in kws_cats[cat] ) or ('FIRST_MATCHED' in kws_cats[cat]) )
			if cat == 'DR7':
				df_cat[cat] = df_cat[cat][df_cat[cat].FIRSTMAG <= 0]
			elif cat == 'DR12':				
				df_cat[cat] = df_cat[cat][df_cat[cat].FIRST_MATCHED < 1]

		# Some columns renaming before merging the cats
		if cat == 'DR7':
			df_cat[cat] = df_cat[cat].rename(columns={'SDSSJ':'SDSS_NAME'})  
			if 'JMAGERR' in kws_cats:
				df_cat[cat] = df_cat[cat].rename(columns={'JMAGERR':'ERR_JMAG'})  
			if 'HMAGERR' in kws_cats:
				df_cat[cat] = df_cat[cat].rename(columns={'HMAGERR':'ERR_HMAG'})  
			if 'KMAGERR' in kws_cats:
				df_cat[cat] = df_cat[cat].rename(columns={'KMAGERR':'ERR_KMAG'})  

		if cat == 'DR12':
			df_cat[cat] = df_cat[cat].rename(columns={'Z_PIPE':'Z'})  

		del tmp_dict, tmp_cat

	df_merged = pd.concat([x for x in df_cat.values()])

	if len(cats) > 1:
		df_merged = df_merged.drop_duplicates(subset='SDSS_NAME', keep='last')

	return df_merged

if __name__ == "__main__":
	df = GetSDSSCat()
	embed()