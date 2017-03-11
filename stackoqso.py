import numpy as np
import os
import random
from astropy.io import fits
from read_cats import GetSDSSCat
from maps import Skymap
import matplotlib.pyplot as plt
from IPython import embed
import cPickle as pickle
import gzip
from photutils import CircularAnnulus
from photutils import CircularAperture
from photutils import aperture_photometry

data_path = '/Volumes/LACIE_SHARE/Data/H-ATLAS/'

def GetCutout(pixmap, pixcent, npix):
	"""
	Extracts a cutout of size (npix,npix) centered in (pixcent[0], pixcent[1]) from a bigger map pixmap
	"""
	x, y = pixcent
	x, y = np.int(x), np.int(y)
	return pixmap[y-npix:y+npix+1, x-npix:x+npix+1]

def GoGetStack(x, y, skymap, mask, npix, noise=None, extras=None, z=None):
	results = {}
	results['maps'] = []

	if noise is not None:
		results['noise'] = []

	# if z is not None:
	# 	results['z'] = []

	if extras is not None:
		for name in extras.iterkeys():
			results[name] = []
 
	for i in xrange(len(x)):
		cutmask = GetCutout(mask, (x[i],y[i]), npix=npix)
		isgood = True if np.mean(cutmask) == 1 else False # Do analysis if all the cutout within the footprint

		# print x[i], y[i], isgood, np.mean(cutmask)

		if isgood: # Cutout is *completely* in the footprint
			results['maps'].append(GetCutout(skymap, (x[i],y[i]), npix=npix))
			if noise is not None:
				results['noise'].append(GetCutout(noise, (x[i],y[i]), npix=npix))
			# if z is not None:
			# 	results['z'].append(z[i])
			if extras is not None:
				for name in extras.iterkeys():
					results[name].append(extras[name][i])				
		else: # discard object
			pass

	return results

if __name__ == '__main__':
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	print '...Hello, let us get started...'

	# Results folder
	results_folder = 'results/'
	estimate_background = False

	# Some parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	nrnd = 50000

	# Redshift bins
	# zbins = [(0.1, 1.), (1.,2.), (2.,3.), (3.,4.), (4.,5.)]
	# zbins = [(0.1, 5.)]
	# zbins = [(1., 5.)]
	zbins = [(1.,2.15), (2.15,2.50),(2.50,5.0)]

	# Reading in QSO catalogs
	qso_cat = GetSDSSCat(cats=['DR7', 'DR12'], discard_FIRST=True, z_DR12='Z_PIPE') # path_cats
	
	# SPIRE channels
	lambdas = [250, 350, 500]
	npix    = {250:25, 350:19, 500:13} # SPIRE pixels are ~ 6/8/12 arcsec -> cutouts are 5'
	psf     = {250:17.8, 350:24.0, 500:35.2} # in arcsec 
	factor  = {250:469./36., 350:831./64., 500:1804./144.} # Jy/beam -> Jy/pixel 
	reso    = {250:6., 350:8., 500:12.} # in arcsec 
	positions = {250: (25.5, 25.5), 350: (19.5,19.5), 500:(13.5,13.5)}
	# boxsize = {250:51, 350:39, 500:27}
	# boxsize    = {250:101, 350:77, 500:51} # SPIRE pixels are ~ 6/8/12 arcsec -> cutouts are 5'
	# positions = {250: (101/2., 101/2.), 350: (77/2.,77/2.), 500:(51/2.,51/2.)}

	# H-ATLAS patches
	patches = ['G9', 'G12', 'G15']#, 'NGP', 'SGP']

	# QSO features to be included
	extras_names = [
		'Z',
	    'JMAG',			 # J magnitude 2MASS
	    'ERR_JMAG',		 # Error J magnitude 2MASS
	    'HMAG',			 # H magnitude 2MASS
	    'ERR_HMAG',		 # Error H magnitude 2MASS
	    'KMAG',			 # K magnitude 2MASS
	    'ERR_KMAG',		 # Error K magnitude 2MASS
	    'W1MAG',		 # w1 magnitude WISE
	    'ERR_W1MAG',     # Error w1 magnitude WISE
	    'W2MAG',		 # w2 magnitude WISE
	    'ERR_W2MAG',     # Error w2 magnitude WISE
	    'W3MAG',		 # w3 magnitude WISE
	    'ERR_W3MAG',     # Error w3 magnitude WISE
	    'W4MAG',		 # w4 magnitude WISE
	    'ERR_W4MAG',     # Error w4 magnitude WISE
	    'CC_FLAGS',       # WISE contamination and confusion flag
	    'UKIDSS_MATCHED',# UKIDSS matched
	    'YFLUX',         # Y-band flux density [Jy]
	    'YFLUX_ERR',     # Error in Y-band density flux [Jy]
	    'JFLUX',         # J-band flux density [Jy]
	    'JFLUX_ERR',     # Error in J-band density flux [Jy]
	    'HFLUX',         # H-band flux density [Jy]
	    'HFLUX_ERR',     # Error in H-band density flux [Jy]
	    'KFLUX',         # K-band flux density [Jy]
	    'KFLUX_ERR',     # Error in K-band density flux [Jy]
	    'PSFFLUX_U',       # Flux in the ugriz bands (not corrected for Gal extin) [nanomaggies] 
	    'PSFFLUX_G',       # Flux in the ugriz bands (not corrected for Gal extin) [nanomaggies] 
	    'PSFFLUX_R',       # Flux in the ugriz bands (not corrected for Gal extin) [nanomaggies] 
	    'PSFFLUX_I',       # Flux in the ugriz bands (not corrected for Gal extin) [nanomaggies] 
	    'PSFFLUX_Z',       # Flux in the ugriz bands (not corrected for Gal extin) [nanomaggies] 
	    'IVAR_PSFFLUX_U',  # Inverse variance of ugriz fluxes
	    'IVAR_PSFFLUX_G',  # Inverse variance of ugriz fluxes
	    'IVAR_PSFFLUX_R',  # Inverse variance of ugriz fluxes
	    'IVAR_PSFFLUX_I',  # Inverse variance of ugriz fluxes
	    'IVAR_PSFFLUX_Z',  # Inverse variance of ugriz fluxes
	    'EXTINCTION_U',       # Galactic extintion in the 5 SDSS bands (from Schlegel+98)      
	    'EXTINCTION_G',       # Galactic extintion in the 5 SDSS bands (from Schlegel+98)      
	    'EXTINCTION_R',       # Galactic extintion in the 5 SDSS bands (from Schlegel+98)      
	    'EXTINCTION_I',       # Galactic extintion in the 5 SDSS bands (from Schlegel+98)      
	    'EXTINCTION_Z',       # Galactic extintion in the 5 SDSS bands (from Schlegel+98)      
	    'EXTINCTION_RECAL_U', # Galactic extintion in the 5 SDSS bands (from Schafly&Finkbeiner11)      
	    'EXTINCTION_RECAL_G', # Galactic extintion in the 5 SDSS bands (from Schafly&Finkbeiner11)      
	    'EXTINCTION_RECAL_R', # Galactic extintion in the 5 SDSS bands (from Schafly&Finkbeiner11)      
	    'EXTINCTION_RECAL_I', # Galactic extintion in the 5 SDSS bands (from Schafly&Finkbeiner11)      
	    'EXTINCTION_RECAL_Z', # Galactic extintion in the 5 SDSS bands (from Schafly&Finkbeiner11)      
	    'FLUX02_12KEV',     # Total flux (0.2 - 12 keV) XMM [erg/cm^2/s]
	    'ERR_FLUX02_12KEV', # Error in total flux (0.2 - 12 keV) XMM [erg/cm^2/s]
	    'FLUX02_2KEV',      # Soft flux (0.2 - 2 keV) XMM [erg/cm^2/s]
	    'ERR_FLUX02_2KEV',  # Error in soft flux (0.2 - 2 keV) XMM [erg/cm^2/s]
	    'FLUX2_12KEV',      # Hard flux (2 - 12 keV) XMM [erg/cm^2/s]
	    'ERR_FLUX2_12KEV',  # Error in hard flux (2 - 12 keV) XMM [erg/cm^2/s]
	    'LUM05_2KEV',       # Soft X-ray luminostiy [erg/s]
	    'LUM2_12KEV',       # Hard X-ray luminostiy [erg/s]
	    'LUM02_2KEV'        # Total X-ray luminostiy [erg/s]
	    ]

	# Loop over wavelengths ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	print("...starting stacking...")
	for lambda_ in lambdas:
		print("...lambda : " + str(lambda_))
		
		# Loop over patches
		for patch in patches:
			print("\t...patch : " + patch)
			num = ''.join(x for x in patch if x.isdigit())
			
			# Fits files
			fmap = data_path + patch + '/HATLAS_GAMA' + str(num) + '_DR1_BACKSUB' + str(lambda_) + '.FITS'
			fmask = data_path + patch + '/HATLAS_GAMA' + str(num) + '_DR1_MASK' + str(lambda_) + '.FITS'
			fnoise = data_path + patch + '/HATLAS_GAMA' + str(num) + '_DR1_SIGMA' + str(lambda_) + '.FITS'

			fluxmap = Skymap(fmap, psf[lambda_], fnoise=fnoise, fmask=fmask, color_correction=1.0)
			
			print("\t...the mean of the map is : %.5f Jy/beam" %(np.mean(fluxmap.map[np.where(fluxmap.mask == 1.)])))

			# Loop over redshift bins
			for zmin, zmax in zbins:
				print("\t...z-bin : " + str(zmin) + " < z < " + str(zmax))
				qso = qso_cat[(qso_cat.Z >= zmin) & (qso_cat.Z <= zmax)]
				# print len(qso)

				# Remember that x refers to axis=0 and y refers to axis=1 -> MAP[y,x]
				x, y = fluxmap.w.wcs_world2pix(qso.RA, qso.DEC, 0) # 0 because numpy arrays start from 0
				good_idx = (~np.isnan(x)) & (~np.isnan(y))
				x = x[good_idx]
				y = y[good_idx]
				# z = qso.Z[good_idx].values

				extras = {}
				for name in extras_names:
					extras[name] = qso[name][good_idx].values
				# embed()

				results = GoGetStack(x, y, fluxmap.map, fluxmap.mask, npix[lambda_], noise=fluxmap.noise, extras=extras)
				
				# Saving stuff
				results['lambda'] = lambda_
				results['patch']  = patch
				results['zbin'] = (zmin, zmax)
				# results['color_correction'] = 

				print("\t\t...stacking on data terminated...")
				print("\t\t...saving to output...\n")
				pickle.dump(results, gzip.open(results_folder + '/patch'+patch+'_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'.pkl','wb'), protocol=2)
				# pickle.dump(results, open('results/patch'+patch+'_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'.pkl', 'wb'))

				if estimate_background:
					print("\t\t...start stacking on random...")
					rnd_x = [random.uniform(0, fluxmap.map.shape[1]) for i in xrange(nrnd)]
					rnd_y = [random.uniform(0, fluxmap.map.shape[0]) for i in xrange(nrnd)]
					results_rnd = GoGetStack(rnd_x, rnd_y, fluxmap.map, fluxmap.mask, npix[lambda_], noise=fluxmap.noise)
					nrnd_ = len(results_rnd['maps'])
					
					maps_rnd = np.asarray(results_rnd['maps'])
					noise_rnd = np.asarray(results_rnd['noise'])
					
					fluxes = np.zeros(maps_rnd.shape[0])

					apertures = CircularAperture(positions[lambda_], r=2*psf[lambda_]/reso[lambda_])
					for i in xrange(maps_rnd.shape[0]):
						fluxes[i] = aperture_photometry(maps_rnd[i]/(factor[lambda_])/1e-3, apertures).field('aperture_sum')[0]

					results_rnd['maps'] = np.mean(maps_rnd, axis=0)
					results_rnd['noise'] = np.mean(noise_rnd, axis=0)
					results_rnd['fluxes'] = fluxes

					results_rnd['nrnd_'] = nrnd_
					results_rnd['lambda'] = lambda_
					results_rnd['patch']  = patch
					results_rnd['zbin'] = (zmin, zmax)

					print("\t\t...stacking on random terminated...")
					print("\t\t...saving to output...\n")
					pickle.dump(results_rnd, gzip.open(results_folder + 'patch'+patch+'_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'_RND.pkl','wb'), protocol=2)

