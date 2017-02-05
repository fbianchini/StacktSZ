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

data_path = '/Volumes/LACIE_SHARE/Data/H-ATLAS/'

def GetCutout(pixmap, pixcent, npix=38):
	"""
	Extracts a cutout of size (npix,npix) centered in (pixcent[0], pixcent[1]) from a bigger map pixmap
	"""
	x, y = pixcent
	x, y = np.int(x), np.int(y)
	return pixmap[y-npix:y+npix+1, x-npix:x+npix+1]

def GoGetStack(x, y, skymap, mask, npix, noise=None):
	results = {}
	results['maps'] = []
	if noise is not None:
		results['noise'] = []

	for i in xrange(len(x)):
		cutmask = GetCutout(mask, (x[i],y[i]), npix=npix)
		isgood = True if np.mean(cutmask) == 1 else False # Do analysis if all the cutout within the footprint

		if isgood: # Cutout is *completely* in the footprint
			results['maps'].append(GetCutout(skymap, (x[i],y[i]), npix=npix))
			if noise is not None:
				results['maps'].append(GetCutout(noise, (x[i],y[i]), npix=npix))
		else: # discard object
			pass

	return results

if __name__ == '__main__':
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	print '...Hello, let us get started...'

	# Some parameter ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	npix = 38 # SPIRE pixels are ~ 8 arcsec -> cutouts are 5' wide ~ 17/12/8 beams at 250/350/500 microns

	# Redshift bins
	zbins = [(0.1, 5.)]

	# Reading in QSO catalogs
	qso_cat = GetSDSSCat(cats=['DR7', 'DR12'], discard_FIRST=True, z_DR12='Z_PIPE') # path_cats

	# SPIRE channels
	lambdas = [250, 350, 500]
	psf = {250:18.1, 350:25.2, 500:36.6} # in arcsec 

	# H-ATLAS patches
	patches = ['G9', 'G12', 'G15']#, 'NGP', 'SGP']


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
			
			# Loop over redshift bins
			for zmin, zmax in zbins:
				qso = qso_cat[(qso_cat.Z >= zmin) & (qso_cat.Z <= zmax)]

				# Remember that x refers to axis=0 and y refers to axis=1 -> MAP[y,x]
				x, y = fluxmap.w.wcs_world2pix(qso.RA, qso.DEC, 0) # 0 because numpy arrays start from 0
				good_idx = (~np.isnan(x)) & (~np.isnan(y))
				x = x[good_idx]
				y = y[good_idx]

				results = GoGetStack(x, y, fluxmap.map, fluxmap.mask, npix, noise=fluxmap.noise)
				
				# Saving stuff
				results['lambda'] = lambda_
				results['patch']  = patch
				results['zbin'] = (zmin, zmax)
				# results['color_correction'] = 

				print("\t...stacking on data terminated...")
				print("\t...saving to output...")
				# pickle.dump(results, open('results/patch'+patch+'_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'.pkl', 'wb'))

				# if estimate_background:
				# 	ra_rnd, dec_rnd 


