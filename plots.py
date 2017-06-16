import numpy as np
import os
import matplotlib.pyplot as plt
import cPickle as pk
import gzip as gz
from photutils import CircularAnnulus
from photutils import CircularAperture
from photutils import aperture_photometry
from IPython import embed

# SPIRE channels
lambdas = [250, 350, 500]
psf     = {250:17.8, 350:24.0, 500:35.2} # in arcsec 
factor  = {250:469./36., 350:831./64., 500:1804./144.} # Jy/beam -> Jy/pixel 
reso    = {250:6., 350:8., 500:12.} # in arcsec 
positions = {250: (25.5, 25.5), 350: (19.5,19.5), 500:(13.5,13.5)}

# H-ATLAS patches
patches = ['G9', 'G12', 'G15']#, 'NGP', 'SGP']

# Results folder
results_folder = 'results/'

zmin = 0.1
zmax = 5.

zbins = [(1.0,2.15), (2.15, 2.5), (2.5,5.)]

# planck

i=1
plt.figure(figsize=(20,6))
for idz, zbin in enumerate(zbins):
	print zbin
	plt.subplot(1,3,i)
	plt.title(str(zbin[0]) + r' $ < z < $ '+ str(zbin[1]))
	j = 0
	i +=1
	cuts = {}
	for patch in patches:
		cuts[patch] = pk.load(gz.open('results/Planck353_patch'+patch+'_lambda850_zmin'+str(zbin[0])+'_zmax'+str(zbin[1])+'.pkl','r'))
		cuts[patch] = np.asarray(cuts[patch]['maps'])
		j += cuts[patch].shape[0]
		cuts[patch] = np.mean(cuts[patch], axis=0)
	print j
	plt.imshow(np.mean([cuts[patch] for patch in patches], axis=0))
	print np.mean([cuts[patch] for patch in patches], axis=0)
	plt.colorbar()

plt.show()


i=1
plt.figure(figsize=(20,6))
for idz, zbin in enumerate(zbins):
	print zbin
	plt.subplot(1,3,i)
	plt.title(str(zbin[0]) + r' $ < z < $ '+ str(zbin[1]))
	j = 0
	i +=1
	cuts = {}
	for patch in patches:
		cuts[patch] = pk.load(gz.open('results/AKARI_patch'+patch+'_lambda90_zmin'+str(zbin[0])+'_zmax'+str(zbin[1])+'.pkl','r'))
		cuts[patch] = np.asarray(cuts[patch]['maps'])
		j += cuts[patch].shape[0]
		cuts[patch] = np.mean(cuts[patch], axis=0)
	print j
	plt.imshow(np.mean([cuts[patch] for patch in patches], axis=0))
	# print np.mean([cuts[patch] for patch in patches], axis=0)
	plt.colorbar()

plt.show()

embed()




