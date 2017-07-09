#!/usr/bin/env python
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import pylab as pl
import sys

am2r=np.pi/180/60.
catalog_file='../data/AKARI/MP_1001_catp.txt'
nside=4096
fwhm=1.3*am2r #arcmin
radius = 3*fwhm
fname='../data/AKARI/AKARI_FIR_PS_mask_%d_3fwhm.fits'%nside

print "Reading catalog"
ra,dec=np.loadtxt(catalog_file,unpack=True,usecols=(2,3))
print "converting coords"
coord=SkyCoord(ra=ra, dec=dec, unit='deg').transform_to('galactic')
print "convering v"
v=hp.ang2vec(np.array(coord.l.value),np.array(coord.b.value),lonlat=True)

print "quaering disc"
masked_pix=[hp.query_disc(nside,v[i],radius) for i in range(len(v))]

mask=np.ones(hp.nside2npix(nside))
print "masking"
for p in masked_pix:
	mask[p]=0.0
hp.write_map(fname,mask,coord='G')
print "plotting"
hp.mollview(mask)
pl.show() 
