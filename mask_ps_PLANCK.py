#!/usr/bin/env

from astropy.io import fits
import healpy as hp
import numpy as np
import pylab as pl
am2r=np.pi/180/60
data_path='/Users/fabbian/Work/quasar_stack/data/PlanckDR2/point_sources'
freq=[30,44,70,100,143,217,353,545,857]
freq=[70,100,143,217,353,545,857]
nside=2048
radius_size=3
fname='/Users/fabbian/Work/quasar_stack/data/PlanckDR2/LFI70_HFI_Mask_AllPointSrc_2048_PCCS2.fits'
fwhm_list={30: 32.29299927, 44: 27.,70: 13.2130003, 100: 9.682, 143: 7.303,  217: 5.021,  353: 4.944, 545: 4.831, 857:4.638} # effective beam FWHM

mask=np.ones(hp.nside2npix(nside))

for nu in freq:
	if nu<80:
		catalog_name=data_path+'/COM_PCCS_%03d_R2.04.fits'%nu
	else:
		catalog_name=data_path+'/COM_PCCS_%03d_R2.01.fits'%nu
	ps=fits.open(catalog_name)
	print "Processing",catalog_name,
	l=ps[1].data[:]['GLON']
	b=ps[1].data[:]['GLAT']
	print len(l),"sources"
	v=hp.ang2vec(l,b,lonlat=True)
	radius=radius_size*fwhm_list[nu]*am2r
	masked_pix=[hp.query_disc(nside,v[i],radius) for i in range(len(v))]
	for p in masked_pix:
		mask[p]=0.0	

hp.write_map(fname,mask,coord='G')
print "plotting"
hp.mollview(mask)
pl.show()	
	

