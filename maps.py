import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

class Skymap:

	def __init__(self, fmap, psf, fnoise=None, fmask=None, color_correction=1.0, hudn=0):
		''' 
		This Class creates Objects for a set of maps/noisemaps/beams/TransferFunctions/etc., 
		at each Wavelength.
		Based on Marco Viero's class
		'''

		# Reading signal map -> smap
		smap, shd = fits.getdata(fmap, hudn, header=True)

		# Reading noise map -> nmap
		if fnoise: 
			nmap, nhd = fits.getdata(fnoise, hudn, header=True)

		# Reading mask map -> mmap
		if fmask: 
			mmap, mhd = fits.getdata(fmask, hudn, header=True)

			# Check signal/noise/mask headers ???

		# Maps pixel size	
		if 'CD2_2' in shd:
			pix = shd['CD2_2'] * 3600. # arcsec
		else:
			pix = shd['CDELT2'] * 3600. # arcses

		# Read beams
		# Check first if beam is a filename (actual beam) or a number (approximate with Gaussian)
		# if isinstance(psf, six.string_types):
		# 	beam, phd = fits.getdata(psf, 0, header = True)
		# 	#GET PSF PIXEL SIZE	
		# 	if 'CD2_2' in phd:
		# 		pix_beam = phd['CD2_2'] * 3600.
		# 	elif 'CDELT2' in phd:
		# 		pix_beam = phd['CDELT2'] * 3600.
		# 	else: pix_beam = pix
		# 	#SCALE PSF IF NECESSARY 
		# 	if np.round(10.*pix_beam) != np.round(10.*pix):
		# 		raise ValueError("Beam and Map have different size pixels")
		# 		scale_beam = pix_beam / pix
		# 		pms = np.shape(beam)
		# 		new_shape=(np.round(pms[0]*scale_beam),np.round(pms[1]*scale_beam))
		# 		pdb.set_trace()
		# 		kern = rebin(clean_nans(beam),new_shape=new_shape,operation='ave')
		# 		#kern = rebin(clean_nans(beam),new_shape[0],new_shape[1])
		# 	else: 
		# 		kern = clean_nans(beam)
		# 	self.psf_pixel_size = pix_beam
		# else:
		# 	sig = psf / 2.355 / pix
		# 	kern = gauss_kern(psf, np.floor(psf * 8.), pix)

		self.map = np.nan_to_num(smap) * color_correction

		if fnoise:
			self.noise = np.nan_to_num(nmap) * color_correction
		else:
			self.noise = None

		if fmask:
			self.mask = np.nan_to_num(mmap)
		else:
			self.mask = None

		self.header = shd
		self.pixel_size = pix
		self.psf = psf
		self.wavelength = shd['WAVELNTH']

		# Setting up the WCS for coords transformation
		self.w = WCS(self.header)

		# Patch coordinates
		self.dec_min, self.ra_max = self.w.wcs_pix2world(self.map.shape[1],self.map.shape[0],0)
		self.dec_max, self.ra_min = self.w.wcs_pix2world(0,0,0)

	def printhd(self):
		print(repr(self.header))

	def beam_area_correction(self,beam_area):
		self.map *= beam_area * 1e6
		
	# def add_weights(self,file_weights):
	# 	weights, whd = fits.getdata(file_weights, 0, header = True)
	# 	#pdb.set_trace()
	# 	self.noise = clean_nans(1./weights,replacement_char=1e10)
