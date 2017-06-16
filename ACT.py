import numpy as np
import matplotlib.pyplot as plt
import sys, random
sys.path.append('/Users/fbianchini/Research/FlatSpec/')
from Spec2D import *
from Sims import *
from scipy.integrate import simps
from astropy.io import fits

from read_cats import GetSDSSCat
from maps import Skymap, Healpixmap

from IPython import embed

arcmin2rad = np.pi / 180. / 60. 


def GetCutout(pixmap, pixcent, npix):
	"""
	Extracts a cutout of size (npix,npix) centered in (pixcent[0], pixcent[1]) from a bigger map pixmap
	"""
	x, y = pixcent
	x, y = np.int(x), np.int(y)
	return pixmap[y-npix:y+npix+1, x-npix:x+npix+1]

def GoGetStack(x, y, skymap, mask, npix):
	results = {}
	results['maps'] = []
	
	for i in xrange(len(x)):
		cutmask = GetCutout(mask, (x[i],y[i]), npix=npix)#         print cutmask.shape
		isgood = True if (np.mean(cutmask) == 1) & (cutmask.shape == (2*npix+1,2*npix+1)) else False # Do analysis if all the cutout within the footprint
		if isgood: # Cutout is *completely* in the footprint
			results['maps'].append(GetCutout(skymap, (x[i],y[i]), npix=npix))
		else: # discard object
			pass

	results['maps'] = np.asarray(results['maps'])			
	
	return results

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x=1, sigma_y=1, theta=0., offset=0):
	xo = float(xo)
	yo = float(yo)    
	a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
	return g#.ravel()

def Add2dGauss(x, y, data, npix, amp=5.):
	for i in xrange(len(x)):
		X, Y = np.meshgrid(np.arange(y[i]-npix, y[i]+npix+1), np.arange(x[i]-npix, x[i]+npix+1))
		try:
			data[y[i]-npix:y[i]+npix+1, x[i]-npix:x[i]+npix+1] += twoD_Gaussian((Y,X), amplitude=amp, xo=x[i], yo=y[i], sigma_x=1, sigma_y=1)
		except:
			pass
	return data

def FilterMap(pixmap, lmin, lmax, reso, pad=1):
	if pad !=1:
		pixmap = np.pad(pixmap, ((pixmap.shape[0], pixmap.shape[0])), mode='constant', constant_values=0.)
	
	ft = np.fft.fftshift(np.fft.fft2(pixmap))
	mask = GetLMask(pixmap.shape[0], dx=reso, lmin=lmin, lmax=lmax, shift=True)
	
	return np.fft.ifft2(np.fft.fftshift(ft*mask)).real
	
def GetLMask(nx, dx, ny=None, dy=None, shift=False, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None):
	""" 
	return a Fourier mask for the pixelization associated with this object which is zero over customizable ranges of L. 
	"""
	if ny == None: ny = nx
	mask      = np.ones((ny, nx), dtype=np.complex)
	lx, ly    = GetLxLy(nx, dx, ny=ny, dy=dy, shift=shift)
	L         = GetL(nx, dx, dy=dy, ny=ny, shift=shift)
	if lmin  != None: mask[ np.where(L < lmin) ] = 0.0
	if lmax  != None: mask[ np.where(L >=lmax) ] = 0.0
	if lxmin != None: mask[ np.where(np.abs(lx) < lxmin) ] = 0.0
	if lymin != None: mask[ np.where(np.abs(ly) < lymin) ] = 0.0
	if lxmax != None: mask[ np.where(np.abs(lx) >=lxmax) ] = 0.0
	if lymax != None: mask[ np.where(np.abs(ly) >=lymax) ] = 0.0

	return mask

def GetLxLy(nx, dx, ny=None, dy=None, shift=True):
	""" 
	Returns two grids with the (lx, ly) pair associated with each Fourier mode in the map. 
	If shift=True (default), \ell = 0 is centered in the grid
	~ Note: already multiplied by 2\pi 
	"""
	if ny is None: ny = nx
	if dy is None: dy = dx
	
	dx *= arcmin2rad
	dy *= arcmin2rad
	
	if shift:
		return np.meshgrid( np.fft.fftshift(np.fft.fftfreq(nx, dx))*2.*np.pi, np.fft.fftshift(np.fft.fftfreq(ny, dy))*2.*np.pi )
	else:
		return np.meshgrid( np.fft.fftfreq(nx, dx)*2.*np.pi, np.fft.fftfreq(ny, dy)*2.*np.pi )

def GetL(nx, dx, ny=None, dy=None, shift=True):
	""" 
	Returns a grid with the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode in the map. 
	If shift=True (default), \ell = 0 is centered in the grid
	"""
	lx, ly = GetLxLy(nx, dx, ny=ny, dy=dy, shift=shift)
	return np.sqrt(lx**2 + ly**2)

def GetACTbeam(nx, dx, dy=None, ny=None, shift=True, filepath='/Users/fbianchini/Documents/beams_AR2_2010_season_130224.dat.txt'):
	""" 
	Returns 2D FT of ACT beam.
	If shift=True (default), \ell = 0 is centered in the grid
	"""
	if ny is None: ny = nx
	if dy is None: dy = dx
 
	l, bl = np.loadtxt(filepath, unpack=True)
	L = GetL(nx, dx, ny=ny, dy=dy, shift=shift)
	idx = np.where(L > 20000)
	xx = np.interp(L, l, bl)
	xx[idx] = 0.
	return xx

def Get2dSpectra(pixmap, dx, dy=None, shift=True):
	ny, nx = pixmap.shape
	if dy is None: 
		dy = dx
	dx *= arcmin2rad
	dy *= arcmin2rad

	if shift:
		ft = np.fft.fftshift(np.fft.fft2(pixmap))
	else:
		ft = np.fft.fft2(pixmap)
	return (ft * np.conj(ft)).real * (dx*dy)/(nx*ny)

def MatchFilter(pixmap, dx, dy=None, beam='ACT', lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None, shift=True):
	ny, nx = pixmap.shape

	# Anisotropic filter
	F = GetLMask(nx, dx, dy=dy, ny=ny, shift=shift, lmin=lmin, lmax=lmax, lxmin=lxmin, lxmax=lxmax, lymin=lymin, lymax=lymax)
	
	
	# Beam
	if beam == 'ACT':
		B = GetACTbeam(nx, dx, dy=None, ny=None, shift=shift)
		print beam
		
	T = Get2dSpectra(pixmap, dx, dy=dy, shift=shift)
	T2_inv = np.nan_to_num(1./np.abs(T))**2
	BFTB = np.conj(B) * F * T2_inv * B #reduce(np.dot, [np.conj(B), F, T2_inv, B])
	integral = simps(simps(BFTB, x=GetLxLy(nx, dx, dy=dy, ny=ny)[1][:,0]), x=GetLxLy(nx, dx, dy=dy, ny=ny)[0][0,:])

	return np.nan_to_num(F * np.conj(B) * T2_inv) / integral #reduce(np.dot, [F, B, T2_inv]) / integral 

def MatchFilterMap(pixmap, dx, dy=None, ny=None, beam=0., lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None, shift=False):
	mf = MatchFilter(pixmap, dx, dy=dy, beam=beam, shift=shift, lmin=lmin, lmax=lmax, lxmin=lxmin, lxmax=lxmax, lymin=lymin, lymax=lymax)
	return np.fft.ifft2(np.fft.fft2(pixmap)*mf).real

def make_2d_gaussian_beam(nx, dx, fwhm, shift=True):
	"""Already in L-space"""
	L = GetL(nx, dx, shift=shift)
	idx = np.where(L > 20000)

	B_l = bl(fwhm, L.max())

	xx = np.interp(L, np.arange(B_l.size), B_l)
	xx[idx] = 0.

	return xx

tmap = fits.open('/Volumes/LACIE_SHARE/Data/ACT/ACT_220_equ_season_3_1way_v3_src_free.fits')
hits = fits.open('/Volumes/LACIE_SHARE/Data/ACT/ACT_220_equ_season_3_1way_hits_v3.fits')     
tmap_weighted = np.sqrt(hits[0].data[:,:])*tmap[0].data[:,:]/np.sum(np.sqrt(hits[0].data[:,:])) 

# Some parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nrnd = 200000

nu = 220

# Redshift bins
zbins = [(1.,2.15), (2.15,2.50),(2.50,5.0)]

# Reading in QSO catalogs
qso_cat = GetSDSSCat(cats=['DR7', 'DR12'], discard_FIRST=True, z_DR12='Z_PIPE') # path_cats

# ACT/ACTpol channels
nus = [220]

# Cutouts (half-)size in pixels 
npix    = {148:10,
		   220:10}  # ACTpol pixels are ~ 30 arcsec, cutouts are 10'x10' 

# Beam @ different freq 
psf     = {148: 1.4*60.,
		   220: 1.4*60.} # arcsec

factor  = {140: 1.,
		   220: 1.} # \muK/pix -> mJy/beam

# Pixel resolution
reso    = {148: 30.,
		   220: 30.} # arcsec

positions = {148: (10.5, 10.5),
			 220: (10.5, 10.5)}

# boxsize = {250:51, 350:39, 500:27}
# boxsize    = {250:101, 350:77, 500:51} # SPIRE pixels are ~ 6/8/12 arcsec -> cutouts are 5'
# positions = {250: (101/2., 101/2.), 350: (77/2.,77/2.), 500:(51/2.,51/2.)}

# H-ATLAS patches
patches = ['G9', 'G12', 'G15']#, 'NGP', 'SGP']

# Fits files
fmap = '/Volumes/LACIE_SHARE/Data/ACT/ACT_220_equ_season_3_1way_v3_src_free.fits'
fmask = '/Volumes/LACIE_SHARE/Data/ACT/ACT_220_equ_season_3_1way_hits_v3.fits'
fnoise = None#data_path + 'ACT/ACTPol_'+str(nu)+'_D56_PA2_S2_1way_noise.fits'

# ANALYSIS ON FILTERED MAPS
fluxmap = Skymap(fmap, psf[nu], fnoise=fnoise, fmask=fmask, color_correction=1.0)

print('Filtering map')
fluxmap.map = MatchFilterMap(tmap_weighted, 0.5, lmin=1000, lxmin=100, beam='ACT')
print('Done')

mask = fits.open(fmask)
mask = mask[0].data
mask[mask < 70000.] = 0.
mask[(mask >= 70000)] = 1.

# print("\t...the mean of the map is : %.5f Jy/beam" %(np.mean(fluxmap.map[np.where(fluxmap.mask == 1.)])))
results_filtered = {}

# Loop over redshift bins
for idz, (zmin, zmax) in enumerate(zbins)[:1]:
	print("\t...z-bin : " + str(zmin) + " < z < " + str(zmax))
	qso = qso_cat[(qso_cat.Z >= zmin) & (qso_cat.Z <= zmax)]
	# print len(qso)

	# Remember that x refers to axis=0 and y refers to axis=1 -> MAP[y,x]
	x, y = fluxmap.w.wcs_world2pix(qso.RA, qso.DEC, 0) # 0 because numpy arrays start from 0
	good_idx = (~np.isnan(x)) & (~np.isnan(y))
	x = x[good_idx]
	y = y[good_idx]

	results_filtered[idz] = GoGetStack(x, y, fluxmap.map, mask, npix[nu])

	# Saving stuff
	results_filtered[idz]['nu'] = nu
	results_filtered[idz]['zbin'] = (zmin, zmax)

	
embed()
