#mport numpy as np
import os
import random
from astropy.io import fits
from astropy.coordinates import SkyCoord
from read_cats import GetSDSSCat
import matplotlib.pyplot as pl
import cPickle as pickle
import gzip
import healpy as hp
import sys

import numpy as np
import astropy.constants as const

from astropy.visualization import hist
from IPython.display import display, Math, Latex
lfi_freq=[30,44,70]
hfi_freq=[100, 143, 217, 353, 545, 857]

do_ps_mask=True
if sys.argv[1]=='AKARI': freq=sys.argv[1]
else:	freq=int(sys.argv[1])

print "Processing frequency",freq

data_path = '/Users/fabbian/Work/quasar_stack/'
if freq in hfi_freq:
	if ((freq==545) or (freq==857)):
		fmap = data_path + 'data/PlanckDR2/HFI_SkyMap_%d-field-Int_2048_R2.02_full.fits'%freq
	else:
		fmap = data_path + 'data/PlanckDR2/HFI_SkyMap_%d-field-IQU_2048_R2.02_full.fits'%freq 
elif freq in lfi_freq:
	fmap = data_path + 'data/PlanckDR2/LFI_SkyMap_0%d-field-IQU_2048_R2.01_full.fits'%freq
elif freq=='AKARI':
	fmap = data_path + 'data/Akari/AKARI_WideS_1_4096_matchf_pixwin.fits'
else:
	raise ValueError('Unknown frequency')
print "Processing", fmap
if freq=='AKARI':
	fmask = data_path + 'data/PlanckDR2/HFI_Mask_GalPlane-apo2_4096_R2.00_fsky0p4.fits'
	#psfmask = data_path +'data/AKARI/AKARI_FIR_PS_mask_4096.fits'
	psfmask = data_path +'data/PlanckDR2/LFI_HFI_AKARI_Mask_PointSrc_4096_R2.00_common.fits'
	#psfmask = data_path +'data/PlanckDR2/LFI_HFI_AKARI_Mask_AllPointSrc_4096_PCCS2.fits'
	#psfmask = data_path +'data/PlanckDR2/LFI70_HFI_AKARI_Mask_AllPointSrc_4096_PCCS2.fits'
	mask = hp.read_map(fmask)
	planck = hp.read_map(fmap)

else:
	fmask = data_path + 'data/PlanckDR2/HFI_Mask_GalPlane-apo2_2048_R2.00.fits'
	#psfmask = data_path +'data/PlanckDR2/LFI_HFI_Mask_PointSrc_2048_R2.00_common.fits'
	psfmask = data_path +'data/PlanckDR2/LFI_HFI_AKARI_Mask_PointSrc_2048_R2.00_common.fits'
	#psfmask = data_path +'data/PlanckDR2/LFI_HFI_AKARI_Mask_AllPointSrc_2048_PCCS2.fits'
	#psfmask = data_path +'data/PlanckDR2/LFI70_HFI_AKARI_Mask_AllPointSrc_2048_PCCS2.fits'
	mask = hp.reorder(fits.open(fmask)[1].data['GAL040'], n2r=True) # Planck GAL mask
	#planck = hp.reorder(fits.open(fmap)[1].data['I_STOKES'], n2r=False) # Planck 353 GHz
	planck = hp.read_map(fmap) # just for LFI upgraded maps

if do_ps_mask:
	print "Use point source mask",psfmask
	mask *= hp.read_map(psfmask)

#hp.mollview(planck,norm='hist')
#pl.show()

am2r=np.pi/180/60.
am2r2=am2r**2
beam_area={30: 1189.513*am2r2, 44: 832.946*am2r2, 70: 200.742*am2r2, 100: 105.778*am2r2 , 143:59.954*am2r2,217 : 28.447*am2r2, 353 : 26.714*am2r2, 545: 26.535*am2r2, 857: 24.244*am2r2,'AKARI': 2*np.pi*(1.3*am2r/np.sqrt(8*np.log(2)))**2}

def filter_highpass_lowpass_1d(reclen, lmin, dl):
    # central_freq = samp_freq/float(reclen)
    l = np.arange(reclen)#/2+1) * central_freq
    filt = np.ones(reclen)
    filt[l<lmin-dl/2] = 0.0

    window_for_transition_to_zero = (lmin-dl/2 <= l) * (l <= lmin+dl/2)
    ind=np.where(window_for_transition_to_zero==True)[0]
    reclen_transition_window = len(filt[window_for_transition_to_zero])
    filt[window_for_transition_to_zero] = (1. - np.cos( np.pi* np.arange(reclen_transition_window) /(reclen_transition_window-1))) /2.0
    ### GF ADD
    filt[l<lmin]=0.0
    return filt
        
def GetFl(pixmap,frequency, mask=None, lmax=None, lmin=300, dl=100):
    if lmax is None:
        lmax = 3.*hp.npix2nside(pixmap.size) - 1.
    fl=pickle.load(open(data_path+'/match_filters_planck_nobl.pkl','r'))[frequency]
    ell=np.arange(lmax+1)
    return fl * filter_highpass_lowpass_1d(ell.size, lmin=lmin, dl=dl)

def FilterMap(pixmap, freq, mask=None, lmax=None, pixelwindow=False):
    if mask is None:
        mask = np.ones_like(pixmap)
    else:
        assert hp.isnpixok(mask.size)
    Fl = GetFl(pixmap,freq, mask=mask, lmax=lmax)
    alm_fname='alm_%s.fits'%freq
    if not os.path.exists(alm_fname):
	print "computing alms"
	alm = hp.map2alm(pixmap, lmax=lmax)
	hp.write_alm(alm_fname,alm)
    else:
	print "reading alms",alm_fname
	alm = hp.read_alm(alm_fname)
    if pixelwindow:
	print "Correcting alms for pixelwindow"
        pl=hp.pixwin(hp.npix2nside(pixmap.size))[:lmax+1]
    else: pl=np.ones(len(Fl))
    return hp.alm2map(hp.almxfl(alm, Fl/pl), hp.npix2nside(pixmap.size), lmax=lmax), Fl

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def Kcmb2mJysr(nu, cmb, T_cmb=2.725):
    if nu == 143:
	return 371.74 * cmb * 1e09
    elif nu == 353:
        return 287.450 * cmb * 1e09
    elif nu == 217:
        return  483.690 * cmb * 1e09
    elif nu == 100:
	return 244.1 * cmb * 1e09
    elif nu == 545:
	#return 58.04 * cmb * 1e09
	return cmb * 1e09
    elif nu == 857:
	#return 2.27 * cmb * 1e09
	return cmb*1e09 # maps already in MJy/sr
    elif nu == 70:
	return (132.07+129.75+126.05) * cmb *1e09/3 # mean value from table 3 https://arxiv.org/pdf/1502.01588.pdf
    elif nu == 44:
	return 55.735 * cmb * 1e09
    elif nu == 30:
	return 23.510 * cmb * 1e09
    elif nu == 'AKARI':
	return 1 * cmb * 1e09	

def Sky2Hpx(sky1, sky2, nside, rot=None, nest=False, rad=False):
	"""
	Converts sky coordinates, i.e. (RA,DEC), to Healpix pixel at a given resolution nside.
	By default, it rotates from EQ -> GAL
	Parameters
	----------
	sky1 : array-like
    print "computing alms"
		First coordinate, it can be RA, LON, ...
	sky2 : array-like
		Second coordinate, it can be DEC, LAT, ...
		
	nside : int
		Resolution of the Healpix pixelization scheme
	coord : str 'C','E' or 'G' [def]
		Coordinate system of the output. If coord='C', *no* rotation is applied
	nest : bool [def=False]
		If True, nested pixelation scheme is considered for the output pixels
	rad : bool [def=False]
		If True, input coordinates are expressed in radians, otherwise in degree
	Returns
	-------
	ipix : array-like
		Pixel indices corresponding to (sky1,sky2) coordinates
	"""
	sky1, sky2 = np.asarray(sky1), np.asarray(sky2)

	if rad == False: # deg -> rad
		print "converting to rad"
		theta = np.deg2rad(90.-sky2) 
		phi   = np.deg2rad(sky1) 
		print theta
	else: 
		theta = np.pi/2. - sky2
		phi   = sky1 	     

	# Apply rotation if needed (default EQ -> GAL)
	if rot is not None:
		print "rotationg coordinates",rot
		r = hp.Rotator(coord=rot, deg=False)
		theta, phi = r(theta, phi)

	npix = hp.nside2npix(nside)

	return hp.ang2pix(nside, theta, phi, nest=nest) # Converting galaxy coordinates -> pixel

if freq!='AKARI':
	planck_filt, fl = FilterMap(planck, str(freq), mask=mask,lmax=6144,pixelwindow=True)
else:
	planck_filt = planck
planck_filt_mJy_sr = Kcmb2mJysr(freq,planck_filt)
#planck_filt_mJy_sr = hp.ma(Kcmb2mJysr(freq,planck_filt))
#planck_filt_mJy_sr.mask = np.logical_not(mask)
planck_mJy = planck_filt_mJy_sr * beam_area[freq] # mJy

nside = hp.npix2nside(len(planck_mJy))
print "nside",nside

#zbins = [(1.,5.)]#2.15)]
zbins = [(0.1,5.)]
#zbins = [(1,2.15),(2.15,2.5),(2.5,5)]
zbins= [(1,2.15),(2.15,2.5),(2.5,5),(1.,5.),(0.1,5.)]
# Reading in QSO catalogs
qso_cat = GetSDSSCat(cats=['DR7', 'DR12'], discard_FIRST=True, z_DR12='Z_PIPE') # path_cats
data={'mask':psfmask,'freq':freq, 'stack':{}}
for i,(zmin, zmax) in enumerate(zbins):
    print "Redshift bin %.2f< z < %.2f"%(zmin,zmax)
    print("\t...z-bin : " + str(zmin) + " < z < " + str(zmax))
    qso = qso_cat[(qso_cat.Z >= zmin) & (qso_cat.Z <= zmax)]

    coord = SkyCoord(ra=qso.RA, dec=qso.DEC, unit='deg').transform_to('galactic')
    l = coord.l.value
    b = coord.b.value
    allpix=Sky2Hpx(l, b, nside)
    pix = np.unique(allpix) # this is basically ang2pix, i'm getting the pixels of the sources
    #masked_good_pix = mask[pix]==1 # avoid sources on apodized part of the mask
    masked_good_pix = mask[pix]!=0 # avoid sources on apodized part of the mask
    good_pix = pix[masked_good_pix]
    print "# of non overlapping sources %d/%d"%(len(good_pix),len(pix))
    fluxes = planck_mJy[good_pix] 
    rnd_pix = np.random.randint(0, hp.nside2npix(nside), fluxes.size) # da correggere
    #masked_rnd_pix = mask[rnd_pix]==1
    masked_rnd_pix = mask[rnd_pix]!=0
    good_rnd_pix = rnd_pix[masked_rnd_pix]
    fluxes_rnd = planck_mJy[good_rnd_pix]


    print 'Mean of fluxes is........: %.3f mJy' %np.mean(fluxes)
    print 'STD  of fluxes is........: %.3f mJy' %np.std(fluxes)
    print 'STD/sqrt(n)  of fluxes is: %.3f mJy' %(np.std(fluxes)/fluxes.size**.5)

    print 'Mean of **RND** fluxes is........: %.3f mJy' %np.mean(fluxes_rnd)
    print 'STD  of **RND** fluxes is........: %.3f mJy' %np.std(fluxes_rnd)
    print 'STD/sqrt(n)  of **RND** fluxes is: %.3f mJy' %(np.std(fluxes_rnd)/fluxes_rnd.size**.5)
    dict_id='z%d'%(i+1)
    data['stack'][dict_id]={'zbounds':(zmin,zmax),'flux':fluxes, 'flux_rnd':fluxes_rnd, 'pix': good_pix,'pix_rnd':good_rnd_pix}   

pickle.dump(data,open('fluxes_ps_mask_from_mask_lfi_hfi_akari_%s.pkl'%(str(freq)),'wb'))
#hp.mollview(planck_filt,norm='hist')
#pl.show()
