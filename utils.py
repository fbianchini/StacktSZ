import numpy as np
import cPickle as pickle
import gzip as gzip
import matplotlib.pyplot as plt

from photutils import CircularAnnulus
from photutils import CircularAperture
from photutils import aperture_photometry

from astropy.visualization import hist
from astropy.modeling import models, fitting
from scipy import optimize

psf     = {250:17.8, 350:24.0, 500:35.2} # in arcsec 
factor  = {250:469./36., 350:831./64., 500:1804./144.} # Jy/beam -> Jy/pixel 
# reso    = {250:6., 350:8., 500:12.} # in arcsec 
# positions = {250: (25.5, 25.5), 350: (19.5,19.5), 500:(13.5,13.5)}
# boxsize = {250:51, 350:39, 500:27}

lambda_sur = {'SDSS':[0.35, 0.48, 0.62, 0.79, 0.91],
              'WISE':[3.4, 4.6, 12, 22],
              'UKIDSS':[1.02, 1.25, 1.49, 2.03],
              '2MASS':[]
              } 

WmHz2mJy = 1e29

def lambda_to_GHz(lam):
    """
    Converts from wavelenght (in micron) to GHz
    """
    hz  = 3e8/(lam*1e-6)
    ghz = 1e-9*hz
    return ghz

def GHz_to_lambda(ghz):
    """
    Converts from GHz to wavelenght (in micron)
    """
    lam = 3e8/ghz * 1e-3
    return lam


def WISEMag2mJy(mags, W): # mags in Vega system
    if np.isscalar(mags) or (np.size(mags) == 1):
        if W == 'W1':
            return 309.540 * 10**(-mags/2.5) * 1e3
        if W == 'W2':
            return 171.787 * 10**(-mags/2.5) * 1e3
        if W == 'W3':
            return 31.674 * 10**(-mags/2.5) * 1e3
        if W == 'W4':
            return 8.363 * 10**(-mags/2.5) * 1e3
    else:
        return np.asarray([ WISEMag2mJy(mag, W) for mag in mags ])

def TwoMASSMag2mJy(mags, W): # mags in Vega system
    if np.isscalar(mags) or (np.size(mags) == 1):
        if W == 'J':
            return 1594 * 10**(-mags/2.5) * 1e3
        if W == 'H':
            return 1024 * 10**(-mags/2.5) * 1e3
        if W == 'K_S':
            return 666.7 * 10**(-mags/2.5) * 1e3
    else:
        return np.asarray([ TwoMASSMag2mJy(mag, W) for mag in mags ])

def SDSSMag2mJy(mags): # mags in nanomaggies
    return mags * 3.631e-3 

class CutoutAnalysis(object):
    def __init__(self, folder, lambdas=[250,350,500], patches=['G9','G12','G15'], zbins=[(1.0,5.0)], extras_names=['Z'], size=5.):
        if size == 5.:
            self.positions = {250: (25.5, 25.5), 350: (19.5,19.5), 500:(13.5,13.5), 90:(20,20)}
            self.boxsize = {250:51, 350:39, 500:27, 90:40}


        self.lambdas = lambdas
        self.patches = patches
        self.zbins   = zbins

        self.bkd = {}
        self.cuts = {}
        self.fluxes_bkd = {}
        self.extras = {}
        self.noise = {}

        # Loop over wavebands
        for lambda_ in lambdas:

            if lambda_ == 900: # AKARI
                self.bkd[lambda_] = {}
                self.cuts[lambda_] = {}

                # Loop over z-bins
                for idz, (zmin, zmax) in enumerate(self.zbins):
                    cuts_ = pickle.load(gzip.open('/Volumes/LACIE_SHARE/Data/AKARI_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'.pkl','rb'))
                    self.cuts[lambda_][patch][idz] = np.asarray(cuts_['maps'])
                    self.noise[lambda_][patch][idz] = np.asarray(cuts_['noise'])
                    bkd_ = pickle.load(gzip.open ('/Volumes/LACIE_SHARE/Data/AKARI_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'.pkl','rb'))
                    self.bkd[lambda_][patch][idz] = bkd_['maps']
                    # self.fluxes_bkd[lambda_][patch][idz] = bkd_['fluxes']
                    self.extras[lambda_][patch][idz] = {}

            else:
                self.bkd[lambda_] = {}
                self.cuts[lambda_] = {}
                self.fluxes_bkd[lambda_] = {}
                self.extras[lambda_] = {}
                self.noise[lambda_] = {}

                # Loop over patches
                for patch in self.patches:
                        self.bkd[lambda_][patch] = {}
                        self.cuts[lambda_][patch] = {}
                        self.fluxes_bkd[lambda_][patch] = {}
                        self.extras[lambda_][patch] = {}
                        self.noise[lambda_][patch] = {}

                        # Loop over z-bins
                        for idz, (zmin, zmax) in enumerate(self.zbins):
                            cuts_ = pickle.load(gzip.open(folder + '/patch'+patch+'_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'.pkl','rb'))
                            self.cuts[lambda_][patch][idz] = np.asarray(cuts_['maps'])
                            try:
                                self.noise[lambda_][patch][idz] = np.asarray(cuts_['noise'])
                            except:
                                pass
                            bkd_ = pickle.load(gzip.open(folder + '/patch'+patch+'_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'_RND.pkl','rb'))
                            self.bkd[lambda_][patch][idz] = bkd_['maps']
                            # self.fluxes_bkd[lambda_][patch][idz] = bkd_['fluxes']
                            self.extras[lambda_][patch][idz] = {}

                            # Other QSO specs?
                            for name in extras_names:
                                try:
                                    self.extras[lambda_][patch][idz][name] = np.asarray(cuts_[name])
                                except:
                                    pass

    # def GetHist(self, xtr, lmbd=250):
    #     if (xtr == 'JMAG') or (xtr == 'HMAG') or (xtr == 'KMAG'):
    #         good_idx =
    #     hist(, bins="knuth", histtype='step', label='250')

    def GetFluxesCats(self, survey):
        if survey == 'WISE' or survey == 'wise':
            
            self.w1mag = {}
            self.w2mag = {}
            self.w3mag = {}
            self.w4mag = {}

            self.meanw1 = {}
            self.meanw2 = {}
            self.meanw3 = {}
            self.meanw4 = {}

            self.medw1 = {}
            self.medw2 = {}
            self.medw3 = {}
            self.medw4 = {}

            self.low1 = {}
            self.hiw1 = {}
            self.low2 = {}
            self.hiw2 = {}
            self.low3 = {}
            self.hiw3 = {}
            self.low4 = {}
            self.hiw4 = {}

            self.errw1mag = {}
            self.errw2mag = {}
            self.errw3mag = {}
            self.errw4mag = {}

            self.errw1 = {}
            self.errw2 = {}
            self.errw3 = {}
            self.errw4 = {}

            for idz, zbin in enumerate(self.zbins):

                ccflag = np.concatenate([self.extras[350][patch][idz]['CC_FLAGS'].copy() for patch in self.patches], axis=0)

                self.w1mag[idz] = WISEMag2mJy(np.concatenate([self.extras[350][patch][idz]['W1MAG'].copy() for patch in self.patches], axis=0), W='W1')[ccflag=='0000']
                self.w2mag[idz] = WISEMag2mJy(np.concatenate([self.extras[350][patch][idz]['W2MAG'].copy() for patch in self.patches], axis=0), W='W2')[ccflag=='0000']
                self.w3mag[idz] = WISEMag2mJy(np.concatenate([self.extras[350][patch][idz]['W3MAG'].copy() for patch in self.patches], axis=0), W='W3')[ccflag=='0000']
                self.w4mag[idz] = WISEMag2mJy(np.concatenate([self.extras[350][patch][idz]['W4MAG'].copy() for patch in self.patches], axis=0), W='W4')[ccflag=='0000']

                self.errw1mag[idz] = WISEMag2mJy(np.concatenate([self.extras[350][patch][idz]['ERR_W1MAG'].copy() for patch in self.patches], axis=0), W='W1')[ccflag=='0000']
                self.errw2mag[idz] = WISEMag2mJy(np.concatenate([self.extras[350][patch][idz]['ERR_W2MAG'].copy() for patch in self.patches], axis=0), W='W2')[ccflag=='0000']
                self.errw3mag[idz] = WISEMag2mJy(np.concatenate([self.extras[350][patch][idz]['ERR_W3MAG'].copy() for patch in self.patches], axis=0), W='W3')[ccflag=='0000']
                self.errw4mag[idz] = WISEMag2mJy(np.concatenate([self.extras[350][patch][idz]['ERR_W4MAG'].copy() for patch in self.patches], axis=0), W='W4')[ccflag=='0000']

                self.meanw1[idz] = np.sum(self.w1mag[idz]/self.errw1mag[idz]**2)/np.sum(1./self.errw1mag[idz]**2)
                self.meanw2[idz] = np.sum(self.w2mag[idz]/self.errw2mag[idz]**2)/np.sum(1./self.errw2mag[idz]**2)
                self.meanw3[idz] = np.sum(self.w3mag[idz]/self.errw3mag[idz]**2)/np.sum(1./self.errw3mag[idz]**2)
                self.meanw4[idz] = np.sum(self.w4mag[idz]/self.errw4mag[idz]**2)/np.sum(1./self.errw4mag[idz]**2)

                self.medw1[idz] = np.median(self.w1mag[idz])
                self.medw2[idz] = np.median(self.w2mag[idz])
                self.medw3[idz] = np.median(self.w3mag[idz])
                self.medw4[idz] = np.median(self.w4mag[idz])

                self.low1[idz] = np.percentile(self.w1mag[idz], 14)
                self.hiw1[idz] = np.percentile(self.w1mag[idz], 86)
                self.low2[idz] = np.percentile(self.w2mag[idz], 14)
                self.hiw2[idz] = np.percentile(self.w2mag[idz], 86)
                self.low3[idz] = np.percentile(self.w3mag[idz], 14)
                self.hiw3[idz] = np.percentile(self.w3mag[idz], 86)
                self.low4[idz] = np.percentile(self.w4mag[idz], 14)
                self.hiw4[idz] = np.percentile(self.w4mag[idz], 86)

                self.errw1[idz] = [(self.medw1[idz]-self.low1[idz], self.hiw1[idz]-self.medw1[idz])]
                self.errw2[idz] = [(self.medw2[idz]-self.low2[idz], self.hiw2[idz]-self.medw2[idz])]
                self.errw3[idz] = [(self.medw3[idz]-self.low3[idz], self.hiw3[idz]-self.medw3[idz])]
                self.errw4[idz] = [(self.medw4[idz]-self.low4[idz], self.hiw4[idz]-self.medw4[idz])]

            self.w1mag['all'] = np.concatenate(self.w1mag.values())
            self.w2mag['all'] = np.concatenate(self.w2mag.values())
            self.w3mag['all'] = np.concatenate(self.w3mag.values())
            self.w4mag['all'] = np.concatenate(self.w4mag.values())

            self.errw1mag['all'] = np.concatenate(self.errw1mag.values())
            self.errw2mag['all'] = np.concatenate(self.errw2mag.values())
            self.errw3mag['all'] = np.concatenate(self.errw3mag.values())
            self.errw4mag['all'] = np.concatenate(self.errw4mag.values())

            self.meanw1['all'] = np.sum(self.w1mag['all']/self.errw1mag['all']**2)/np.sum(1./self.errw1mag['all']**2)
            self.meanw2['all'] = np.sum(self.w2mag['all']/self.errw2mag['all']**2)/np.sum(1./self.errw2mag['all']**2)
            self.meanw3['all'] = np.sum(self.w3mag['all']/self.errw3mag['all']**2)/np.sum(1./self.errw3mag['all']**2)
            self.meanw4['all'] = np.sum(self.w4mag['all']/self.errw4mag['all']**2)/np.sum(1./self.errw4mag['all']**2)

            self.medw1['all'] = np.median(self.w1mag['all'])
            self.medw2['all'] = np.median(self.w2mag['all'])
            self.medw3['all'] = np.median(self.w3mag['all'])
            self.medw4['all'] = np.median(self.w4mag['all'])

            self.low1['all'] = np.percentile(self.w1mag['all'], 14)
            self.low2['all'] = np.percentile(self.w2mag['all'], 14)
            self.low3['all'] = np.percentile(self.w3mag['all'], 14)
            self.low4['all'] = np.percentile(self.w4mag['all'], 14)

            self.hiw1['all'] = np.percentile(self.w1mag['all'], 86)
            self.hiw2['all'] = np.percentile(self.w2mag['all'], 86)
            self.hiw3['all'] = np.percentile(self.w3mag['all'], 86)
            self.hiw4['all'] = np.percentile(self.w4mag['all'], 86)

            self.errw1['all'] = [(self.medw1['all']-self.low1['all'], self.hiw1['all']-self.medw1['all'])]
            self.errw2['all'] = [(self.medw2['all']-self.low2['all'], self.hiw2['all']-self.medw2['all'])]
            self.errw3['all'] = [(self.medw3['all']-self.low3['all'], self.hiw3['all']-self.medw3['all'])]
            self.errw4['all'] = [(self.medw4['all']-self.low4['all'], self.hiw4['all']-self.medw4['all'])]

        elif survey == '2MASS' or survey == '2mass':
            
            self.jmag2MASS = {}
            self.hmag2MASS = {}
            self.kmag2MASS = {}

            self.errjmag2MASS = {}
            self.errhmag2MASS = {}
            self.errkmag2MASS = {}

            self.meanj2MASS = {}
            self.meanh2MASS = {}
            self.meank2MASS = {}

            self.medj2MASS = {}
            self.medh2MASS = {}
            self.medk2MASS = {}

            self.loj2MASS = {}
            self.loh2MASS = {}
            self.lok2MASS = {}
            self.hij2MASS = {}
            self.hih2MASS = {}
            self.hik2MASS = {}

            self.errj2MASS = {}
            self.errh2MASS = {}
            self.errk2MASS = {}

            for idz, zbin in enumerate(self.zbins):
                
                jflag = np.where(np.concatenate([self.extras[350][patch][idz]['JMAG'].copy() for patch in self.patches], axis=0)>0)
                hflag = np.where(np.concatenate([self.extras[350][patch][idz]['HMAG'].copy() for patch in self.patches], axis=0)>0)
                kflag = np.where(np.concatenate([self.extras[350][patch][idz]['KMAG'].copy() for patch in self.patches], axis=0)>0)
                
                self.jmag2MASS[idz] = TwoMASSMag2mJy(np.concatenate([self.extras[350][patch][idz]['JMAG'].copy() for patch in self.patches], axis=0), W='J')[jflag]
                self.hmag2MASS[idz] = TwoMASSMag2mJy(np.concatenate([self.extras[350][patch][idz]['HMAG'].copy() for patch in self.patches], axis=0), W='H')[hflag]
                self.kmag2MASS[idz] = TwoMASSMag2mJy(np.concatenate([self.extras[350][patch][idz]['KMAG'].copy() for patch in self.patches], axis=0), W='K_S')[kflag]

                self.errjmag2MASS[idz] = TwoMASSMag2mJy(np.concatenate([self.extras[350][patch][idz]['ERR_JMAG'].copy() for patch in self.patches], axis=0), W='J')[jflag]
                self.errhmag2MASS[idz] = TwoMASSMag2mJy(np.concatenate([self.extras[350][patch][idz]['ERR_HMAG'].copy() for patch in self.patches], axis=0), W='H')[hflag]
                self.errkmag2MASS[idz] = TwoMASSMag2mJy(np.concatenate([self.extras[350][patch][idz]['ERR_KMAG'].copy() for patch in self.patches], axis=0), W='K_S')[kflag]

                self.meanj2MASS[idz] = np.mean(self.jmag2MASS[idz])#np.sum(jmag[jflag]/errjmag[jflag]**2)/np.sum(1./errjmag[jflag]**2)
                self.meanh2MASS[idz] = np.mean(self.hmag2MASS[idz])#np.sum(hmag[hflag]/errhmag[hflag]**2)/np.sum(1./errhmag[hflag]**2)
                self.meank2MASS[idz] = np.mean(self.kmag2MASS[idz])#np.sum(kmag[kflag]/errkmag[kflag]**2)/np.sum(1./errkmag[kflag]**2)

                self.medj2MASS[idz] = np.median(self.jmag2MASS[idz])
                self.medk2MASS[idz] = np.median(self.kmag2MASS[idz])
                self.medh2MASS[idz] = np.median(self.hmag2MASS[idz])

                self.loj2MASS[idz] = np.percentile(self.jmag2MASS[idz], 14)
                self.lok2MASS[idz] = np.percentile(self.kmag2MASS[idz], 14)
                self.loh2MASS[idz] = np.percentile(self.hmag2MASS[idz], 14)

                self.hij2MASS[idz] = np.percentile(self.jmag2MASS[idz], 86)
                self.hik2MASS[idz] = np.percentile(self.kmag2MASS[idz], 86)
                self.hih2MASS[idz] = np.percentile(self.hmag2MASS[idz], 86)

                self.errj2MASS[idz] = [(self.medj2MASS[idz]-self.loj2MASS[idz], self.hij2MASS[idz]-self.medj2MASS[idz])]
                self.errh2MASS[idz] = [(self.medh2MASS[idz]-self.loh2MASS[idz], self.hih2MASS[idz]-self.medh2MASS[idz])]
                self.errk2MASS[idz] = [(self.medk2MASS[idz]-self.lok2MASS[idz], self.hik2MASS[idz]-self.medk2MASS[idz])]

            self.jmag2MASS['all'] = np.concatenate(self.jmag2MASS.values())
            self.hmag2MASS['all'] = np.concatenate(self.hmag2MASS.values())
            self.kmag2MASS['all'] = np.concatenate(self.kmag2MASS.values())

            self.errjmag2MASS['all'] = np.concatenate(self.errjmag2MASS.values())
            self.errhmag2MASS['all'] = np.concatenate(self.errhmag2MASS.values())
            self.errkmag2MASS['all'] = np.concatenate(self.errkmag2MASS.values())

            self.meanj2MASS['all'] = np.mean(self.jmag2MASS['all'])
            self.meanh2MASS['all'] = np.mean(self.hmag2MASS['all'])
            self.meank2MASS['all'] = np.mean(self.kmag2MASS['all'])

            self.medj2MASS['all'] = np.median(self.jmag2MASS['all'])
            self.medh2MASS['all'] = np.median(self.hmag2MASS['all'])
            self.medk2MASS['all'] = np.median(self.kmag2MASS['all'])

            self.loj2MASS['all'] = np.percentile(self.jmag2MASS['all'], 14)
            self.lok2MASS['all'] = np.percentile(self.kmag2MASS['all'], 14)
            self.loh2MASS['all'] = np.percentile(self.hmag2MASS['all'], 14)

            self.hij2MASS['all'] = np.percentile(self.jmag2MASS['all'], 86)
            self.hik2MASS['all'] = np.percentile(self.kmag2MASS['all'], 86)
            self.hih2MASS['all'] = np.percentile(self.hmag2MASS['all'], 86)

            self.errj2MASS['all'] = [(self.medj2MASS['all']-self.loj2MASS['all'], self.hij2MASS['all']-self.medj2MASS['all'])]
            self.errh2MASS['all'] = [(self.medh2MASS['all']-self.loh2MASS['all'], self.hih2MASS['all']-self.medh2MASS['all'])]
            self.errk2MASS['all'] = [(self.medk2MASS['all']-self.lok2MASS['all'], self.hik2MASS['all']-self.medk2MASS['all'])]

        elif survey == 'UKIDSS' or survey == 'ukidss':

            self.yfluxUKIDSS = {}
            self.jfluxUKIDSS = {}
            self.hfluxUKIDSS = {}
            self.kfluxUKIDSS = {}

            self.erryfluxUKIDSS = {}
            self.errjfluxUKIDSS = {}
            self.errhfluxUKIDSS = {}
            self.errkfluxUKIDSS = {}

            self.meanyUKIDSS = {}
            self.meanjUKIDSS = {}
            self.meanhUKIDSS = {}
            self.meankUKIDSS = {}

            self.medyUKIDSS = {}
            self.medjUKIDSS = {}
            self.medhUKIDSS = {}
            self.medkUKIDSS = {}

            self.loyUKIDSS = {}
            self.lojUKIDSS = {}
            self.lohUKIDSS = {}
            self.lokUKIDSS = {}

            self.hiyUKIDSS = {}
            self.hijUKIDSS = {}
            self.hihUKIDSS = {}
            self.hikUKIDSS = {}

            self.erryUKIDSS = {}
            self.errjUKIDSS = {}
            self.errhUKIDSS = {}
            self.errkUKIDSS = {}

            for idz, zbin in enumerate(self.zbins):

                flag = np.concatenate([self.extras[350][patch][idz]['UKIDSS_MATCHED'].copy() for patch in self.patches], axis=0)

                self.yfluxUKIDSS[idz] = np.concatenate([self.extras[350][patch][idz]['YFLUX'].copy() for patch in self.patches], axis=0)[flag==1] * WmHz2mJy
                self.jfluxUKIDSS[idz] = np.concatenate([self.extras[350][patch][idz]['JFLUX'].copy() for patch in self.patches], axis=0)[flag==1] * WmHz2mJy
                self.hfluxUKIDSS[idz] = np.concatenate([self.extras[350][patch][idz]['HFLUX'].copy() for patch in self.patches], axis=0)[flag==1] * WmHz2mJy
                self.kfluxUKIDSS[idz] = np.concatenate([self.extras[350][patch][idz]['KFLUX'].copy() for patch in self.patches], axis=0)[flag==1] * WmHz2mJy
                
                self.erryfluxUKIDSS[idz] = np.concatenate([self.extras[350][patch][idz]['YFLUX_ERR'].copy() for patch in self.patches], axis=0)[flag==1] * WmHz2mJy
                self.errjfluxUKIDSS[idz] = np.concatenate([self.extras[350][patch][idz]['JFLUX_ERR'].copy() for patch in self.patches], axis=0)[flag==1] * WmHz2mJy
                self.errhfluxUKIDSS[idz] = np.concatenate([self.extras[350][patch][idz]['HFLUX_ERR'].copy() for patch in self.patches], axis=0)[flag==1] * WmHz2mJy
                self.errkfluxUKIDSS[idz] = np.concatenate([self.extras[350][patch][idz]['KFLUX_ERR'].copy() for patch in self.patches], axis=0)[flag==1] * WmHz2mJy

                self.meanyUKIDSS[idz] = np.mean(self.yfluxUKIDSS[idz])#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.meanjUKIDSS[idz] = np.mean(self.jfluxUKIDSS[idz])#np.sum(jflux[flag==1]/errjflux[flag==1]**2)/np.sum(1./errjflux[flag==1]**2)
                self.meanhUKIDSS[idz] = np.mean(self.hfluxUKIDSS[idz])#np.sum(hflux[flag==1]/errhflux[flag==1]**2)/np.sum(1./errhflux[flag==1]**2)
                self.meankUKIDSS[idz] = np.mean(self.kfluxUKIDSS[idz])#np.sum(kflux[flag==1]/errkflux[flag==1]**2)/np.sum(1./errkflux[flag==1]**2)

                self.medyUKIDSS[idz] = np.median(self.yfluxUKIDSS[idz])#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.medjUKIDSS[idz] = np.median(self.jfluxUKIDSS[idz])#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.medhUKIDSS[idz] = np.median(self.hfluxUKIDSS[idz])#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.medkUKIDSS[idz] = np.median(self.kfluxUKIDSS[idz])#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)

                self.loyUKIDSS[idz] = np.percentile(self.yfluxUKIDSS[idz], 14)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.lojUKIDSS[idz] = np.percentile(self.jfluxUKIDSS[idz], 14)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.lohUKIDSS[idz] = np.percentile(self.hfluxUKIDSS[idz], 14)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.lokUKIDSS[idz] = np.percentile(self.kfluxUKIDSS[idz], 14)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)

                self.hiyUKIDSS[idz] = np.percentile(self.yfluxUKIDSS[idz], 86)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.hijUKIDSS[idz] = np.percentile(self.jfluxUKIDSS[idz], 86)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.hihUKIDSS[idz] = np.percentile(self.hfluxUKIDSS[idz], 86)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.hikUKIDSS[idz] = np.percentile(self.kfluxUKIDSS[idz], 86)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            
                self.erryUKIDSS[idz] = [(self.medyUKIDSS[idz]-self.loyUKIDSS[idz], self.hiyUKIDSS[idz]-self.medyUKIDSS[idz])]
                self.errjUKIDSS[idz] = [(self.medjUKIDSS[idz]-self.lojUKIDSS[idz], self.hijUKIDSS[idz]-self.medjUKIDSS[idz])]
                self.errhUKIDSS[idz] = [(self.medhUKIDSS[idz]-self.lohUKIDSS[idz], self.hihUKIDSS[idz]-self.medhUKIDSS[idz])]
                self.errkUKIDSS[idz] = [(self.medkUKIDSS[idz]-self.lokUKIDSS[idz], self.hikUKIDSS[idz]-self.medkUKIDSS[idz])]

            self.yfluxUKIDSS['all'] = np.concatenate(self.yfluxUKIDSS.values())
            self.jfluxUKIDSS['all'] = np.concatenate(self.jfluxUKIDSS.values())
            self.hfluxUKIDSS['all'] = np.concatenate(self.hfluxUKIDSS.values())
            self.kfluxUKIDSS['all'] = np.concatenate(self.kfluxUKIDSS.values())

            self.erryfluxUKIDSS['all'] = np.concatenate(self.erryfluxUKIDSS.values())
            self.errjfluxUKIDSS['all'] = np.concatenate(self.errjfluxUKIDSS.values())
            self.errhfluxUKIDSS['all'] = np.concatenate(self.errhfluxUKIDSS.values())
            self.errkfluxUKIDSS['all'] = np.concatenate(self.errkfluxUKIDSS.values())

            self.meanyUKIDSS['all'] = np.mean(self.yfluxUKIDSS['all'])
            self.meanjUKIDSS['all'] = np.mean(self.jfluxUKIDSS['all'])
            self.meanhUKIDSS['all'] = np.mean(self.hfluxUKIDSS['all'])
            self.meankUKIDSS['all'] = np.mean(self.kfluxUKIDSS['all'])

            self.medyUKIDSS['all'] = np.median(self.yfluxUKIDSS['all'])#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            self.medjUKIDSS['all'] = np.median(self.jfluxUKIDSS['all'])#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            self.medhUKIDSS['all'] = np.median(self.hfluxUKIDSS['all'])#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            self.medkUKIDSS['all'] = np.median(self.kfluxUKIDSS['all'])#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)

            self.loyUKIDSS['all'] = np.percentile(self.yfluxUKIDSS['all'], 14)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            self.lojUKIDSS['all'] = np.percentile(self.jfluxUKIDSS['all'], 14)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            self.lohUKIDSS['all'] = np.percentile(self.hfluxUKIDSS['all'], 14)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            self.lokUKIDSS['all'] = np.percentile(self.kfluxUKIDSS['all'], 14)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)

            self.hiyUKIDSS['all'] = np.percentile(self.yfluxUKIDSS['all'], 86)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            self.hijUKIDSS['all'] = np.percentile(self.jfluxUKIDSS['all'], 86)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            self.hihUKIDSS['all'] = np.percentile(self.hfluxUKIDSS['all'], 86)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
            self.hikUKIDSS['all'] = np.percentile(self.kfluxUKIDSS['all'], 86)#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)

            self.erryUKIDSS['all'] = [(self.medyUKIDSS['all']-self.loyUKIDSS['all'], self.hiyUKIDSS['all']-self.medyUKIDSS['all'])]
            self.errjUKIDSS['all'] = [(self.medjUKIDSS['all']-self.lojUKIDSS['all'], self.hijUKIDSS['all']-self.medjUKIDSS['all'])]
            self.errhUKIDSS['all'] = [(self.medhUKIDSS['all']-self.lohUKIDSS['all'], self.hihUKIDSS['all']-self.medhUKIDSS['all'])]
            self.errkUKIDSS['all'] = [(self.medkUKIDSS['all']-self.lokUKIDSS['all'], self.hikUKIDSS['all']-self.medkUKIDSS['all'])]

        elif survey == 'SDSS' or survey == 'sdss':

            self.uflux = {}
            self.gflux = {}
            self.rflux = {}
            self.iflux = {}
            self.zflux = {}

            self.erruflux = {}
            self.errgflux = {}
            self.errrflux = {}
            self.erriflux = {}
            self.errzflux = {}

            self.meanu = {}
            self.meang = {}
            self.meanr = {}
            self.meani = {}
            self.meanz = {}

            self.medu = {}
            self.medg = {}
            self.medr = {}
            self.medi = {}
            self.medz = {}

            self.lou = {}
            self.log = {}
            self.lor = {}
            self.loi = {}
            self.loz = {}

            self.hiu = {}
            self.hig = {}
            self.hir = {}
            self.hii = {}
            self.hiz = {}

            self.erru = {}
            self.errg = {}
            self.errr = {}
            self.erri = {}
            self.errz = {}

            for idz, zbin in enumerate(self.zbins):
                
                flag = np.concatenate([~np.isnan(self.extras[350][patch][idz]['PSFFLUX_Z']) for patch in self.patches], axis=0) 
                
                uflux = np.concatenate([self.extras[350][patch][idz]['PSFFLUX_U'].copy() for patch in self.patches], axis=0) 
                gflux = np.concatenate([self.extras[350][patch][idz]['PSFFLUX_G'].copy() for patch in self.patches], axis=0) 
                rflux = np.concatenate([self.extras[350][patch][idz]['PSFFLUX_R'].copy() for patch in self.patches], axis=0) 
                iflux = np.concatenate([self.extras[350][patch][idz]['PSFFLUX_I'].copy() for patch in self.patches], axis=0) 
                zflux = np.concatenate([self.extras[350][patch][idz]['PSFFLUX_Z'].copy() for patch in self.patches], axis=0) 
                
                uflux -= np.concatenate([self.extras[350][patch][idz]['EXTINCTION_U'].copy() for patch in self.patches], axis=0) 
                gflux -= np.concatenate([self.extras[350][patch][idz]['EXTINCTION_G'].copy() for patch in self.patches], axis=0) 
                rflux -= np.concatenate([self.extras[350][patch][idz]['EXTINCTION_R'].copy() for patch in self.patches], axis=0) 
                iflux -= np.concatenate([self.extras[350][patch][idz]['EXTINCTION_I'].copy() for patch in self.patches], axis=0) 
                zflux -= np.concatenate([self.extras[350][patch][idz]['EXTINCTION_Z'].copy() for patch in self.patches], axis=0)     

                self.uflux[idz] = SDSSMag2mJy(uflux[flag])                
                self.gflux[idz] = SDSSMag2mJy(gflux[flag])               
                self.rflux[idz] = SDSSMag2mJy(rflux[flag])               
                self.iflux[idz] = SDSSMag2mJy(iflux[flag])               
                self.zflux[idz] = SDSSMag2mJy(zflux[flag])               

                self.erruflux[idz] = SDSSMag2mJy(np.concatenate([self.extras[350][patch][idz]['IVAR_PSFFLUX_U'].copy() for patch in self.patches], axis=0))[flag]
                self.errgflux[idz] = SDSSMag2mJy(np.concatenate([self.extras[350][patch][idz]['IVAR_PSFFLUX_G'].copy() for patch in self.patches], axis=0))[flag]
                self.errrflux[idz] = SDSSMag2mJy(np.concatenate([self.extras[350][patch][idz]['IVAR_PSFFLUX_R'].copy() for patch in self.patches], axis=0))[flag]
                self.erriflux[idz] = SDSSMag2mJy(np.concatenate([self.extras[350][patch][idz]['IVAR_PSFFLUX_I'].copy() for patch in self.patches], axis=0))[flag]
                self.errzflux[idz] = SDSSMag2mJy(np.concatenate([self.extras[350][patch][idz]['IVAR_PSFFLUX_Z'].copy() for patch in self.patches], axis=0))[flag]

                self.meanu[idz] = np.mean((self.uflux[idz]))#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.meang[idz] = np.mean((self.gflux[idz]))#np.sum(jflux[flag==1]/errjflux[flag==1]**2)/np.sum(1./errjflux[flag==1]**2)
                self.meanr[idz] = np.mean((self.rflux[idz]))#np.sum(hflux[flag==1]/errhflux[flag==1]**2)/np.sum(1./errhflux[flag==1]**2)
                self.meani[idz] = np.mean((self.iflux[idz]))#np.sum(kflux[flag==1]/errkflux[flag==1]**2)/np.sum(1./errkflux[flag==1]**2)
                self.meanz[idz] = np.mean((self.zflux[idz]))#np.sum(kflux[flag==1]/errkflux[flag==1]**2)/np.sum(1./errkflux[flag==1]**2)

                self.medu[idz] = np.median((self.uflux[idz]))#np.sum(yflux[flag==1]/erryflux[flag==1]**2)/np.sum(1./erryflux[flag==1]**2)
                self.medg[idz] = np.median((self.gflux[idz]))#np.sum(jflux[flag==1]/errjflux[flag==1]**2)/np.sum(1./errjflux[flag==1]**2)
                self.medr[idz] = np.median((self.rflux[idz]))#np.sum(hflux[flag==1]/errhflux[flag==1]**2)/np.sum(1./errhflux[flag==1]**2)
                self.medi[idz] = np.median((self.iflux[idz]))#np.sum(kflux[flag==1]/errkflux[flag==1]**2)/np.sum(1./errkflux[flag==1]**2)
                self.medz[idz] = np.median((self.zflux[idz]))#np.sum(kflux[flag==1]/errkflux[flag==1]**2)/np.sum(1./errkflux[flag==1]**2)
                
                self.lou[idz] = np.percentile(self.uflux[idz], 14)
                self.log[idz] = np.percentile(self.gflux[idz], 14)
                self.lor[idz] = np.percentile(self.rflux[idz], 14)
                self.loi[idz] = np.percentile(self.iflux[idz], 14)
                self.loz[idz] = np.percentile(self.zflux[idz], 14)

                self.hiu[idz] = np.percentile(self.uflux[idz], 86)
                self.hig[idz] = np.percentile(self.gflux[idz], 86)
                self.hir[idz] = np.percentile(self.rflux[idz], 86)
                self.hii[idz] = np.percentile(self.iflux[idz], 86)
                self.hiz[idz] = np.percentile(self.zflux[idz], 86)

                self.erru[idz] = [(self.medu[idz]-self.lou[idz], self.hiu[idz]-self.medu[idz])]
                self.errg[idz] = [(self.medg[idz]-self.log[idz], self.hig[idz]-self.medg[idz])]
                self.errr[idz] = [(self.medr[idz]-self.lor[idz], self.hir[idz]-self.medr[idz])]
                self.erri[idz] = [(self.medi[idz]-self.loi[idz], self.hii[idz]-self.medi[idz])]
                self.errz[idz] = [(self.medz[idz]-self.loz[idz], self.hiz[idz]-self.medz[idz])]

            self.uflux['all'] = np.concatenate(self.uflux.values())
            self.gflux['all'] = np.concatenate(self.gflux.values())
            self.rflux['all'] = np.concatenate(self.rflux.values())
            self.iflux['all'] = np.concatenate(self.iflux.values())
            self.zflux['all'] = np.concatenate(self.zflux.values())

            self.meanu['all'] = np.mean(self.uflux['all'])
            self.meang['all'] = np.mean(self.gflux['all'])
            self.meanr['all'] = np.mean(self.rflux['all'])
            self.meani['all'] = np.mean(self.iflux['all'])
            self.meanz['all'] = np.mean(self.zflux['all'])

            self.medu['all'] = np.median(self.uflux['all'])
            self.medg['all'] = np.median(self.gflux['all'])
            self.medr['all'] = np.median(self.rflux['all'])
            self.medi['all'] = np.median(self.iflux['all'])
            self.medz['all'] = np.median(self.zflux['all'])
    
            self.lou['all'] = np.percentile(self.uflux['all'], 14)
            self.log['all'] = np.percentile(self.gflux['all'], 14)
            self.lor['all'] = np.percentile(self.rflux['all'], 14)
            self.loi['all'] = np.percentile(self.iflux['all'], 14)
            self.loz['all'] = np.percentile(self.zflux['all'], 14)

            self.hiu['all'] = np.percentile(self.uflux['all'], 86)
            self.hig['all'] = np.percentile(self.gflux['all'], 86)
            self.hir['all'] = np.percentile(self.rflux['all'], 86)
            self.hii['all'] = np.percentile(self.iflux['all'], 86)
            self.hiz['all'] = np.percentile(self.zflux['all'], 86)

            self.erru['all'] = [(self.medu['all']-self.lou['all'], self.hiu['all']-self.medu['all'])]
            self.errg['all'] = [(self.medg['all']-self.log['all'], self.hig['all']-self.medg['all'])]
            self.errr['all'] = [(self.medr['all']-self.lor['all'], self.hir['all']-self.medr['all'])]
            self.erri['all'] = [(self.medi['all']-self.loi['all'], self.hii['all']-self.medi['all'])]
            self.errz['all'] = [(self.medz['all']-self.loz['all'], self.hiz['all']-self.medz['all'])]

        elif survey == 'all':
            for sur in ['WISE', '2MASS', 'UKIDSS', 'SDSS']:
                self.GetFluxesCats(sur) 

    def PrintFluxesCats(self, survey):
        if survey == 'WISE' or survey == 'wise':
            print '~~~~~~~~~~ WISE ~~~~~~~~~~~~~~~'
            for idz, zbin in enumerate(self.zbins):
                print '........ ' + str(zbin[0]) + ' < z < ' + str(zbin[1]) + ' ........'
                print 'W1: %.4f^{+%.4f}_{-%.4f}' %(self.medw1[idz], self.hiw1[idz]-self.medw1[idz], self.medw1[idz]-self.low1[idz])
                print 'W2: %.4f^{+%.4f}_{-%.4f}' %(self.medw2[idz], self.hiw2[idz]-self.medw2[idz], self.medw2[idz]-self.low2[idz])
                print 'W3: %.4f^{+%.4f}_{-%.4f}' %(self.medw3[idz], self.hiw3[idz]-self.medw4[idz], self.medw3[idz]-self.low3[idz])
                print 'W4: %.4f^{+%.4f}_{-%.4f}' %(self.medw4[idz], self.hiw4[idz]-self.medw4[idz], self.medw4[idz]-self.low4[idz])
                print '- - - - - - - - - - - - - - - -'

        elif survey == '2MASS' or survey == '2mass':
            print '~~~~~~~~~ 2MASS ~~~~~~~~~~~~~~~'
            for idz, zbin in enumerate(self.zbins):
                print '........ ' + str(zbin[0]) + ' < z < ' + str(zbin[1]) + ' ........'
                print 'J: %.4f^{+%.4f}_{-%.4f}' %(self.medj2MASS[idz], self.hij2MASS[idz]-self.medj2MASS[idz], self.medj2MASS[idz]-self.loj2MASS[idz])
                print 'H: %.4f^{+%.4f}_{-%.4f}' %(self.medh2MASS[idz], self.hih2MASS[idz]-self.medh2MASS[idz], self.medh2MASS[idz]-self.loh2MASS[idz])
                print 'K: %.4f^{+%.4f}_{-%.4f}' %(self.medk2MASS[idz], self.hik2MASS[idz]-self.medk2MASS[idz], self.medk2MASS[idz]-self.lok2MASS[idz])
                print '- - - - - - - - - - - - - - - -'

        elif survey == 'UKIDSS' or survey == 'ukidss':
            print '~~~~~~~~~ UKIDSS ~~~~~~~~~~~~~'
            for idz, zbin in enumerate(self.zbins):
                print '........ ' + str(zbin[0]) + ' < z < ' + str(zbin[1]) + ' ........'
                print 'Y: %.4f^{+%.4f}_{-%.4f}' %(self.medyUKIDSS[idz], self.hiyUKIDSS[idz]-self.medyUKIDSS[idz], self.medyUKIDSS[idz]-self.loyUKIDSS[idz])
                print 'J: %.4f^{+%.4f}_{-%.4f}' %(self.medjUKIDSS[idz], self.hijUKIDSS[idz]-self.medjUKIDSS[idz], self.medjUKIDSS[idz]-self.lojUKIDSS[idz])
                print 'H: %.4f^{+%.4f}_{-%.4f}' %(self.medhUKIDSS[idz], self.hihUKIDSS[idz]-self.medhUKIDSS[idz], self.medhUKIDSS[idz]-self.lohUKIDSS[idz])
                print 'K: %.4f^{+%.4f}_{-%.4f}' %(self.medkUKIDSS[idz], self.hikUKIDSS[idz]-self.medkUKIDSS[idz], self.medkUKIDSS[idz]-self.lokUKIDSS[idz])
                print '- - - - - - - - - - - - - - - -'

        elif survey == 'SDSS' or survey == 'sdss':
            print '~~~~~~~~~ SDSS ~~~~~~~~~~~~~~~'
            for idz, zbin in enumerate(self.zbins):
                print '........ ' + str(zbin[0]) + ' < z < ' + str(zbin[1]) + ' ........'
                print 'u: %.4f^{+%.4f}_{-%.4f}' %(self.medu[idz], self.hiu[idz]-self.medu[idz], self.medu[idz]-self.lou[idz])
                print 'g: %.4f^{+%.4f}_{-%.4f}' %(self.medg[idz], self.hig[idz]-self.medg[idz], self.medg[idz]-self.log[idz])
                print 'r: %.4f^{+%.4f}_{-%.4f}' %(self.medr[idz], self.hir[idz]-self.medr[idz], self.medr[idz]-self.lor[idz])
                print 'i: %.4f^{+%.4f}_{-%.4f}' %(self.medi[idz], self.hii[idz]-self.medi[idz], self.medi[idz]-self.loi[idz])
                print 'z: %.4f^{+%.4f}_{-%.4f}' %(self.medz[idz], self.hiz[idz]-self.medz[idz], self.medz[idz]-self.loz[idz])
                print '- - - - - - - - - - - - - - - -'

        elif survey == 'all':
            for sur in ['WISE', '2MASS', 'UKIDSS', 'SDSS']:
                self.PrintFluxesCats(sur) 
                print ''

    def GetBootstrapErrs(self, lambda_, patch, zbin, r, r_in=None, r_out=None, remove_mean=True, remove_max=False, nsim=100, nboot=2.):
        simuls = self.cuts[lambda_][patch][zbin].copy()
            
        if remove_max > 0:
            for _ in xrange(remove_max):
                simuls = np.delete(simuls, np.argmax(np.mean(simuls, axis=(1,2))), axis=0)

        if remove_mean:
            simuls -= self.bkd[lambda_][patch][zbin].mean()

        flux = np.zeros(nsim)

        for i in xrange(nsim):
            cuts_id = np.random.choice(np.arange(simuls.shape[0]), size=simuls.shape[0]/nboot)
                
            if (r_in is None) or (r_out is None):
                apertures = CircularAperture(self.positions[lambda_], r=r)
                # if remove_mean:
                #     stacked_map = (simuls[cuts_id].mean(axis=0) - self.bkd[lambda_][patch][zbin].mean())
                # else:
                #     stacked_map = simuls[cuts_id].mean(axis=0)
                stacked_map = simuls[cuts_id].mean(axis=0)
                phot_table = aperture_photometry(stacked_map/(factor[lambda_])/1e-3, apertures)
            else:
                apertures = CircularAperture(self.positions[lambda_], r=r)
                annulus_apertures = CircularAnnulus(self.positions[lambda_], r_in=r_in, r_out=r_out)
                apers = [apertures, annulus_apertures]
                # if remove_mean:
                #     stacked_map = (simuls[cuts_id].mean(axis=0) - self.bkd[lambda_][patch][zbin].mean())
                # else:
                #     stacked_map = simuls[cuts_id].mean(axis=0)
                stacked_map = simuls[cuts_id].mean(axis=0)
                phot_table = aperture_photometry(stacked_map/(factor[lambda_])/1e-3, apers)  
                bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
                bkg_sum = bkg_mean * apertures.area()
                final_sum = phot_table['aperture_sum_0'] - bkg_sum
                phot_table['aperture_sum'] = final_sum

            flux[i] = phot_table.field('aperture_sum')[0]

        return np.std(flux)/np.sqrt(nboot)

    def GetTotBootstrapErrs(self, lambda_, zbin, r, r_in=None, r_out=None, remove_mean=True, remove_max=False, nsim=100, nboot=2.):            
        simuls = {}
        for patch in self.patches:
            simuls[patch] = self.cuts[lambda_][patch][zbin].copy()

        if remove_max > 0:
            for patch in self.patches:
                for _ in xrange(remove_max):
                    simuls[patch] = np.delete(simuls[patch], np.argmax(np.mean(simuls[patch], axis=(1,2))), axis=0)

        if remove_mean:
            for patch in self.patches:
                simuls[patch] -= self.bkd[lambda_][patch][zbin].mean()

        simuls = np.concatenate([simuls[patch] for patch in self.patches])

        ncuts = simuls.shape[0]
        # ncuts = np.sum([simuls[patch].shape[0] for patch in patches]) 

        flux = np.zeros(nsim)

        for i in xrange(nsim):
            cuts_id = np.random.choice(ncuts, size=ncuts/nboot)
                
            if (r_in is None) or (r_out is None):
                apertures = CircularAperture(self.positions[lambda_], r=r)
                stacked_map = simuls[cuts_id].mean(axis=0)
                phot_table = aperture_photometry(stacked_map/(factor[lambda_])/1e-3, apertures)
            else:
                apertures = CircularAperture(self.positions[lambda_], r=r)
                annulus_apertures = CircularAnnulus(self.positions[lambda_], r_in=r_in, r_out=r_out)
                apers = [apertures, annulus_apertures]
                stacked_map = simuls[cuts_id].mean(axis=0)
                phot_table = aperture_photometry(stacked_map/(factor[lambda_])/1e-3, apers)  
                bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
                bkg_sum = bkg_mean * apertures.area()
                final_sum = phot_table['aperture_sum_0'] - bkg_sum
                phot_table['aperture_sum'] = final_sum

            flux[i] = phot_table.field('aperture_sum')[0]

        return np.std(flux)/np.sqrt(nboot)

    def GetTotPhotometryFromStacks(self, lambda_, zbin, r, r_in=None, r_out=None, remove_mean=True, remove_max=False):
        simuls = {}
        for patch in self.patches:
            simuls[patch] = self.cuts[lambda_][patch][zbin].copy()
            
        if remove_max > 0:
            for patch in self.patches:
                for _ in xrange(remove_max):
                    simuls[patch] = np.delete(simuls[patch], np.argmax(np.mean(simuls[patch], axis=(1,2))), axis=0)

        if remove_mean:
            for patch in self.patches:
                simuls[patch] -= self.bkd[lambda_][patch][zbin].mean()

        if (r_in is None) or (r_out is None): 
            apertures = CircularAperture(self.positions[lambda_], r=r)
            # if remove_mean:
            #     stacked_map = np.concatenate([simuls[patch].copy() - self.bkd[lambda_][patch][zbin].mean() for patch in patches], axis=0).mean(0)
            # else:
            #     stacked_map = np.concatenate([simuls[patch].copy() for patch in patches], axis=0).mean(0)
            stacked_map = np.concatenate([simuls[patch].copy() for patch in self.patches], axis=0).mean(0)
            phot_table = aperture_photometry(stacked_map/(factor[lambda_])/1e-3, apertures)
        else:
            apertures = CircularAperture(self.positions[lambda_], r=r)
            annulus_apertures = CircularAnnulus(self.positions[lambda_], r_in=r_in, r_out=r_out)
            apers = [apertures, annulus_apertures]
            # if remove_mean:
            #     stacked_map = np.concatenate([simuls[patch].copy() - self.bkd[lambda_][patch][zbin].mean() for patch in patches], axis=0).mean(0)
            # else:
            #     stacked_map = np.concatenate([simuls[patch].copy() for patch in patches], axis=0).mean(0)
            stacked_map = np.concatenate([simuls[patch].copy() for patch in self.patches], axis=0).mean(0)
            phot_table = aperture_photometry(stacked_map/(factor[lambda_])/1e-3, apers)  
            bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
            bkg_sum = bkg_mean * apertures.area()
            final_sum = phot_table['aperture_sum_0'] - bkg_sum
            phot_table['aperture_sum'] = final_sum
        
        return phot_table.field('aperture_sum')[0]

    def GetPhotometryFromStacks(self, lambda_, patch, zbin, r, r_in=None, r_out=None, remove_mean=True, remove_max=False):
        simuls = self.cuts[lambda_][patch][zbin].copy()
            
        if remove_max > 0:
            for _ in xrange(remove_max):
                simuls = np.delete(simuls, np.argmax(np.mean(simuls, axis=(1,2))), axis=0)

        if remove_mean:
            simuls -= self.bkd[lambda_][patch][zbin].mean()

        if (r_in is None) or (r_out is None): 
            apertures = CircularAperture(self.positions[lambda_], r=r)
            stacked_map = simuls.mean(axis=0)
            phot_table = aperture_photometry(stacked_map/(factor[lambda_])/1e-3, apertures)
        else:
            apertures = CircularAperture(self.positions[lambda_], r=r)
            annulus_apertures = CircularAnnulus(self.positions[lambda_], r_in=r_in, r_out=r_out)
            apers = [apertures, annulus_apertures]
            stacked_map = simuls.mean(axis=0)
            phot_table = aperture_photometry(stacked_map/(factor[lambda_])/1e-3, apers)  
            bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
            bkg_sum = bkg_mean * apertures.area()
            final_sum = phot_table['aperture_sum_0'] - bkg_sum
            phot_table['aperture_sum'] = final_sum
        
        return phot_table.field('aperture_sum')[0]

    def FitMe(self, data, return_xy=False):
        y, x = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
        p_init = models.Gaussian2D(amplitude=np.max(data), x_mean=data.shape[0]/2, y_mean=data.shape[0]/2, x_stddev=1, y_stddev=1)
        fit_p = fitting.LevMarLSQFitter()
        if return_xy:
            return fit_p(p_init, x, y, data), x, y
        else:
            return fit_p(p_init, x, y, data)

    def GaussFit(self, lambda_, patch, zbin, remove_mean=True, remove_max=0, plot=False):
        simuls = self.cuts[lambda_][patch][zbin].copy()

        if remove_max > 0:
            for i in xrange(remove_max):
                simuls = np.delete(simuls, np.argmax(np.mean(simuls, axis=(1,2))), axis=0)

        if remove_mean:
            data = simuls.mean(axis=0) - self.bkd[lambda_][patch][zbin].mean()
        else:
            data = simuls.mean(axis=0)

        # Jy/beam -> mJy/beam
        data /= 1e-3
        
        p, x, y = self.FitMe(data, return_xy=True)

        if plot:
            plt.figure(figsize=(10,5))
            plt.suptitle(r'$\lambda =$'+ str(lambda_) + r' $\mu$m '+patch +' Fit and residuals')
            plt.subplot(121)
            im = plt.imshow(data,extent=[0,self.boxsize[lambda_]-1,0,self.boxsize[lambda_]-1], interpolation='bilinear')
            plt.colorbar(im,fraction=0.046, pad=0.04)
            plt.contour(p(y,x), 7)
            plt.plot(self.positions[lambda_][0],self.positions[lambda_][0], 'r+', mew=2.)
            # plt.axhline(self.positions[lambda_][1],color='w')
            # plt.axvline(self.positions[lambda_][0],color='w')
            plt.subplot(122)
            im = plt.imshow(data-p(y,x),extent=[0,self.boxsize[lambda_]-1,0,self.boxsize[lambda_]-1], interpolation='bilinear')
            plt.plot(self.positions[lambda_][0], self.positions[lambda_][0], 'r+', mew=2.)
            plt.colorbar(im,fraction=0.046, pad=0.04)

        print p.amplitude

        return p

    def GaussFitTot(self, lambda_, zbin, remove_mean=True, remove_max=0, plot=False):
        simuls = {}
        noises = {}
        for patch in self.patches:
            simuls[patch] = self.cuts[lambda_][patch][zbin].copy()
            try:
                noises[patch] = self.noise[lambda_][patch][zbin].copy()
            except:
                pass

        if remove_max > 0:
            for patch in self.patches:
                for _ in xrange(remove_max):
                    killme = np.argmax(np.mean(simuls[patch], axis=(1,2)))
                    simuls[patch] = np.delete(simuls[patch], killme, axis=0)
                    try:
                        noises[patch] = np.delete(noises[patch], killme, axis=0)
                    except:
                        pass

        if remove_mean:
            for patch in self.patches:
                simuls[patch] -= self.bkd[lambda_][patch][zbin].mean()
                # simuls[patch] = np.asarray([simuls[patch][i]-simuls[patch][i].mean(0) for i in xrange(simuls[patch].shape[0])])


        data = np.concatenate([simuls[patch].copy() for patch in self.patches], axis=0).mean(0)

        # Stuff for inverse variance weighting
        # data = np.concatenate([simuls[patch].copy()/noises[patch].copy()**2 for patch in self.patches], axis=0)#
        # nois = np.concatenate([1/noises[patch].copy()**2 for patch in self.patches], axis=0)#
        # data = data.mean(0)/nois.mean(0)

        # Stuff for variance weighting
        # data = np.concatenate([simuls[patch].copy()*noises[patch].copy() for patch in self.patches], axis=0)
        # nois = np.concatenate([noises[patch].copy() for patch in self.patches], axis=0)
        # data = data.mean(0)/nois.mean(0)

        # plt.imshow(data/data1-1., vmin=-10, vmax=10)
        # plt.colorbar()
        # plt.show()

        # plt.subplot(131)
        # plt.imshow(nois.mean(0), interpolation='bicubic'); plt.colorbar()
        # plt.subplot(132)
        # plt.imshow(data, interpolation='bicubic'); plt.colorbar()
        # plt.subplot(133)
        # plt.imshow(data1, interpolation='bicubic'); plt.colorbar()
        # plt.show()


        # Jy/beam -> mJy/beam
        data /= 1e-3

        # plt.imshow(data, interpolation='none', cmap='bone'); plt.colorbar()
        # plt.show()
        
        p, x, y = self.FitMe(data, return_xy=True)

        if plot:
            plt.figure(figsize=(10,5))
            plt.suptitle(r'$\lambda=$ '+str(lambda_) +', '+ str(self.zbins[zbin][0]) + r' $ < z < $ '+ str(self.zbins[zbin][1]) + ', Fit and residuals')
            plt.subplot(121)
            im = plt.imshow(data,extent=[0,self.boxsize[lambda_],0,self.boxsize[lambda_]], interpolation='none')
            plt.colorbar(im,fraction=0.046, pad=0.04)
            plt.contour(p(y,x), 7)
            plt.plot(self.positions[lambda_][0],self.positions[lambda_][0], 'r+', mew=2.)
            # plt.axhline(self.positions[lambda_][1],color='w')
            # plt.axvline(self.positions[lambda_][0],color='w')
            plt.subplot(122)
            im = plt.imshow(data-p(y,x),extent=[0,self.boxsize[lambda_],0,self.boxsize[lambda_]], interpolation='none')
            plt.plot(self.positions[lambda_][0], self.positions[lambda_][0], 'r+', mew=2.)
            plt.colorbar(im,fraction=0.046, pad=0.04)

        # print p.amplitude

        return p

    def GaussFitTotZ(self, lambda_, remove_mean=True, remove_max=0, plot=False):
        simuls = {}
        for idz, zbin in enumerate(self.zbins):
            simuls[idz] = {}
            for patch in self.patches:
                simuls[idz][patch] = self.cuts[lambda_][patch][idz].copy()
            
        if remove_max > 0:
            for idz, zbin in enumerate(self.zbins):
                for patch in self.patches:
                    for _ in xrange(remove_max):
                        simuls[idz][patch] = np.delete(simuls[idz][patch], np.argmax(np.mean(simuls[idz][patch], axis=(1,2))), axis=0)

        if remove_mean:
            for idz, zbin in enumerate(self.zbins):
                for patch in self.patches:
                    simuls[idz][patch] -= self.bkd[lambda_][patch][idz].mean()

        data_ = np.zeros((len(self.zbins), simuls[idz][patch].shape[1],simuls[idz][patch].shape[2]))
        for idz, zbin in enumerate(self.zbins):
            data_[idz] = np.concatenate([simuls[idz][patch].copy() for patch in self.patches], axis=0).mean(0)

        data = data_.mean(0)

        # print data

        # Jy/beam -> mJy/beam
        data /= 1e-3
        
        p, x, y = self.FitMe(data, return_xy=True)

        if plot:
            plt.figure(figsize=(10,5))
            plt.suptitle(r'$\lambda=$ '+str(lambda_) +', Fit and residuals')
            plt.subplot(121)
            im = plt.imshow(data,extent=[0,self.boxsize[lambda_],0,self.boxsize[lambda_]], interpolation='none')
            plt.colorbar(im,fraction=0.046, pad=0.04)
            plt.contour(p(y,x), 7)
            plt.plot(self.positions[lambda_][0],self.positions[lambda_][0], 'r+', mew=2.)
            # plt.axhline(self.positions[lambda_][1],color='w')
            # plt.axvline(self.positions[lambda_][0],color='w')
            plt.subplot(122)
            im = plt.imshow(data-p(y,x),extent=[0,self.boxsize[lambda_],0,self.boxsize[lambda_]], interpolation='none')
            plt.plot(self.positions[lambda_][0], self.positions[lambda_][0], 'r+', mew=2.)
            plt.colorbar(im,fraction=0.046, pad=0.04)

        # print p.amplitude

        return p
 
    def GetTotBootstrapErrsFit2D(self, lambda_, zbin, remove_mean=True, remove_max=False, nsim=100, nboot=2.):            
        simuls = {}
        for patch in self.patches:
            simuls[patch] = self.cuts[lambda_][patch][zbin].copy()

        if remove_max > 0:
            for patch in self.patches:
                for _ in xrange(remove_max):
                    simuls[patch] = np.delete(simuls[patch], np.argmax(np.mean(simuls[patch], axis=(1,2))), axis=0)

        if remove_mean:
            for patch in self.patches:
                simuls[patch] -= self.bkd[lambda_][patch][zbin].mean()

        simuls = np.concatenate([simuls[patch] for patch in self.patches])

        ncuts = simuls.shape[0]
        # ncuts = np.sum([simuls[patch].shape[0] for patch in patches]) 

        flux = np.zeros(nsim)

        for i in xrange(nsim):
            cuts_id = np.random.choice(ncuts, size=ncuts/nboot)
            stacked_map = simuls[cuts_id].mean(axis=0)/1e-3
            flux[i] = self.FitMe(stacked_map).amplitude.value

        return np.std(flux)/np.sqrt(nboot)#, flux

    def GetTotZBootstrapErrsFit2D(self, lambda_, remove_mean=True, remove_max=False, nsim=100, nboot=2.):            
        simuls = {}
        for idz, zbin in enumerate(self.zbins):
            simuls[idz] = {}
            for patch in self.patches:
                simuls[idz][patch] = self.cuts[lambda_][patch][idz].copy()
            
        if remove_max > 0:
            for idz, zbin in enumerate(self.zbins):
                for patch in self.patches:
                    for _ in xrange(remove_max):
                        simuls[idz][patch] = np.delete(simuls[idz][patch], np.argmax(np.mean(simuls[idz][patch], axis=(1,2))), axis=0)

        if remove_mean:
            for idz, zbin in enumerate(self.zbins):
                for patch in self.patches:
                    simuls[idz][patch] -= self.bkd[lambda_][patch][idz].mean()


        simuls = np.concatenate([simuls[idz][patch].copy() for patch in self.patches for idz in xrange(len(self.zbins))], axis=0)

        ncuts = simuls.shape[0]
        # ncuts = np.sum([simuls[patch].shape[0] for patch in patches]) 

        flux = np.zeros(nsim)

        for i in xrange(nsim):
            cuts_id = np.random.choice(ncuts, size=ncuts/nboot)
            stacked_map = simuls[cuts_id].mean(axis=0)/1e-3 # mJy/beam
            flux[i] = self.FitMe(stacked_map).amplitude.value

        return np.std(flux)/np.sqrt(nboot)#, flux

    def twoD_Gaussian(self, (x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                + c*((y-yo)**2)))
        return g.ravel()

    def FitGauss2D(self, lambda_, patch, zbin, remove_mean=False, remove_max=0, plot=True):
        simuls = self.cuts[lambda_][patch][zbin].copy()

        if remove_max > 0:
            for i in xrange(remove_max):
                simuls = np.delete(simuls, np.argmax(np.mean(simuls, axis=(1,2))), axis=0)

        if remove_mean:
            data = simuls.mean(axis=0) - self.bkd[lambda_][patch][zbin].mean()
        else:
            data = simuls.mean(axis=0)

        # Jy/beam -> mJy/beam
        data /= 1e-3

        guess = [np.max(data), data.shape[0]/2, data.shape[0]/2, 1., 1., 0., 0.]

        y, x = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))

        popt, pcov = optimize.curve_fit(self.twoD_Gaussian, (x, y), data.flatten(), p0=guess)

        if plot:
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            im = plt.imshow(data,extent=[0,39,0,39], interpolation='bilinear')
            plt.colorbar(im,fraction=0.046, pad=0.04)
            plt.contour(twoD_Gaussian((y,x), popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]).reshape((40,40)), 7)
            plt.plot(19.5,19.5, 'r+', mew=2.)
            # plt.axhline(self.positions[lambda_][1],color='w')
            # plt.axvline(self.positions[lambda_][0],color='w')
            plt.subplot(122)
            im = plt.imshow(data-twoD_Gaussian((y,x), popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]).reshape((40,40)),extent=[0,39,0,39], interpolation='bilinear')
            plt.plot(19.5, 19.5, 'r+', mew=2.)
            plt.colorbar(im,fraction=0.046, pad=0.04)
        print popt[0]


        return popt, pcov
