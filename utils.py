import numpy as np
import cPickle as pickle
import gzip
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

def SDSSMag2mJy(mags): # mags in nanomaggies
    return mags * 3.631e-3 

class CutoutAnalysis(object):
    def __init__(self, folder, lambdas=[250,350,500], patches=['G9','G12','G15'], zbins=[(1.0,5.0)], extras_names=['Z'], size=5.):
        if size == 5.:
            self.positions = {250: (25.5, 25.5), 350: (19.5,19.5), 500:(13.5,13.5)}
            self.boxsize = {250:51, 350:39, 500:27}


        self.lambdas = lambdas
        self.patches = patches
        self.zbins   = zbins

        self.bkd = {}
        self.cuts = {}
        self.fluxes_bkd = {}
        self.extras = {}

        # Loop over wavebands
        for lambda_ in lambdas:
            self.bkd[lambda_] = {}
            self.cuts[lambda_] = {}
            self.fluxes_bkd[lambda_] = {}
            self.extras[lambda_] = {}

            # Loop over patches
            for patch in self.patches:
                    self.bkd[lambda_][patch] = {}
                    self.cuts[lambda_][patch] = {}
                    self.fluxes_bkd[lambda_][patch] = {}
                    self.extras[lambda_][patch] = {}

                    # Loop over z-bins
                    for idz, (zmin, zmax) in enumerate(self.zbins):
                        cuts_ = pickle.load(gzip.open(folder + '/patch'+patch+'_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'.pkl','rb'))
                        self.cuts[lambda_][patch][idz] = np.asarray(cuts_['maps'])
                        bkd_ = pickle.load(gzip.open(folder + '/patch'+patch+'_lambda'+str(lambda_)+'_zmin'+str(zmin)+'_zmax'+str(zmax)+'_RND.pkl','rb'))
                        self.bkd[lambda_][patch][idz] = bkd_['maps']
                        self.fluxes_bkd[lambda_][patch][idz] = bkd_['fluxes']
                        self.extras[lambda_][patch][idz] = {}

                        # Other QSO specs?
                        for name in extras_names:
                            self.extras[lambda_][patch][idz][name] = np.asarray(cuts_[name])

    # def GetHist(self, xtr, lmbd=250):
    #     if (xtr == 'JMAG') or (xtr == 'HMAG') or (xtr == 'KMAG'):
    #         good_idx =
    #     hist(, bins="knuth", histtype='step', label='250')


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
        for patch in self.patches:
            simuls[patch] = self.cuts[lambda_][patch][zbin].copy()
            
        if remove_max > 0:
            for patch in self.patches:
                for _ in xrange(remove_max):
                    simuls[patch] = np.delete(simuls[patch], np.argmax(np.mean(simuls[patch], axis=(1,2))), axis=0)

        if remove_mean:
            for patch in self.patches:
                simuls[patch] -= self.bkd[lambda_][patch][zbin].mean()

        data = np.concatenate([simuls[patch].copy() for patch in self.patches], axis=0).mean(0)

        # Jy/beam -> mJy/beam
        data /= 1e-3
        
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

        return np.std(flux)/np.sqrt(nboot)


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

        # if plot:
        #     plt.figure(figsize=(10,5))
        #     plt.suptitle(r'$\lambda =$'+ str(lambda_) + r' $\mu$m '+patch +' Fit and residuals')
        #     plt.subplot(121)
        #     im = plt.imshow(data,extent=[0,self.boxsize[lambda_]-1,0,self.boxsize[lambda_]-1], interpolation='bilinear')
        #     plt.colorbar(im,fraction=0.046, pad=0.04)
        #     plt.contour(p(y,x), 7)
        #     plt.plot(self.positions[lambda_][0],self.positions[lambda_][0], 'r+', mew=2.)
        #     # plt.axhline(self.positions[lambda_][1],color='w')
        #     # plt.axvline(self.positions[lambda_][0],color='w')
        #     plt.subplot(122)
        #     im = plt.imshow(data-p(y,x),extent=[0,self.boxsize[lambda_]-1,0,self.boxsize[lambda_]-1], interpolation='bilinear')
        #     plt.plot(self.positions[lambda_][0], self.positions[lambda_][0], 'r+', mew=2.)
        #     plt.colorbar(im,fraction=0.046, pad=0.04)
        print popt[0]


        return popt, pcov

