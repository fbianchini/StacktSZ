import numpy as np
import cPickle as pickle
import gzip as gzip
import matplotlib.pyplot as plt

from astropy.visualization import hist
from astropy.modeling import models, fitting
from scipy import optimize

def plotme(data, random):
	plt.subplot(131)
	plt.title('Stack N=90k')
	im = plt.imshow(data)
	plt.colorbar(im,fraction=0.046, pad=0.04)

	plt.subplot(132)
	plt.title('Random')
	im = plt.imshow(random)
	plt.colorbar(im,fraction=0.046, pad=0.04)

	plt.subplot(133)
	plt.title('Stack - Random')
	im = plt.imshow(data-random)
	plt.colorbar(im,fraction=0.046, pad=0.04)



def FitGauss2D(data, remove_mean=False, remove_max=0, plot=True):
    guess = [np.max(data), data.shape[0]/2, data.shape[0]/2, 1., 1., 0., 0.]

    y, x = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))

    popt, pcov = optimize.curve_fit(twoD_Gaussian, (y, x), data.flatten(), p0=guess)


    # if plot:
    #     plt.figure(figsize=(10,5))
    #     plt.subplot(121)
    #     im = plt.imshow(data,extent=[0,data.shape[0]-1,0,data.shape[0]-1], interpolation='nearest')
    #     plt.colorbar(im,fraction=0.046, pad=0.04)
    #     plt.contour(twoD_Gaussian((y,x), popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]).reshape((21,21)), 7)
    #     # plt.plot(data.shape[1]/2,data.shape[0]/2, 'r+', mew=2.)
    #     # plt.axhline(self.positions[lambda_][1],color='w')
    #     # plt.axvline(self.positions[lambda_][0],color='w')
    #     plt.subplot(122)
    #     im = plt.imshow(data-twoD_Gaussian((y,x), popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]).reshape((21,21)),extent=[0,data.shape[0]-1,0,data.shape[0]-1], interpolation='nearest')
    #     # plt.plot(data.shape[1]/2,data.shape[0]/2, 'r+', mew=2.)
    #     plt.colorbar(im,fraction=0.046, pad=0.04)
    print popt[0]


    return popt, pcov

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()