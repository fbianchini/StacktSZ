#!/usr/bin/env

import cPickle as pickle
import numpy as np
import pylab as pl

freq=[30,44,70,100,143,217,353,545,857,'AKARI']
root='fluxes_ps_mask_from_mask_lfi_hfi_akari'
run_tag='lfihfiPSmaskAKARISourcesmask'
outfile='SDSS_QSO_stacked_fluxes_soergel_%s.dat'%run_tag

zbins_keys=['z1','z2','z3','z4','z5']
data_out=[]
for f in freq:
	print f
	line=[]
	if f=='AKARI': nu=3300
	else: nu=f
	data=pickle.load(open(root+'_'+str(f)+'.pkl','r'))['stack'] 
	fmean=[]
	fstd=[]
	for k in zbins_keys:
		fluxes=data[k]['flux']
		nsrc=len(fluxes)
		fmean.append(np.mean(fluxes))
		fstd.append(np.std(fluxes)/np.sqrt(nsrc))

		if k=='z5': print nu,fmean[-1],'+/-',fstd[-1],np.median(fluxes),np.percentile(fluxes,14),np.percentile(fluxes,86)
	data_out.append([nu*1.e09/3e08/1.e-06]+[nu]+fmean+fstd)

header='''
# SDSS DR-7&12 QSOs stacked fluxes in the Planck LFI and HFI
# as well as AKARI 90 micron following Soergel et al 2017 procedure
# for redshift bins
# z1: 2.15 < z < 2.5
# z2: 2.5 < z < 5
# z3: 2.5 < z < 5
# z4: 1 < z < 2.15
# z1: 0.1 < z < 5
# *ALL wavelengths are in micron*
# *ALL flux densities are in *mJy*
#  lambda[micron] frequency [GHz] mean_flux_z1 [mJy] mean_flux_z2 [mJy] mean_flux_z3 [mJy] mean_flux_z4 [mJy] mean_flux_z5 [mJy] 1sigma_err_z1 [mJy] 1sigma_err_z2 [mJy] 1sigma_err_z3 [mJy] 1sigma_err_z4 [mJy] 1sigma_err_z5 [mJy]
'''
np.savetxt(outfile,data_out,header=header,fmt='%.5e')

data=np.loadtxt(outfile,unpack=True)
nus=data[1]

pl.errorbar(nus,data[6],yerr=2*data[11],marker='o',label='0.1 < z < 5')
pl.errorbar(nus,data[5],yerr=2*data[10],marker='o',label='1 < z < 5',linestyle='')
pl.axhline(0,0,3400,linestyle='--',color='black')
pl.xscale('log')
pl.xlabel('Frequency [GHz]')
pl.ylabel('Flux [mJy]')
pl.legend(loc='best')
pl.grid()
pl.ylim(-2,16)
pl.savefig('one_bin_quasar_stack_%s.png'%run_tag)
#pl.show()

pl.clf()
pl.errorbar(nus,data[2],yerr=2*data[7],marker='o',label='1 < z < 2.15',linestyle='')
pl.errorbar(nus,data[3],yerr=2*data[8],marker='o',label='2.15 < z < 2.5',linestyle='')
pl.errorbar(nus,data[4],yerr=2*data[9],marker='o',label='2.5 < z < 5',linestyle='')
pl.axhline(0,0,3400,linestyle='--',color='black')
pl.xscale('log')
pl.xlabel('Frequency [GHz]')
pl.ylabel('Flux [mJy]')
pl.legend(loc='best')
pl.grid()
pl.ylim(-2,16)
pl.savefig('tomographic_bin_quasar_stack_%s.png'%run_tag)
#pl.show()

