import numpy as np
from scipy import integrate, interpolate, optimize
from read_cats import GetSDSSCat
import matplotlib.pyplot as plt
import matplotlib as mpl

from lifelines import KaplanMeierFitter

from utils import WISEMag2mJy, SDSSMag2mJy, TwoMASSMag2mJy, WmHz2mJy, GHz_to_lambda, lambda_to_GHz

from IPython import embed

class QSOcat:
    def __init__(self, cat, zbins, W4only=False, SN_W4=5, W4_over_r=0.):
        self.cat = {}
        self.zbins = zbins

        # Loop over z-bins
        for idz, (zmin, zmax) in enumerate(self.zbins):
            print("\t...z-bin : " + str(zmin) + " < z < " + str(zmax))
            self.cat[idz] = cat[(cat.Z >= zmin) & (cat.Z <= zmax)]

        # if W4only:
        #     for idz, (zmin, zmax) in enumerate(self.zbins):
        #         self.cat[idz] = self.cat[idz][self.cat[idz].W4SNR > SN_W4]

        if W4only:
            for idz, (zmin, zmax) in enumerate(self.zbins):
                self.cat[idz] = self.cat[idz][self.cat[idz].W4SNR > SN_W4]
                flag = self.cat[idz]['PSFFLUX_R'] > 0.
                ratio = WISEMag2mJy(self.cat[idz].W4MAG,'W4')/SDSSMag2mJy(self.cat[idz].PSFFLUX_R-self.cat[idz].EXTINCTION_R)
                # ratio = WISEMag2mJy(self.cat[idz].W4MAG[flag],'W4')/SDSSMag2mJy(self.cat[idz].PSFFLUX_R[flag]-self.cat[idz].EXTINCTION_R[flag])
                self.cat[idz] = self.cat[idz][(ratio.values > W4_over_r) & (ratio.values < np.inf) ]
                    
        self._fluxes_initialized = False

    def GetFluxesCats(self, survey):
        if survey == 'WISE' or survey == 'wise':

            self.w1good = {}
            self.w2good = {}
            self.w3good = {}
            self.w4good = {}

            self.detected_W1 = {}
            self.detected_W2 = {}
            self.detected_W3 = {}
            self.detected_W4 = {}
            
            self.w1mag = {}
            self.w2mag = {}
            self.w3mag = {}
            self.w4mag = {}

            self.meanw1 = {}
            self.meanw2 = {}
            self.meanw3 = {}
            self.meanw4 = {}

            self.meanw1_noKM = {}
            self.meanw2_noKM = {}
            self.meanw3_noKM = {}
            self.meanw4_noKM = {}

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

            self.low1_limits = {}
            self.hiw1_limits = {}
            self.low2_limits = {}
            self.hiw2_limits = {}
            self.low3_limits = {}
            self.hiw3_limits = {}
            self.low4_limits = {}
            self.hiw4_limits = {}

            self.errw1mag = {}
            self.errw2mag = {}
            self.errw3mag = {}
            self.errw4mag = {}

            self.errw1 = {}
            self.errw2 = {}
            self.errw3 = {}
            self.errw4 = {}

            self.errw1_mean = {}
            self.errw2_mean = {}
            self.errw3_mean = {}
            self.errw4_mean = {}

            self.errw1_limits = {}
            self.errw2_limits = {}
            self.errw3_limits = {}
            self.errw4_limits = {}

            for idz, zbin in enumerate(self.zbins):

                ccflag = self.cat[idz]['CC_FLAGS'].copy()

                # self.w1mag[idz] = WISEMag2mJy(self.cat[idz]['W1MAG'].copy()[ccflag=='0000'], W='W1')
                # self.w2mag[idz] = WISEMag2mJy(self.cat[idz]['W2MAG'].copy()[ccflag=='0000'], W='W2')
                # self.w3mag[idz] = WISEMag2mJy(self.cat[idz]['W3MAG'].copy()[ccflag=='0000'], W='W3')
                # self.w4mag[idz] = WISEMag2mJy(self.cat[idz]['W4MAG'].copy()[ccflag=='0000'], W='W4')

                self.w1good[idz] = self.cat[idz][(ccflag=='0000') & (~np.isnan(self.cat[idz]['W1MAG']))]
                self.w2good[idz] = self.cat[idz][(ccflag=='0000') & (~np.isnan(self.cat[idz]['W2MAG']))]
                self.w3good[idz] = self.cat[idz][(ccflag=='0000') & (~np.isnan(self.cat[idz]['W3MAG']))]
                self.w4good[idz] = self.cat[idz][(ccflag=='0000') & (~np.isnan(self.cat[idz]['W4MAG']))]

                self.detected_W1[idz] = self.w1good[idz].W1SNR >= 2
                self.detected_W2[idz] = self.w2good[idz].W2SNR >= 2
                self.detected_W3[idz] = self.w3good[idz].W3SNR >= 2
                self.detected_W4[idz] = self.w4good[idz].W4SNR >= 2

                self.w1mag[idz] = WISEMag2mJy(self.w1good[idz].W1MAG, 'W1')
                self.w2mag[idz] = WISEMag2mJy(self.w2good[idz].W2MAG, 'W2')
                self.w3mag[idz] = WISEMag2mJy(self.w3good[idz].W3MAG, 'W3')
                self.w4mag[idz] = WISEMag2mJy(self.w4good[idz].W4MAG, 'W4')

                self.errw1mag[idz] = WISEMag2mJy(self.cat[idz]['ERR_W1MAG'].copy()[ccflag=='0000'], W='W1')
                self.errw2mag[idz] = WISEMag2mJy(self.cat[idz]['ERR_W2MAG'].copy()[ccflag=='0000'], W='W2')
                self.errw3mag[idz] = WISEMag2mJy(self.cat[idz]['ERR_W3MAG'].copy()[ccflag=='0000'], W='W3')
                self.errw4mag[idz] = WISEMag2mJy(self.cat[idz]['ERR_W4MAG'].copy()[ccflag=='0000'], W='W4')

                self.meanw1_noKM[idz] = np.mean(self.w1mag[idz])#np.sum(self.w1mag[idz]/self.errw1mag[idz]**2)/np.sum(1./self.errw1mag[idz]**2)
                self.meanw2_noKM[idz] = np.mean(self.w2mag[idz])#np.sum(self.w2mag[idz]/self.errw2mag[idz]**2)/np.sum(1./self.errw2mag[idz]**2)
                self.meanw3_noKM[idz] = np.mean(self.w3mag[idz])#np.sum(self.w3mag[idz]/self.errw3mag[idz]**2)/np.sum(1./self.errw3mag[idz]**2)
                self.meanw4_noKM[idz] = np.mean(self.w4mag[idz])#np.sum(self.w4mag[idz]/self.errw4mag[idz]**2)/np.sum(1./self.errw4mag[idz]**2)

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

                self.low1_limits[idz] = np.sum(WISEMag2mJy(self.cat[idz]['W1MAG'].copy()[(ccflag=='0000') & (self.cat[idz]['W1SNR']>=2)], W='W1')) / len(self.cat[idz]['W1MAG'].copy()[(ccflag=='0000')])
                self.hiw1_limits[idz] = np.sum(WISEMag2mJy(self.cat[idz]['W1MAG'].copy()[ccflag=='0000'], W='W1')) / len(self.cat[idz]['W1MAG'].copy()[ccflag=='0000'])
                
                self.low2_limits[idz] = np.sum(WISEMag2mJy(self.cat[idz]['W2MAG'].copy()[(ccflag=='0000') & (self.cat[idz]['W2SNR']>=2)], W='W2')) / len(self.cat[idz]['W2MAG'].copy()[(ccflag=='0000')])
                self.hiw2_limits[idz] = np.sum(WISEMag2mJy(self.cat[idz]['W2MAG'].copy()[ccflag=='0000'], W='W2')) / len(self.cat[idz]['W2MAG'].copy()[ccflag=='0000'])
                
                self.low3_limits[idz] = np.sum(WISEMag2mJy(self.cat[idz]['W3MAG'].copy()[(ccflag=='0000') & (self.cat[idz]['W3SNR']>=2)], W='W3')) / len(self.cat[idz]['W3MAG'].copy()[(ccflag=='0000')])
                self.hiw3_limits[idz] = np.sum(WISEMag2mJy(self.cat[idz]['W3MAG'].copy()[ccflag=='0000'], W='W3')) / len(self.cat[idz]['W3MAG'].copy()[ccflag=='0000'])
                
                self.low4_limits[idz] = np.sum(WISEMag2mJy(self.cat[idz]['W4MAG'].copy()[(ccflag=='0000') & (self.cat[idz]['W4SNR']>=2)], W='W4')) / len(self.cat[idz]['W4MAG'].copy()[(ccflag=='0000')])
                self.hiw4_limits[idz] = np.sum(WISEMag2mJy(self.cat[idz]['W4MAG'].copy()[ccflag=='0000'], W='W4')) / len(self.cat[idz]['W4MAG'].copy()[ccflag=='0000'])

                # self.errw1[idz] = [(self.medw1[idz]-self.low1[idz], self.hiw1[idz]-self.medw1[idz])]
                # self.errw2[idz] = [(self.medw2[idz]-self.low2[idz], self.hiw2[idz]-self.medw2[idz])]
                # self.errw3[idz] = [(self.medw3[idz]-self.low3[idz], self.hiw3[idz]-self.medw3[idz])]
                # self.errw4[idz] = [(self.medw4[idz]-self.low4[idz], self.hiw4[idz]-self.medw4[idz])]

                kmf = KaplanMeierFitter()

                def BoostrapKM(df, channel, nsim=50, nboot=2):
                    flux = np.zeros(nsim)
                    for i in xrange(nsim):
                        ids = np.random.choice(np.arange(len(df)), size=len(df)/nboot)
                        df_ = df.iloc[ids]
                        detected = df_[channel+'SNR'] >= 2
                        fluxes = WISEMag2mJy(df_[channel+'MAG'], channel)
                        kmf.fit(fluxes, event_observed=detected, left_censorship=True)
                        cd = interpolate.interp1d(kmf.timeline, 1.-kmf.cumulative_density_.values[:,0], fill_value=0., bounds_error=0)
                        flux[i] = integrate.quad(cd, 0, np.inf)[0]
                    print channel, len(fluxes)
                    # return flux.mean(), flux.std()/np.sqrt(len(flux))#np.sqrt(len(fluxes))
                    return flux.std()/np.sqrt(len(flux))#np.sqrt(len(fluxes))

                kmf.fit(self.w1mag[idz], event_observed=self.detected_W1[idz], left_censorship=True)
                cd = interpolate.interp1d(kmf.timeline, 1.-kmf.cumulative_density_.values[:,0], fill_value=0., bounds_error=0)
                self.meanw1[idz] = integrate.quad(lambda x: cd(x)*x, 0, np.inf)[0]/integrate.quad(lambda x: cd(x), 0, np.inf)[0]

                kmf.fit(self.w2mag[idz], event_observed=self.detected_W2[idz], left_censorship=True)
                cd = interpolate.interp1d(kmf.timeline, 1.-kmf.cumulative_density_.values[:,0], fill_value=0., bounds_error=0)
                self.meanw2[idz] = integrate.quad(lambda x: cd(x)*x, 0, np.inf)[0]/integrate.quad(lambda x: cd(x), 0, np.inf)[0]

                kmf.fit(self.w3mag[idz], event_observed=self.detected_W3[idz], left_censorship=True)
                cd = interpolate.interp1d(kmf.timeline, 1.-kmf.cumulative_density_.values[:,0], fill_value=0., bounds_error=0)
                self.meanw3[idz] = integrate.quad(lambda x: cd(x)*x, 0, np.inf)[0]/integrate.quad(lambda x: cd(x), 0, np.inf)[0]

                kmf.fit(self.w4mag[idz], event_observed=self.detected_W4[idz], left_censorship=True)
                cd = interpolate.interp1d(kmf.timeline, 1.-kmf.cumulative_density_.values[:,0], fill_value=0., bounds_error=0)
                self.meanw4[idz] = integrate.quad(lambda x: cd(x)*x, 0, np.inf)[0]/integrate.quad(lambda x: cd(x), 0, np.inf)[0]

                self.errw1[idz] = [(self.meanw1[idz]-self.low1_limits[idz], self.hiw1_limits[idz]-self.meanw1[idz])]
                self.errw2[idz] = [(self.meanw2[idz]-self.low2_limits[idz], self.hiw2_limits[idz]-self.meanw2[idz])]
                self.errw3[idz] = [(self.meanw3[idz]-self.low3_limits[idz], self.hiw3_limits[idz]-self.meanw3[idz])]
                self.errw4[idz] = [(self.meanw4[idz]-self.low4_limits[idz], self.hiw4_limits[idz]-self.meanw4[idz])]


                self.errw1_mean[idz] = BoostrapKM(self.w1good[idz], 'W1')
                self.errw2_mean[idz] = BoostrapKM(self.w2good[idz], 'W2')
                self.errw3_mean[idz] = BoostrapKM(self.w3good[idz], 'W3')
                self.errw4_mean[idz] = BoostrapKM(self.w4good[idz], 'W4')

                # self.meanw1[idz], self.errw1_mean[idz] = BoostrapKM(self.w1good[idz], 'W1')
                # self.meanw2[idz], self.errw2_mean[idz] = BoostrapKM(self.w2good[idz], 'W2')
                # self.meanw3[idz], self.errw3_mean[idz] = BoostrapKM(self.w3good[idz], 'W3')
                # self.meanw4[idz], self.errw4_mean[idz] = BoostrapKM(self.w4good[idz], 'W4')

                # self.errw1_limits[idz] = [(self.medw1[idz]-self.low1_limits[idz], self.hiw1_limits[idz]-self.medw1[idz])]
                # self.errw2_limits[idz] = [(self.medw2[idz]-self.low2_limits[idz], self.hiw2_limits[idz]-self.medw2[idz])]
                # self.errw3_limits[idz] = [(self.medw3[idz]-self.low3_limits[idz], self.hiw3_limits[idz]-self.medw3[idz])]
                # self.errw4_limits[idz] = [(self.medw4[idz]-self.low4_limits[idz], self.hiw4_limits[idz]-self.medw4[idz])]

                # This is for the baseline analysis !!!!!
                self.errw1_limits[idz] = [(nsp.std(self.w1mag[idz]/np.sqrt(len(self.w1mag[idz]))), np.std(self.w1mag[idz]/np.sqrt(len(self.w1mag[idz]))) )]#[(self.meanw1_noKM[idz]-self.low1_limits[idz], self.hiw1_limits[idz]-self.meanw1[idz])]
                self.errw2_limits[idz] = [(np.std(self.w2mag[idz]/np.sqrt(len(self.w2mag[idz]))), np.std(self.w2mag[idz]/np.sqrt(len(self.w2mag[idz]))) )]#[(self.meanw2[idz]-self.low2_limits[idz], self.hiw2_limits[idz]-self.meanw2[idz])]
                self.errw3_limits[idz] = [(np.std(self.w3mag[idz]/np.sqrt(len(self.w3mag[idz]))), np.std(self.w3mag[idz]/np.sqrt(len(self.w3mag[idz]))) )]#[(self.meanw2[idz]-self.low2_limits[idz], self.hiw2_limits[idz]-self.meanw2[idz])]
                # self.errw3_limits[idz] = [(self.meanw3[idz]-self.low3_limits[idz], self.hiw3_limits[idz]-self.meanw3[idz])]
                # if self.errw3_limits[idz][1] <= 0.:
                #     self.errw3_limits[idz] = [(self.meanw3[idz]-self.low3_limits[idz], self.hiw3_limits[idz]-self.meanw3_noKM[idz])]
                self.errw4_limits[idz] = [(self.meanw4[idz]-self.low4_limits[idz], self.hiw4_limits[idz]-self.meanw4[idz])]

                # self.errw1_limits[idz] = [(self.meanw1_noKM[idz]-self.low1_limits[idz], self.hiw1_limits[idz]-self.meanw1_noKM[idz])]
                # self.errw2_limits[idz] = [(self.meanw2_noKM[idz]-self.low2_limits[idz], self.hiw2_limits[idz]-self.meanw2_noKM[idz])]
                # self.errw3_limits[idz] = [(self.meanw3_noKM[idz]-self.low3_limits[idz], self.hiw3_limits[idz]-self.meanw3_noKM[idz])]
                # self.errw4_limits[idz] = [(self.meanw4_noKM[idz]-self.low4_limits[idz], self.hiw4_limits[idz]-self.meanw4_noKM[idz])]

                # This is W4 selected !!!!!
                self.errw1_limits[idz] = [(np.std(self.w1mag[idz])/np.sqrt(len(self.w1mag[idz])), np.std(self.w1mag[idz])/np.sqrt(len(self.w1mag[idz])))] 
                self.errw2_limits[idz] = [(np.std(self.w2mag[idz])/np.sqrt(len(self.w2mag[idz])), np.std(self.w2mag[idz])/np.sqrt(len(self.w2mag[idz])))] 
                self.errw3_limits[idz] = [(np.std(self.w3mag[idz])/np.sqrt(len(self.w3mag[idz])), np.std(self.w3mag[idz])/np.sqrt(len(self.w3mag[idz])))] 
                self.errw4_limits[idz] = [(np.std(self.w4mag[idz])/np.sqrt(len(self.w4mag[idz])), np.std(self.w4mag[idz])/np.sqrt(len(self.w4mag[idz])))] 

            self.w1mag['all'] = np.concatenate(self.w1mag.values())
            self.w2mag['all'] = np.concatenate(self.w2mag.values())
            self.w3mag['all'] = np.concatenate(self.w3mag.values())
            self.w4mag['all'] = np.concatenate(self.w4mag.values())

            self.errw1mag['all'] = np.concatenate(self.errw1mag.values())
            self.errw2mag['all'] = np.concatenate(self.errw2mag.values())
            self.errw3mag['all'] = np.concatenate(self.errw3mag.values())
            self.errw4mag['all'] = np.concatenate(self.errw4mag.values())

            # self.meanw1['all'] = np.sum(self.w1mag['all']/self.errw1mag['all']**2)/np.sum(1./self.errw1mag['all']**2)
            # self.meanw2['all'] = np.sum(self.w2mag['all']/self.errw2mag['all']**2)/np.sum(1./self.errw2mag['all']**2)
            # self.meanw3['all'] = np.sum(self.w3mag['all']/self.errw3mag['all']**2)/np.sum(1./self.errw3mag['all']**2)
            # self.meanw4['all'] = np.sum(self.w4mag['all']/self.errw4mag['all']**2)/np.sum(1./self.errw4mag['all']**2)

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

            # self.errw1['all'] = [(self.medw1['all']-self.low1['all'], self.hiw1['all']-self.medw1['all'])]
            # self.errw2['all'] = [(self.medw2['all']-self.low2['all'], self.hiw2['all']-self.medw2['all'])]
            # self.errw3['all'] = [(self.medw3['all']-self.low3['all'], self.hiw3['all']-self.medw3['all'])]
            # self.errw4['all'] = [(self.medw4['all']-self.low4['all'], self.hiw4['all']-self.medw4['all'])]


        # elif survey == '2MASS' or survey == '2mass':
            
        #     self.jmag2MASS = {}
        #     self.hmag2MASS = {}
        #     self.kmag2MASS = {}

        #     self.errjmag2MASS = {}
        #     self.errhmag2MASS = {}
        #     self.errkmag2MASS = {}

        #     self.meanj2MASS = {}
        #     self.meanh2MASS = {}
        #     self.meank2MASS = {}

        #     self.medj2MASS = {}
        #     self.medh2MASS = {}
        #     self.medk2MASS = {}

        #     self.loj2MASS = {}
        #     self.loh2MASS = {}
        #     self.lok2MASS = {}
        #     self.hij2MASS = {}
        #     self.hih2MASS = {}
        #     self.hik2MASS = {}

        #     self.errj2MASS = {}
        #     self.errh2MASS = {}
        #     self.errk2MASS = {}

        #     for idz, zbin in enumerate(self.zbins):
                
        #         jflag = np.where(self.cat[idz]['JMAG'].copy()>0)
        #         hflag = np.where(self.cat[idz]['HMAG'].copy()>0)
        #         kflag = np.where(self.cat[idz]['KMAG'].copy()>0)
                
        #         self.jmag2MASS[idz] = TwoMASSMag2mJy(self.cat[idz]['JMAG'].copy(), W='J')[jflag]
        #         self.hmag2MASS[idz] = TwoMASSMag2mJy(self.cat[idz]['HMAG'].copy(), W='H')[hflag]
        #         self.kmag2MASS[idz] = TwoMASSMag2mJy(self.cat[idz]['KMAG'].copy(), W='K_S')[kflag]

        #         self.errjmag2MASS[idz] = TwoMASSMag2mJy(self.cat[idz]['ERR_JMAG'].copy(), W='J')[jflag]
        #         self.errhmag2MASS[idz] = TwoMASSMag2mJy(self.cat[idz]['ERR_HMAG'].copy(), W='H')[hflag]
        #         self.errkmag2MASS[idz] = TwoMASSMag2mJy(self.cat[idz]['ERR_KMAG'].copy(), W='K_S')[kflag]

        #         self.meanj2MASS[idz] = np.mean(self.jmag2MASS[idz])#np.sum(jmag[jflag]/errjmag[jflag]**2)/np.sum(1./errjmag[jflag]**2)
        #         self.meanh2MASS[idz] = np.mean(self.hmag2MASS[idz])#np.sum(hmag[hflag]/errhmag[hflag]**2)/np.sum(1./errhmag[hflag]**2)
        #         self.meank2MASS[idz] = np.mean(self.kmag2MASS[idz])#np.sum(kmag[kflag]/errkmag[kflag]**2)/np.sum(1./errkmag[kflag]**2)

        #         self.medj2MASS[idz] = np.median(self.jmag2MASS[idz])
        #         self.medk2MASS[idz] = np.median(self.kmag2MASS[idz])
        #         self.medh2MASS[idz] = np.median(self.hmag2MASS[idz])

        #         self.loj2MASS[idz] = np.percentile(self.jmag2MASS[idz], 14)
        #         self.lok2MASS[idz] = np.percentile(self.kmag2MASS[idz], 14)
        #         self.loh2MASS[idz] = np.percentile(self.hmag2MASS[idz], 14)

        #         self.hij2MASS[idz] = np.percentile(self.jmag2MASS[idz], 86)
        #         self.hik2MASS[idz] = np.percentile(self.kmag2MASS[idz], 86)
        #         self.hih2MASS[idz] = np.percentile(self.hmag2MASS[idz], 86)

        #         self.errj2MASS[idz] = [(self.medj2MASS[idz]-self.loj2MASS[idz], self.hij2MASS[idz]-self.medj2MASS[idz])]
        #         self.errh2MASS[idz] = [(self.medh2MASS[idz]-self.loh2MASS[idz], self.hih2MASS[idz]-self.medh2MASS[idz])]
        #         self.errk2MASS[idz] = [(self.medk2MASS[idz]-self.lok2MASS[idz], self.hik2MASS[idz]-self.medk2MASS[idz])]

        #     self.jmag2MASS['all'] = np.concatenate(self.jmag2MASS.values())
        #     self.hmag2MASS['all'] = np.concatenate(self.hmag2MASS.values())
        #     self.kmag2MASS['all'] = np.concatenate(self.kmag2MASS.values())

        #     self.errjmag2MASS['all'] = np.concatenate(self.errjmag2MASS.values())
        #     self.errhmag2MASS['all'] = np.concatenate(self.errhmag2MASS.values())
        #     self.errkmag2MASS['all'] = np.concatenate(self.errkmag2MASS.values())

        #     self.meanj2MASS['all'] = np.mean(self.jmag2MASS['all'])
        #     self.meanh2MASS['all'] = np.mean(self.hmag2MASS['all'])
        #     self.meank2MASS['all'] = np.mean(self.kmag2MASS['all'])

        #     self.medj2MASS['all'] = np.median(self.jmag2MASS['all'])
        #     self.medh2MASS['all'] = np.median(self.hmag2MASS['all'])
        #     self.medk2MASS['all'] = np.median(self.kmag2MASS['all'])

        #     self.loj2MASS['all'] = np.percentile(self.jmag2MASS['all'], 14)
        #     self.lok2MASS['all'] = np.percentile(self.kmag2MASS['all'], 14)
        #     self.loh2MASS['all'] = np.percentile(self.hmag2MASS['all'], 14)

        #     self.hij2MASS['all'] = np.percentile(self.jmag2MASS['all'], 86)
        #     self.hik2MASS['all'] = np.percentile(self.kmag2MASS['all'], 86)
        #     self.hih2MASS['all'] = np.percentile(self.hmag2MASS['all'], 86)

        #     self.errj2MASS['all'] = [(self.medj2MASS['all']-self.loj2MASS['all'], self.hij2MASS['all']-self.medj2MASS['all'])]
        #     self.errh2MASS['all'] = [(self.medh2MASS['all']-self.loh2MASS['all'], self.hih2MASS['all']-self.medh2MASS['all'])]
        #     self.errk2MASS['all'] = [(self.medk2MASS['all']-self.lok2MASS['all'], self.hik2MASS['all']-self.medk2MASS['all'])]

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

            self.erryUKIDSS_mean = {}
            self.errjUKIDSS_mean = {}
            self.errhUKIDSS_mean = {}
            self.errkUKIDSS_mean = {}

            for idz, zbin in enumerate(self.zbins):

                flag = self.cat[idz]['UKIDSS_MATCHED'].copy() == 1.

                self.yfluxUKIDSS[idz] = self.cat[idz]['YFLUX'][flag & (self.cat[idz]['YFLUX']>0)].copy() * WmHz2mJy
                self.jfluxUKIDSS[idz] = self.cat[idz]['JFLUX'][flag & (self.cat[idz]['JFLUX']>0)].copy() * WmHz2mJy
                self.hfluxUKIDSS[idz] = self.cat[idz]['HFLUX'][flag & (self.cat[idz]['HFLUX']>0)].copy() * WmHz2mJy
                self.kfluxUKIDSS[idz] = self.cat[idz]['KFLUX'][flag & (self.cat[idz]['KFLUX']>0)].copy() * WmHz2mJy
                
                self.erryfluxUKIDSS[idz] = self.cat[idz]['YFLUX_ERR'][flag & (self.cat[idz]['YFLUX']>0)].copy() * WmHz2mJy
                self.errjfluxUKIDSS[idz] = self.cat[idz]['JFLUX_ERR'][flag & (self.cat[idz]['JFLUX']>0)].copy() * WmHz2mJy
                self.errhfluxUKIDSS[idz] = self.cat[idz]['HFLUX_ERR'][flag & (self.cat[idz]['HFLUX']>0)].copy() * WmHz2mJy
                self.errkfluxUKIDSS[idz] = self.cat[idz]['KFLUX_ERR'][flag & (self.cat[idz]['KFLUX']>0)].copy() * WmHz2mJy

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

                self.erryUKIDSS_mean[idz] = np.std(self.yfluxUKIDSS[idz])/np.sqrt(len(self.yfluxUKIDSS[idz]))
                self.errjUKIDSS_mean[idz] = np.std(self.jfluxUKIDSS[idz])/np.sqrt(len(self.jfluxUKIDSS[idz]))
                self.errhUKIDSS_mean[idz] = np.std(self.hfluxUKIDSS[idz])/np.sqrt(len(self.hfluxUKIDSS[idz]))
                self.errkUKIDSS_mean[idz] = np.std(self.kfluxUKIDSS[idz])/np.sqrt(len(self.kfluxUKIDSS[idz]))

                print 'uk', len(self.yfluxUKIDSS[idz])
                print 'uk', len(self.jfluxUKIDSS[idz])
                print 'uk', len(self.hfluxUKIDSS[idz])
                print 'uk', len(self.kfluxUKIDSS[idz])

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

            self.erryUKIDSS_mean['all'] = np.std(self.yfluxUKIDSS['all'])/np.sqrt(len(self.yfluxUKIDSS['all']))
            self.errjUKIDSS_mean['all'] = np.std(self.jfluxUKIDSS['all'])/np.sqrt(len(self.jfluxUKIDSS['all']))
            self.errhUKIDSS_mean['all'] = np.std(self.hfluxUKIDSS['all'])/np.sqrt(len(self.hfluxUKIDSS['all']))
            self.errkUKIDSS_mean['all'] = np.std(self.kfluxUKIDSS['all'])/np.sqrt(len(self.kfluxUKIDSS['all']))

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

            self.erru_mean = {}
            self.errg_mean = {}
            self.errr_mean = {}
            self.erri_mean = {}
            self.errz_mean = {}

            for idz, zbin in enumerate(self.zbins):
                
                flag = ~np.isnan(self.cat[idz]['PSFFLUX_Z'])
                
                uflux = self.cat[idz]['PSFFLUX_U'].copy()
                gflux = self.cat[idz]['PSFFLUX_G'].copy() 
                rflux = self.cat[idz]['PSFFLUX_R'].copy()
                iflux = self.cat[idz]['PSFFLUX_I'].copy()
                zflux = self.cat[idz]['PSFFLUX_Z'].copy()
                
                uflux -= self.cat[idz]['EXTINCTION_U'].copy()
                gflux -= self.cat[idz]['EXTINCTION_G'].copy()
                rflux -= self.cat[idz]['EXTINCTION_R'].copy()
                iflux -= self.cat[idz]['EXTINCTION_I'].copy()
                zflux -= self.cat[idz]['EXTINCTION_Z'].copy()

                self.uflux[idz] = SDSSMag2mJy(uflux[flag])                
                self.gflux[idz] = SDSSMag2mJy(gflux[flag])               
                self.rflux[idz] = SDSSMag2mJy(rflux[flag])               
                self.iflux[idz] = SDSSMag2mJy(iflux[flag])               
                self.zflux[idz] = SDSSMag2mJy(zflux[flag])               

                self.erruflux[idz] = SDSSMag2mJy(self.cat[idz]['IVAR_PSFFLUX_U'].copy()[flag])
                self.errgflux[idz] = SDSSMag2mJy(self.cat[idz]['IVAR_PSFFLUX_G'].copy()[flag])
                self.errrflux[idz] = SDSSMag2mJy(self.cat[idz]['IVAR_PSFFLUX_R'].copy()[flag])
                self.erriflux[idz] = SDSSMag2mJy(self.cat[idz]['IVAR_PSFFLUX_I'].copy()[flag])
                self.errzflux[idz] = SDSSMag2mJy(self.cat[idz]['IVAR_PSFFLUX_Z'].copy()[flag])

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

                self.erru_mean[idz] = np.std(self.uflux[idz])/np.sqrt(len(self.uflux[idz]))
                self.errg_mean[idz] = np.std(self.gflux[idz])/np.sqrt(len(self.gflux[idz]))
                self.errr_mean[idz] = np.std(self.rflux[idz])/np.sqrt(len(self.rflux[idz]))
                self.erri_mean[idz] = np.std(self.iflux[idz])/np.sqrt(len(self.iflux[idz]))
                self.errz_mean[idz] = np.std(self.zflux[idz])/np.sqrt(len(self.zflux[idz]))

                print 'sdss u', len(self.uflux[idz])
                print 'sdss g', len(self.gflux[idz])
                print 'sdss r', len(self.rflux[idz])
                print 'sdss i', len(self.iflux[idz])
                print 'sdss z', len(self.zflux[idz])

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

            self.erru_mean['all'] = np.std(self.uflux['all'])/np.sqrt(len(self.uflux['all'])) 
            self.errg_mean['all'] = np.std(self.gflux['all'])/np.sqrt(len(self.gflux['all'])) 
            self.errr_mean['all'] = np.std(self.rflux['all'])/np.sqrt(len(self.rflux['all'])) 
            self.erri_mean['all'] = np.std(self.iflux['all'])/np.sqrt(len(self.iflux['all'])) 
            self.errz_mean['all'] = np.std(self.zflux['all'])/np.sqrt(len(self.zflux['all'])) 

        elif survey == 'all':
            for sur in ['WISE', '2MASS', 'UKIDSS', 'SDSS']:
                self.GetFluxesCats(sur) 

    def PrintFluxesCats(self, survey='all'):

        if self._fluxes_initialized is False:
            self.GetFluxesCats('all')
            self._fluxes_initialized = True

        if survey == 'WISE' or survey == 'wise':
            print '~~~~~~~~~~ WISE ~~~~~~~~~~~~~~~'
            for idz, zbin in enumerate(self.zbins):
                print '........ ' + str(zbin[0]) + ' < z < ' + str(zbin[1]) + ' ........'
                print 'W1: %.4f^{+%.4f}_{-%.4f}' %(self.medw1[idz], self.hiw1[idz]-self.medw1[idz], self.medw1[idz]-self.low1[idz])
                print 'W2: %.4f^{+%.4f}_{-%.4f}' %(self.medw2[idz], self.hiw2[idz]-self.medw2[idz], self.medw2[idz]-self.low2[idz])
                print 'W3: %.4f^{+%.4f}_{-%.4f}' %(self.medw3[idz], self.hiw3[idz]-self.medw4[idz], self.medw3[idz]-self.low3[idz])
                print 'W4: %.4f^{+%.4f}_{-%.4f}' %(self.medw4[idz], self.hiw4[idz]-self.medw4[idz], self.medw4[idz]-self.low4[idz])
                print '- - - - - - - - - - - - - - - -'

        # elif survey == '2MASS' or survey == '2mass':
        #     print '~~~~~~~~~ 2MASS ~~~~~~~~~~~~~~~'
        #     for idz, zbin in enumerate(self.zbins):
        #         print '........ ' + str(zbin[0]) + ' < z < ' + str(zbin[1]) + ' ........'
        #         print 'J: %.4f^{+%.4f}_{-%.4f}' %(self.medj2MASS[idz], self.hij2MASS[idz]-self.medj2MASS[idz], self.medj2MASS[idz]-self.loj2MASS[idz])
        #         print 'H: %.4f^{+%.4f}_{-%.4f}' %(self.medh2MASS[idz], self.hih2MASS[idz]-self.medh2MASS[idz], self.medh2MASS[idz]-self.loh2MASS[idz])
        #         print 'K: %.4f^{+%.4f}_{-%.4f}' %(self.medk2MASS[idz], self.hik2MASS[idz]-self.medk2MASS[idz], self.medk2MASS[idz]-self.lok2MASS[idz])
        #         print '- - - - - - - - - - - - - - - -'

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
            for sur in ['WISE', 'UKIDSS', 'SDSS']:
                self.PrintFluxesCats(sur) 
                print ''

    # def Plot(self, zbin, GHz=False):

    #     if self._fluxes_initialized is False:
    #         self.GetFluxesCats('all')
    #         self._fluxes_initialized = True

    #     plt.title(r' $%.2f < z < %.2f$' %(self.zbins[zbin][0],self.zbins[zbin][1]))

    #     # Hers
    #     # plt.errorbar(250, 3.28, yerr=0.23, color='tomato', fmt='x', label='SPIRE')
    #     # plt.errorbar(350, 3.46, yerr=0.23, color='tomato', fmt='x')
    #     # plt.errorbar(500, 2.96, yerr=0.24, color='tomato', fmt='x')
    #     if GHz:
    #         if zbin == 0:
    #             if len(self.zbins) == 1: # Soergel-style
    #                 plt.errorbar(lambda_to_GHz(250), 5.01, yerr=0.22, color='tomato', fmt='x', label='SPIRE')
    #                 plt.errorbar(lambda_to_GHz(350), 4.35, yerr=0.15, color='tomato', fmt='x')
    #                 plt.errorbar(lambda_to_GHz(500), 3.00, yerr=0.14, color='tomato', fmt='x')
    #             else:
    #                 plt.errorbar(lambda_to_GHz(250), 6.67, yerr=0.30, color='tomato', fmt='x', label='SPIRE')
    #                 plt.errorbar(lambda_to_GHz(350), 5.46, yerr=0.28, color='tomato', fmt='x')
    #                 plt.errorbar(lambda_to_GHz(500), 3.48, yerr=0.26, color='tomato', fmt='x')
    #         elif zbin == 1:
    #             plt.errorbar(lambda_to_GHz(250), 3.95, yerr=0.25, color='tomato', fmt='x', label='SPIRE')
    #             plt.errorbar(lambda_to_GHz(350), 3.99, yerr=0.27, color='tomato', fmt='x')
    #             plt.errorbar(lambda_to_GHz(500), 2.67, yerr=0.29, color='tomato', fmt='x')
    #         elif zbin == 2:
    #             plt.errorbar(lambda_to_GHz(250), 3.71, yerr=0.24, color='tomato', fmt='x', label='SPIRE')
    #             plt.errorbar(lambda_to_GHz(350), 3.89, yerr=0.26, color='tomato', fmt='x')
    #             plt.errorbar(lambda_to_GHz(500), 3.41, yerr=0.26, color='tomato', fmt='x')

    #         # Planck
    #         if (zbin == 0) and (len(self.zbins) == 1): # Soergel-style
    #             plt.errorbar((30), -1.36, yerr=0.19, color='brown', fmt='s', label='Planck')
    #             plt.errorbar((40), -1.87, yerr=0.25, color='brown', fmt='s')
    #             plt.errorbar((70), -0.56, yerr=0.17, color='brown', fmt='s')
    #             plt.errorbar((100), 0.13, yerr=0.10, color='brown', fmt='s')
    #             plt.errorbar((143), 0.25, yerr=0.06, color='brown', fmt='s')
    #             plt.errorbar((217), 0.84, yerr=0.05, color='brown', fmt='s')
    #             plt.errorbar((353), 3.93, yerr=0.10, color='brown', fmt='s')
    #             plt.errorbar((545), 9.24, yerr=0.17, color='brown', fmt='s')
    #             plt.errorbar((857), 11.0, yerr=0.24, color='brown', fmt='s')

    #         # AKARI
    #         if (zbin == 0) and (len(self.zbins) == 1): # Soergel-style
    #             plt.errorbar(lambda_to_GHz(90), 1.324, yerr=0.294, color='coral', fmt='8', label='AKARI')

    #         # 2MASS 
    #         plt.errorbar(lambda_to_GHz(1.235), self.medj2MASS[zbin], yerr=self.errj2MASS[zbin], color='green', fmt='o', label='2MASS')
    #         plt.errorbar(lambda_to_GHz(1.662), self.medh2MASS[zbin], yerr=self.errh2MASS[zbin], color='green', fmt='o')
    #         plt.errorbar(lambda_to_GHz(2.159), self.medk2MASS[zbin], yerr=self.errk2MASS[zbin], color='green', fmt='o')

    #         # WISE
    #         plt.errorbar(lambda_to_GHz(3.4), self.medw1[zbin], yerr=self.errw1[zbin], color='royalblue', fmt='^', label='WISE')
    #         plt.errorbar(lambda_to_GHz(4.6), self.medw2[zbin], yerr=self.errw2[zbin], color='royalblue', fmt='^')
    #         plt.errorbar(lambda_to_GHz(12), self.medw3[zbin], yerr=self.errw3[zbin], color='royalblue', fmt='^')
    #         plt.errorbar(lambda_to_GHz(22), self.medw4[zbin], yerr=self.errw4[zbin], color='royalblue', fmt='^')

    #         # UKIDSS
    #         plt.errorbar(lambda_to_GHz(1.02), self.medyUKIDSS[zbin], yerr=self.erryUKIDSS[zbin], color='orange', fmt='d', label='UKIDSS')
    #         plt.errorbar(lambda_to_GHz(1.25), self.medjUKIDSS[zbin], yerr=self.errjUKIDSS[zbin], color='orange', fmt='d')
    #         plt.errorbar(lambda_to_GHz(1.49), self.medhUKIDSS[zbin], yerr=self.errhUKIDSS[zbin], color='orange', fmt='d')
    #         plt.errorbar(lambda_to_GHz(2.03), self.medkUKIDSS[zbin], yerr=self.errkUKIDSS[zbin], color='orange', fmt='d')

    #         # SDSS
    #         plt.errorbar(lambda_to_GHz(0.35), self.medu[zbin], yerr=self.erru[zbin], color='purple', fmt='d', label='SDSS')
    #         plt.errorbar(lambda_to_GHz(0.48), self.medg[zbin], yerr=self.errg[zbin], color='purple', fmt='d')
    #         plt.errorbar(lambda_to_GHz(0.62), self.medr[zbin], yerr=self.errr[zbin], color='purple', fmt='d')
    #         plt.errorbar(lambda_to_GHz(0.76), self.medi[zbin], yerr=self.erri[zbin], color='purple', fmt='d')
    #         plt.errorbar(lambda_to_GHz(0.91), self.medz[zbin], yerr=self.errz[zbin], color='purple', fmt='d')

    #         #ACT
    #         # if zbin == 0:
    #         #     plt.errorbar(GHz_to_lambda(148), 0.05, yerr=0.04, color='brown', fmt='s', label='ACT')
    #         #     plt.errorbar(GHz_to_lambda(218), 0.51,  yerr=0.07, color='brown', fmt='s')
    #         #     plt.errorbar(GHz_to_lambda(277), 1.3,  yerr=0.16, color='brown', fmt='s')
    #         # elif zbin == 1:
    #         #     plt.errorbar(GHz_to_lambda(148), 0.11, yerr=0.04, color='brown', fmt='s', label='ACT')
    #         #     plt.errorbar(GHz_to_lambda(218), 0.57,  yerr=0.07, color='brown', fmt='s')
    #         #     plt.errorbar(GHz_to_lambda(277), 1.3,  yerr=0.17, color='brown', fmt='s')
    #         # elif zbin == 2:
    #         #     plt.errorbar(GHz_to_lambda(148), 0.12, yerr=0.05, color='brown', fmt='s', label='ACT')
    #         #     plt.errorbar(GHz_to_lambda(218), 0.59,  yerr=0.07, color='brown', fmt='s')
    #         #     plt.errorbar(GHz_to_lambda(277), 1.8,  yerr=0.17, color='brown', fmt='s')

    #         plt.xlabel(r'$\nu$ [GHz]')
    #         plt.ylabel(r'$S_{\nu}$ [mJy]')
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         plt.ylim([1e-3,1e1]) 
    #     else:
    #         if zbin == 0:
    #             if len(self.zbins) == 1: # Soergel-style
    #                 plt.errorbar(250, 5.01, yerr=0.22, color='tomato', fmt='x', label='SPIRE')
    #                 plt.errorbar(350, 4.35, yerr=0.15, color='tomato', fmt='x')
    #                 plt.errorbar(500, 3.00, yerr=0.14, color='tomato', fmt='x')
    #             else:
    #                 plt.errorbar(250, 6.67, yerr=0.30, color='tomato', fmt='x', label='SPIRE')
    #                 plt.errorbar(350, 5.46, yerr=0.28, color='tomato', fmt='x')
    #                 plt.errorbar(500, 3.48, yerr=0.26, color='tomato', fmt='x')
    #         elif zbin == 1:
    #             plt.errorbar(250, 3.95, yerr=0.25, color='tomato', fmt='x', label='SPIRE')
    #             plt.errorbar(350, 3.99, yerr=0.27, color='tomato', fmt='x')
    #             plt.errorbar(500, 2.67, yerr=0.29, color='tomato', fmt='x')
    #         elif zbin == 2:
    #             plt.errorbar(250, 3.71, yerr=0.24, color='tomato', fmt='x', label='SPIRE')
    #             plt.errorbar(350, 3.89, yerr=0.26, color='tomato', fmt='x')
    #             plt.errorbar(500, 3.41, yerr=0.26, color='tomato', fmt='x')

    #         # Planck
    #         if (zbin == 0) and (len(self.zbins) == 1): # Soergel-style
    #             plt.errorbar(GHz_to_lambda(30), -1.36, yerr=0.19, color='brown', fmt='s', label='Planck')
    #             plt.errorbar(GHz_to_lambda(40), -1.87, yerr=0.25, color='brown', fmt='s')
    #             plt.errorbar(GHz_to_lambda(70), -0.56, yerr=0.17, color='brown', fmt='s')
    #             plt.errorbar(GHz_to_lambda(100), 0.13, yerr=0.10, color='brown', fmt='s')
    #             plt.errorbar(GHz_to_lambda(143), 0.25, yerr=0.06, color='brown', fmt='s')
    #             plt.errorbar(GHz_to_lambda(217), 0.84, yerr=0.05, color='brown', fmt='s')
    #             plt.errorbar(GHz_to_lambda(353), 3.93, yerr=0.10, color='brown', fmt='s')
    #             plt.errorbar(GHz_to_lambda(545), 9.24, yerr=0.17, color='brown', fmt='s')
    #             plt.errorbar(GHz_to_lambda(857), 11.0, yerr=0.24, color='brown', fmt='s')

    #         # AKARI
    #         if (zbin == 0) and (len(self.zbins) == 1): # Soergel-style
    #             plt.errorbar(90, 1.324, yerr=0.294, color='coral', fmt='8', label='AKARI')

    #         # 2MASS 
    #         plt.errorbar((1.235), self.medj2MASS[zbin], yerr=self.errj2MASS[zbin], color='green', fmt='o', label='2MASS')
    #         plt.errorbar((1.662), self.medh2MASS[zbin], yerr=self.errh2MASS[zbin], color='green', fmt='o')
    #         plt.errorbar((2.159), self.medk2MASS[zbin], yerr=self.errk2MASS[zbin], color='green', fmt='o')

    #         # WISE
    #         plt.errorbar(3.4, self.medw1[zbin], yerr=self.errw1[zbin], color='royalblue', fmt='^', label='WISE')
    #         plt.errorbar(4.6, self.medw2[zbin], yerr=self.errw2[zbin], color='royalblue', fmt='^')
    #         plt.errorbar(12, self.medw3[zbin], yerr=self.errw3[zbin], color='royalblue', fmt='^')
    #         plt.errorbar(22, self.medw4[zbin], yerr=self.errw4[zbin], color='royalblue', fmt='^')

    #         # UKIDSS
    #         plt.errorbar(1.02, self.medyUKIDSS[zbin], yerr=self.erryUKIDSS[zbin], color='orange', fmt='d', label='UKIDSS')
    #         plt.errorbar(1.25, self.medjUKIDSS[zbin], yerr=self.errjUKIDSS[zbin], color='orange', fmt='d')
    #         plt.errorbar(1.49, self.medhUKIDSS[zbin], yerr=self.errhUKIDSS[zbin], color='orange', fmt='d')
    #         plt.errorbar(2.03, self.medkUKIDSS[zbin], yerr=self.errkUKIDSS[zbin], color='orange', fmt='d')

    #         # SDSS
    #         plt.errorbar(0.35, self.medu[zbin], yerr=self.erru[zbin], color='purple', fmt='d', label='SDSS')
    #         plt.errorbar(0.48, self.medg[zbin], yerr=self.errg[zbin], color='purple', fmt='d')
    #         plt.errorbar(0.62, self.medr[zbin], yerr=self.errr[zbin], color='purple', fmt='d')
    #         plt.errorbar(0.76, self.medi[zbin], yerr=self.erri[zbin], color='purple', fmt='d')
    #         plt.errorbar(0.91, self.medz[zbin], yerr=self.errz[zbin], color='purple', fmt='d')

    #         #ACT
    #         # if zbin == 0:
    #         #     plt.errorbar(GHz_to_lambda(148), 0.05, yerr=0.04, color='brown', fmt='s', label='ACT')
    #         #     plt.errorbar(GHz_to_lambda(218), 0.51,  yerr=0.07, color='brown', fmt='s')
    #         #     plt.errorbar(GHz_to_lambda(277), 1.3,  yerr=0.16, color='brown', fmt='s')
    #         # elif zbin == 1:
    #         #     plt.errorbar(GHz_to_lambda(148), 0.11, yerr=0.04, color='brown', fmt='s', label='ACT')
    #         #     plt.errorbar(GHz_to_lambda(218), 0.57,  yerr=0.07, color='brown', fmt='s')
    #         #     plt.errorbar(GHz_to_lambda(277), 1.3,  yerr=0.17, color='brown', fmt='s')
    #         # elif zbin == 2:
    #         #     plt.errorbar(GHz_to_lambda(148), 0.12, yerr=0.05, color='brown', fmt='s', label='ACT')
    #         #     plt.errorbar(GHz_to_lambda(218), 0.59,  yerr=0.07, color='brown', fmt='s')
    #         #     plt.errorbar(GHz_to_lambda(277), 1.8,  yerr=0.17, color='brown', fmt='s')

    #         plt.xlabel(r'$\lambda \, [\mu$m]')
    #         plt.ylabel(r'$S_{\nu}$ [mJy]')
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         plt.ylim([1e-3,1e1]) 

    def PlotFromAnal(self, zbin, herschel=None, planck=None, akari=None, GHz=False, method='mean', cmap=mpl.cm.RdBu):
        # akari and planck must be given as [lambda, flux, flux_err]

        n_exps = 3

        if akari is not None:
            n_exps += 1

        if planck is not None:
            n_exps += 1

        if herschel is not None:
            n_exps += 1

        if self._fluxes_initialized is False:
            self.GetFluxesCats('all')
            self._fluxes_initialized = True

        fig = plt.figure()
        
        ax1 = fig.add_subplot(111)

        ax1.set_title(r' $%.2f < z < %.2f$' %(self.zbins[zbin][0],self.zbins[zbin][1]), y=1.15)

        # AKARI
        if akari is not None:
            akari_lam  = akari[0,0]
            akari_flux = akari[0,1]
            akari_err  = akari[0,2]

        # Planck
        if planck is not None:
            planck_lam  = planck[:,0]
            planck_flux = planck[:,1]
            planck_err  = planck[:,2]

        # Herschel
        if herschel is not None:
            herschel_flux = {}
            herschel_err  = {}
            for lam in herschel.lambdas:
                if lam == 100:
                    herschel_flux[lam] = herschel.GetTotPhotometryFromStacks(lam,  zbin, 5.6/herschel.reso[lam], r_in=35/herschel.reso[lam], r_out=45/herschel.reso[lam],remove_max=1)
                    herschel_err[lam]  = herschel.GetTotBootstrapErrs(lam, zbin, 5.6/herschel.reso[lam], r_in=35/herschel.reso[lam], r_out=45/herschel.reso[lam], nsim=100,remove_max=1)
                elif lam == 160:
                    herschel_flux[lam] = herschel.GetTotPhotometryFromStacks(lam,  zbin, 10.2/herschel.reso[lam], r_in=35/herschel.reso[lam], r_out=45/herschel.reso[lam],remove_max=1)
                    herschel_err[lam]  = herschel.GetTotBootstrapErrs(lam, zbin, 10.2/herschel.reso[lam], r_in=35/herschel.reso[lam], r_out=45/herschel.reso[lam], nsim=100,remove_max=1)
                else: #SPIRE
                    herschel_flux[lam] = herschel.GaussFitTot(lam, zbin,remove_max=1).amplitude.value
                    herschel_err[lam]  = np.median([herschel.GetTotBootstrapErrsFit2D(lam, zbin, nsim=200,remove_max=1) for i in xrange(10)])

        if GHz:
            for lam in anal.lambdas:
                plt.errorbar(lambda_to_GHz(lam), anal_flux[lam], yerr=anal_err[lam], color='tomato', fmt='x')#, label='')
                print lam, anal_flux[lam], anal_err[lam]
            plt.errorbar(-100,-100,fmt='x',color='tomato', label='Herschel')

            # WISE
            if method == 'mean':
                plt.errorbar(lambda_to_GHz(3.4), self.meanw1[zbin], yerr=self.errw1_mean[zbin], color='royalblue', fmt='^', label='WISE')
                plt.errorbar(lambda_to_GHz(4.6), self.meanw2[zbin], yerr=self.errw2_mean[zbin], color='royalblue', fmt='^')
                plt.errorbar(lambda_to_GHz(12), self.meanw3[zbin], yerr=self.errw3_mean[zbin], color='royalblue', fmt='^')
                plt.errorbar(lambda_to_GHz(22), self.meanw4[zbin], yerr=self.errw4_mean[zbin], color='royalblue', fmt='^')
            elif method == 'median':
                plt.errorbar(lambda_to_GHz(3.4), self.medw1[zbin], yerr=self.errw1[zbin], color='royalblue', fmt='^', label='WISE')
                plt.errorbar(lambda_to_GHz(4.6), self.medw2[zbin], yerr=self.errw2[zbin], color='royalblue', fmt='^')
                plt.errorbar(lambda_to_GHz(12), self.medw3[zbin], yerr=self.errw3[zbin], color='royalblue', fmt='^')
                plt.errorbar(lambda_to_GHz(22), self.medw4[zbin], yerr=self.errw4[zbin], color='royalblue', fmt='^')

            # UKIDSS
            if method == 'mean':
                plt.errorbar(lambda_to_GHz(1.02), self.meanyUKIDSS[zbin], yerr=self.erryUKIDSS_mean[zbin], color='orange', fmt='d', label='UKIDSS')
                plt.errorbar(lambda_to_GHz(1.25), self.meanjUKIDSS[zbin], yerr=self.errjUKIDSS_mean[zbin], color='orange', fmt='d')
                plt.errorbar(lambda_to_GHz(1.49), self.meanhUKIDSS[zbin], yerr=self.errhUKIDSS_mean[zbin], color='orange', fmt='d')
                plt.errorbar(lambda_to_GHz(2.03), self.meankUKIDSS[zbin], yerr=self.errkUKIDSS_mean[zbin], color='orange', fmt='d')
            elif method == 'median':
                plt.errorbar(lambda_to_GHz(1.02), self.medyUKIDSS[zbin], yerr=self.erryUKIDSS[zbin], color='orange', fmt='d', label='UKIDSS')
                plt.errorbar(lambda_to_GHz(1.25), self.medjUKIDSS[zbin], yerr=self.errjUKIDSS[zbin], color='orange', fmt='d')
                plt.errorbar(lambda_to_GHz(1.49), self.medhUKIDSS[zbin], yerr=self.errhUKIDSS[zbin], color='orange', fmt='d')
                plt.errorbar(lambda_to_GHz(2.03), self.medkUKIDSS[zbin], yerr=self.errkUKIDSS[zbin], color='orange', fmt='d')

            # SDSS
            if method == 'mean':
                plt.errorbar(lambda_to_GHz(0.35), self.meanu[zbin], yerr=self.erru[zbin], color='purple', fmt='d', label='SDSS')
                plt.errorbar(lambda_to_GHz(0.48), self.meang[zbin], yerr=self.errg[zbin], color='purple', fmt='d')
                plt.errorbar(lambda_to_GHz(0.62), self.meanr[zbin], yerr=self.errr[zbin], color='purple', fmt='d')
                plt.errorbar(lambda_to_GHz(0.76), self.meani[zbin], yerr=self.erri[zbin], color='purple', fmt='d')
                plt.errorbar(lambda_to_GHz(0.91), self.meanz[zbin], yerr=self.errz[zbin], color='purple', fmt='d')
            elif method == 'median':
                plt.errorbar(lambda_to_GHz(0.35), self.medu[zbin], yerr=self.erru[zbin], color='purple', fmt='d', label='SDSS')
                plt.errorbar(lambda_to_GHz(0.48), self.medg[zbin], yerr=self.errg[zbin], color='purple', fmt='d')
                plt.errorbar(lambda_to_GHz(0.62), self.medr[zbin], yerr=self.errr[zbin], color='purple', fmt='d')
                plt.errorbar(lambda_to_GHz(0.76), self.medi[zbin], yerr=self.erri[zbin], color='purple', fmt='d')
                plt.errorbar(lambda_to_GHz(0.91), self.medz[zbin], yerr=self.errz[zbin], color='purple', fmt='d')

            plt.xlabel(r'$\nu$ [GHz]', size=13)
            plt.ylabel(r'$S_{\nu}$ [mJy]', size=13)
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.ylim([1e-3,1e1]) 
        else:
            if planck is not None:
                for i in xrange(len(planck_lam)):
                    if i == 0:
                        ax1.errorbar(planck_lam[i], planck_flux[i], yerr=planck_err[i], color='#E65552', fmt='h', label='Planck')
                    else:
                        ax1.errorbar(planck_lam[i], planck_flux[i], yerr=planck_err[i], color='#E65552', fmt='h')

            # Herschel
            if herschel is not None:
                for lam in herschel.lambdas:
                    ax1.errorbar((lam), herschel_flux[lam], yerr=herschel_err[lam], color='#EA9358', fmt='x')#, label='')
                    print lam, herschel_flux[lam], herschel_err[lam]
                ax1.errorbar(-100,-100,fmt='x',color='#EA9358', label='Herschel')

            if akari is not None:
                ax1.errorbar(akari_lam, akari_flux, yerr=akari_err, color='#EEDD82', fmt='s', label='AKARI')

            # WISE
            if method == 'mean':
                ax1.errorbar(3.4, self.meanw1_noKM[zbin], yerr=np.asarray(self.errw1_limits[zbin]).T, color='#69C9A3', fmt='^', label='WISE')
                ax1.errorbar(4.6, self.meanw2_noKM[zbin], yerr=np.asarray(self.errw2_limits[zbin]).T, color='#69C9A3', fmt='^')
                ax1.errorbar(12,  self.meanw3_noKM[zbin], yerr=np.asarray(self.errw3_limits[zbin]).T, color='#69C9A3', fmt='^')
                ax1.errorbar(22,  self.meanw4_noKM[zbin], yerr=np.asarray(self.errw4_limits[zbin]).T, color='#69C9A3', fmt='^')
                # ax1.errorbar(3.4, self.meanw1[zbin], yerr=self.errw1_mean[zbin], color='#69C9A3', fmt='^', label='WISE')
                # ax1.errorbar(4.6, self.meanw2[zbin], yerr=self.errw2_mean[zbin], color='#69C9A3', fmt='^')
                # ax1.errorbar(12,  self.meanw3[zbin], yerr=self.errw3_mean[zbin], color='#69C9A3', fmt='^')
                # ax1.errorbar(22,  self.meanw4[zbin], yerr=self.errw4_mean[zbin], color='#69C9A3', fmt='^')
            elif method == 'median':
                ax1.errorbar(3.4, self.medw1[zbin], yerr=self.errw1[zbin], color='royalblue', fmt='^', label='WISE')
                ax1.errorbar(4.6, self.medw2[zbin], yerr=self.errw2[zbin], color='royalblue', fmt='^')
                ax1.errorbar(12, self.medw3[zbin], yerr=self.errw3[zbin], color='royalblue', fmt='^')
                ax1.errorbar(22, self.medw4[zbin], yerr=self.errw4[zbin], color='royalblue', fmt='^')

            # UKIDSS
            if method == 'mean':
                ax1.errorbar(1.02, self.meanyUKIDSS[zbin], yerr=self.erryUKIDSS_mean[zbin], color='#589372', fmt='d', label='UKIDSS')
                ax1.errorbar(1.25, self.meanjUKIDSS[zbin], yerr=self.errjUKIDSS_mean[zbin], color='#589372', fmt='d')
                ax1.errorbar(1.49, self.meanhUKIDSS[zbin], yerr=self.errhUKIDSS_mean[zbin], color='#589372', fmt='d')
                ax1.errorbar(2.03, self.meankUKIDSS[zbin], yerr=self.errkUKIDSS_mean[zbin], color='#589372', fmt='d')
            elif method == 'median':
                ax1.errorbar(1.02, self.medyUKIDSS[zbin], yerr=self.erryUKIDSS[zbin], color='orange', fmt='d', label='UKIDSS')
                ax1.errorbar(1.25, self.medjUKIDSS[zbin], yerr=self.errjUKIDSS[zbin], color='orange', fmt='d')
                ax1.errorbar(1.49, self.medhUKIDSS[zbin], yerr=self.errhUKIDSS[zbin], color='orange', fmt='d')
                ax1.errorbar(2.03, self.medkUKIDSS[zbin], yerr=self.errkUKIDSS[zbin], color='orange', fmt='d')

            # SDSS
            if method == 'mean':
                ax1.errorbar(0.35, self.meanu[zbin], yerr=self.erru_mean[zbin], color='#19647E', fmt='v', label='SDSS')
                ax1.errorbar(0.48, self.meang[zbin], yerr=self.errg_mean[zbin], color='#19647E', fmt='v')
                ax1.errorbar(0.62, self.meanr[zbin], yerr=self.errr_mean[zbin], color='#19647E', fmt='v')
                ax1.errorbar(0.76, self.meani[zbin], yerr=self.erri_mean[zbin], color='#19647E', fmt='v')
                ax1.errorbar(0.91, self.meanz[zbin], yerr=self.errz_mean[zbin], color='#19647E', fmt='v')
            elif method == 'median':
                ax1.errorbar(0.35, self.medu[zbin], yerr=self.erru[zbin], color='purple', fmt='d', label='SDSS')
                ax1.errorbar(0.48, self.medg[zbin], yerr=self.errg[zbin], color='purple', fmt='d')
                ax1.errorbar(0.62, self.medr[zbin], yerr=self.errr[zbin], color='purple', fmt='d')
                ax1.errorbar(0.76, self.medi[zbin], yerr=self.erri[zbin], color='purple', fmt='d')
                ax1.errorbar(0.91, self.medz[zbin], yerr=self.errz[zbin], color='purple', fmt='d')

            #ACT
            # if zbin == 0:
            #     ax1.errorbar(GHz_to_lambda(148), 0.05, yerr=0.04, color='brown', fmt='s', label='ACT')
            #     ax1.errorbar(GHz_to_lambda(218), 0.51,  yerr=0.07, color='brown', fmt='s')
            #     ax1.errorbar(GHz_to_lambda(277), 1.3,  yerr=0.16, color='brown', fmt='s')
            # elif zbin == 1:
            #     ax1.errorbar(GHz_to_lambda(148), 0.11, yerr=0.04, color='brown', fmt='s', label='ACT')
            #     ax1.errorbar(GHz_to_lambda(218), 0.57,  yerr=0.07, color='brown', fmt='s')
            #     ax1.errorbar(GHz_to_lambda(277), 1.3,  yerr=0.17, color='brown', fmt='s')
            # elif zbin == 2:
            #     ax1.errorbar(GHz_to_lambda(148), 0.12, yerr=0.05, color='brown', fmt='s', label='ACT')
            #     ax1.errorbar(GHz_to_lambda(218), 0.59,  yerr=0.07, color='brown', fmt='s')
            #     ax1.errorbar(GHz_to_lambda(277), 1.8,  yerr=0.17, color='brown', fmt='s')

            ax1.set_xlabel(r'$\lambda \, [\mu$m]', size=13)
            ax1.set_ylabel(r'$S_{\nu}$ [mJy]', size=13)
            ax1.set_xscale('log')
            ax1.set_yscale('log', nonposy='clip')
            ax1.legend(loc='upper left', frameon=0)
            ax1.set_ylim([1e-5,3e1]) 
            ax1.set_xlim([1e-1,2e4]) 

            new_tick_locations = np.array([1, 100, 1000, 1e4])

            def tick_function(X):
                return ["%d" % z for z in lambda_to_GHz(X)]

            ax2 = ax1.twiny()

            ax2.set_xscale('log')
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(new_tick_locations)
            ax2.set_xticklabels(tick_function(new_tick_locations))
            ax2.set_xlabel(r"$\nu$ [GHz]")

    def WriteToFileFromAnal(self, zbin, herschel=None, planck=None, akari=None, filename='stacked_cats'):
        if self._fluxes_initialized is False:
            self.GetFluxesCats('all')
            self._fluxes_initialized = True

        # AKARI
        if akari is not None:
            akari_lam  = akari[0,0]
            akari_flux = akari[0,1]
            akari_err  = akari[0,2]

        # Planck
        if planck is not None:
            planck_lam  = planck[:,0]
            planck_flux = planck[:,1]
            planck_err  = planck[:,2]

        # Herschel
        if herschel is not None:
            herschel_flux = {}
            herschel_err  = {}
            for lam in herschel.lambdas:
                if lam == 100:
                    herschel_flux[lam] = herschel.GetTotPhotometryFromStacks(lam,  zbin, 5.6/herschel.reso[lam], r_in=35/herschel.reso[lam], r_out=45/herschel.reso[lam])
                    herschel_err[lam]  = herschel.GetTotBootstrapErrs(lam, zbin, 5.6/herschel.reso[lam], r_in=35/herschel.reso[lam], r_out=45/herschel.reso[lam], nsim=1000,)
                elif lam == 160:
                    herschel_flux[lam] = herschel.GetTotPhotometryFromStacks(lam,  zbin, 10.2/herschel.reso[lam], r_in=35/herschel.reso[lam], r_out=45/herschel.reso[lam])
                    herschel_err[lam]  = herschel.GetTotBootstrapErrs(lam, zbin, 10.2/herschel.reso[lam], r_in=35/herschel.reso[lam], r_out=45/herschel.reso[lam], nsim=1000,)
                else: #SPIRE
                    herschel_flux[lam] = herschel.GaussFitTot(lam, zbin).amplitude.value
                    herschel_err[lam]  = herschel.GetTotBootstrapErrsFit2D(lam, zbin, nsim=1000,)

                print lam, herschel_flux[lam]

        lambdas = np.asarray([22, 12, 4.6, 3.4, 2.03, 1.49, 1.25, 1.02, 0.91, 0.76, 0.62, 0.48, 0.35])

        fluxes_mean = np.asarray([self.meanw4_noKM[zbin], self.meanw3_noKM[zbin], self.meanw2_noKM[zbin], self.meanw1_noKM[zbin],\
                                  self.meankUKIDSS[zbin], self.meanhUKIDSS[zbin], self.meanjUKIDSS[zbin], self.meanyUKIDSS[zbin], \
                                  self.meanz[zbin], self.meani[zbin], self.meanr[zbin], self.meang[zbin], self.meanu[zbin]])

        err_low = np.asarray([self.errw4_limits[zbin][0][0], self.errw3_limits[zbin][0][0], self.errw2_limits[zbin][0][0], self.errw1_limits[zbin][0][0], \
                              self.errkUKIDSS_mean[zbin], self.errhUKIDSS_mean[zbin], self.errjUKIDSS_mean[zbin], self.erryUKIDSS_mean[zbin], \
                              self.errz_mean[zbin], self.erri_mean[zbin], self.errr_mean[zbin], self.errg_mean[zbin], self.erru_mean[zbin]])
        
        err_hi = np.asarray([self.errw4_limits[zbin][0][1], self.errw3_limits[zbin][0][1], self.errw2_limits[zbin][0][1], self.errw1_limits[zbin][0][1],\
                              self.errkUKIDSS_mean[zbin], self.errhUKIDSS_mean[zbin], self.errjUKIDSS_mean[zbin], self.erryUKIDSS_mean[zbin], \
                              self.errz_mean[zbin], self.erri_mean[zbin], self.errr_mean[zbin], self.errg_mean[zbin], self.erru_mean[zbin]])
        
        # err_low = np.asarray([self.errw4_mean[zbin]/2., self.errw3_mean[zbin]/2., self.errw2_mean[zbin]/2., self.errw1_mean[zbin]/2.,\
        #                       self.errkUKIDSS_mean[zbin]/2., self.errhUKIDSS_mean[zbin]/2., self.errjUKIDSS_mean[zbin]/2., self.erryUKIDSS_mean[zbin]/2., \
        #                       self.errz_mean[zbin]/2., self.erri_mean[zbin]/2., self.errr_mean[zbin]/2., self.errg_mean[zbin]/2., self.erru_mean[zbin]/2.])
        
        # err_hi = np.asarray([self.errw4_mean[zbin]/2., self.errw3_mean[zbin]/2., self.errw2_mean[zbin]/2., self.errw1_mean[zbin]/2.,\
        #                       self.errkUKIDSS_mean[zbin]/2., self.errhUKIDSS_mean[zbin]/2., self.errjUKIDSS_mean[zbin]/2., self.erryUKIDSS_mean[zbin]/2., \
        #                       self.errz_mean[zbin]/2., self.erri_mean[zbin]/2., self.errr_mean[zbin]/2., self.errg_mean[zbin]/2., self.erru_mean[zbin]/2.])
    
        # err_low = np.asarray([self.errw4[zbin][0][0], self.errw3[zbin][0][0], self.errw2[zbin][0][0], self.errw1[zbin][0][0],\
        #                       self.errkUKIDSS[zbin][0][0], self.errhUKIDSS[zbin][0][0], self.errjUKIDSS[zbin][0][0], self.erryUKIDSS[zbin][0][0], \
        #                       self.errz[zbin][0][0], self.erri[zbin][0][0], self.errr[zbin][0][0], self.errg[zbin][0][0], self.erru[zbin][0][0]])
        
        # err_hi = np.asarray([self.errw4[zbin][0][1], self.errw3[zbin][0][1], self.errw2[zbin][0][1], self.errw1[zbin][0][1],\
        #                      self.errkUKIDSS[zbin][0][1], self.errhUKIDSS[zbin][0][1], self.errjUKIDSS[zbin][0][1], self.erryUKIDSS[zbin][0][1], \
        #                      self.errz[zbin][0][1], self.erri[zbin][0][1], self.errr[zbin][0][1], self.errg[zbin][0][1], self.erru[zbin][0][1]])

        if herschel is not None:
            lambdas = np.concatenate([lambdas, herschel.lambdas])
            fluxes_mean = np.concatenate([fluxes_mean, np.asarray([herschel_flux[lam] for lam in herschel.lambdas])])
            err_low = np.concatenate([err_low, np.asarray([herschel_err[lam] for lam in herschel.lambdas])])
            err_hi = np.concatenate([err_hi, np.asarray([herschel_err[lam] for lam in herschel.lambdas])])

        if akari is not None: 
            lambdas = np.concatenate([lambdas, np.asarray([akari_lam])])
            fluxes_mean = np.concatenate([fluxes_mean, np.asarray([akari_flux])])
            err_low = np.concatenate([err_low, np.asarray([akari_err])])
            err_hi = np.concatenate([err_hi, np.asarray([akari_err])])

        if planck is not None: 
            lambdas = np.concatenate([lambdas, planck_lam])
            fluxes_mean = np.concatenate([fluxes_mean, planck_flux])
            err_low = np.concatenate([err_low, planck_err])
            err_hi = np.concatenate([err_hi, planck_err])

        np.savetxt(filename+'_zbin_'+str(self.zbins[zbin][0])+'_'+str(self.zbins[zbin][1])+'.dat', \
                   np.c_[lambdas, fluxes_mean, err_low, err_hi], fmt='%.4e %.4e %.4e %.4e' , \
                   header='SDSS DR-7&12 QSOs stacked fluxes in the Herschel-SPIRE bands (map-based stacking), \n\
                   in the WISE W1-4, UKIDSS Y/J/H/K, and SDSS u/g/r/i/z bands (catalogue-based stacking), \n\
                   in the AKARI and PLANCK bands (map-based stacking) \n\
                   *ALL wavelengths are in micron*\n\
                   *ALL flux densities are in mJy*\n\
                   lambda[micron]   flux [mJy]   1sigma_err_low    1sigma_err_high')

    # def WriteToFile(self, filename='stacked_cats'):
    #     if self._fluxes_initialized is False:
    #         self.GetFluxesCats('all')
    #         self._fluxes_initialized = True
  
    #     lambdas = np.asarray([500, 350, 250, 22, 12, 4.6, 3.4, 2.03, 1.49, 1.25, 1.02, 0.91, 0.76, 0.62, 0.48, 0.35])
    #     # lambdas = lambdas[::-1]

    #     for idz, zbin in enumerate(self.zbins):
    #         if idz == 0:
    #             fluxes_median = [3.48, 5.46, 6.67, \
    #             self.medw4[idz], self.medw3[idz], self.medw2[idz], self.medw1[idz],\
    #             self.medkUKIDSS[idz], self.medhUKIDSS[idz], self.medjUKIDSS[idz], self.medyUKIDSS[idz], \
    #             self.medz[idz], self.medi[idz], self.medr[idz], self.medg[idz], self.medu[idz]]
                
    #             fluxes_mean = [3.48, 5.46, 6.67, \
    #             self.meanw4[idz], self.meanw3[idz], self.meanw2[idz], self.meanw1[idz],\
    #             self.meankUKIDSS[idz], self.meanhUKIDSS[idz], self.meanjUKIDSS[idz], self.meanyUKIDSS[idz], \
    #             self.meanz[idz], self.meani[idz], self.meanr[idz], self.meang[idz], self.meanu[idz]]
            
    #             err_low = [0.13, 0.14, 0.15,\
    #             self.errw4[idz][0][0], self.errw3[idz][0][0], self.errw2[idz][0][0], self.errw1[idz][0][0],\
    #             self.errkUKIDSS[idz][0][0], self.errhUKIDSS[idz][0][0], self.errjUKIDSS[idz][0][0], self.erryUKIDSS[idz][0][0], \
    #             self.errz[idz][0][0], self.erri[idz][0][0], self.errr[idz][0][0], self.errg[idz][0][0], self.erru[idz][0][0]]
 
    #             err_hi = [0.13, 0.14, 0.15,\
    #             self.errw4[idz][0][1], self.errw3[idz][0][1], self.errw2[idz][0][1], self.errw1[idz][0][1],\
    #             self.errkUKIDSS[idz][0][1], self.errhUKIDSS[idz][0][1], self.errjUKIDSS[idz][0][1], self.erryUKIDSS[idz][0][1], \
    #             self.errz[idz][0][1], self.erri[idz][0][1], self.errr[idz][0][1], self.errg[idz][0][1], self.erru[idz][0][1]]

    #         elif idz == 1:
    #             fluxes_median = [2.67, 3.99, 3.95,\
    #             self.medw4[idz], self.medw3[idz], self.medw2[idz], self.medw1[idz],\
    #             self.medkUKIDSS[idz], self.medhUKIDSS[idz], self.medjUKIDSS[idz], self.medyUKIDSS[idz], \
    #             self.medz[idz], self.medi[idz], self.medr[idz], self.medg[idz], self.medu[idz]]
                
    #             fluxes_mean = [2.67, 3.99, 3.95,\
    #             self.meanw4[idz], self.meanw3[idz], self.meanw2[idz], self.meanw1[idz],\
    #             self.meankUKIDSS[idz], self.meanhUKIDSS[idz], self.meanjUKIDSS[idz], self.meanyUKIDSS[idz], \
    #             self.meanz[idz], self.meani[idz], self.meanr[idz], self.meang[idz], self.meanu[idz]]
            
    #             err_low = [0.145, 0.135, 0.125,\
    #             self.errw4[idz][0][0], self.errw3[idz][0][0], self.errw2[idz][0][0], self.errw1[idz][0][0],\
    #             self.errkUKIDSS[idz][0][0], self.errhUKIDSS[idz][0][0], self.errjUKIDSS[idz][0][0], self.erryUKIDSS[idz][0][0], \
    #             self.errz[idz][0][0], self.erri[idz][0][0], self.errr[idz][0][0], self.errg[idz][0][0], self.erru[idz][0][0]]
                
    #             err_hi = [0.145, 0.135, 0.125,\
    #             self.errw4[idz][0][1], self.errw3[idz][0][1], self.errw2[idz][0][1], self.errw1[idz][0][1],\
    #             self.errkUKIDSS[idz][0][1], self.errhUKIDSS[idz][0][1], self.errjUKIDSS[idz][0][1], self.erryUKIDSS[idz][0][1], \
    #             self.errz[idz][0][1], self.erri[idz][0][1], self.errr[idz][0][1], self.errg[idz][0][1], self.erru[idz][0][1]]

    #         elif idz == 2:
    #             fluxes_median = [3.41, 3.89, 3.71,\
    #             self.medw4[idz], self.medw3[idz], self.medw2[idz], self.medw1[idz],\
    #             self.medkUKIDSS[idz], self.medhUKIDSS[idz], self.medjUKIDSS[idz], self.medyUKIDSS[idz], \
    #             self.medz[idz], self.medi[idz], self.medr[idz], self.medg[idz], self.medu[idz]]
                
    #             fluxes_mean = [3.41, 3.89, 3.71,\
    #             self.meanw4[idz], self.meanw3[idz], self.meanw2[idz], self.meanw1[idz],\
    #             self.meankUKIDSS[idz], self.meanhUKIDSS[idz], self.meanjUKIDSS[idz], self.meanyUKIDSS[idz], \
    #             self.meanz[idz], self.meani[idz], self.meanr[idz], self.meang[idz], self.meanu[idz]]
    
    #             err_low = [0.13, 0.13, 0.12,\
    #             self.errw4[idz][0][0], self.errw3[idz][0][0], self.errw2[idz][0][0], self.errw1[idz][0][0],\
    #             self.errkUKIDSS[idz][0][0], self.errhUKIDSS[idz][0][0], self.errjUKIDSS[idz][0][0], self.erryUKIDSS[idz][0][0], \
    #             self.errz[idz][0][0], self.erri[idz][0][0], self.errr[idz][0][0], self.errg[idz][0][0], self.erru[idz][0][0]]
                
    #             err_hi = [0.13, 0.13, 0.12,\
    #             self.errw4[idz][0][1], self.errw3[idz][0][1], self.errw2[idz][0][1], self.errw1[idz][0][1],\
    #             self.errkUKIDSS[idz][0][1], self.errhUKIDSS[idz][0][1], self.errjUKIDSS[idz][0][1], self.erryUKIDSS[idz][0][1], \
    #             self.errz[idz][0][1], self.erri[idz][0][1], self.errr[idz][0][1], self.errg[idz][0][1], self.erru[idz][0][1]]


    #         print fluxes_median
    #         np.savetxt(filename+'_zbin_'+str(self.zbins[idz][0])+'_'+str(self.zbins[idz][1])+'.dat', \
    #                    np.c_[lambdas, fluxes_median, fluxes_mean, err_low, err_hi], \
    #                    header='SDSS DR-7&12 QSOs stacked fluxes in the Herschel-SPIRE bands (map-based stacking) and \n\
    #                    in the WISE W1-4, UKIDSS Y/J/H/K, and SDSS u/g/r/i/z bands (catalogue-based stacking)\n\
    #                    *ALL wavelengths are in micron*\n\
    #                    *ALL flux densities are in mJy*\n\
    #                    lambda[micron]   median_flux [mJy]   mean_flux [mJy]   1sigma_err_low    1sigma_err_high')

if __name__ == '__main__':

    # Redshift bins
    zbins = [(0.,10.0)]
    # zbins = [(0.1,5.0)]
    # zbins = [(1.,2.15), (2.15,2.50),(2.50,5.0)]

    # Reading in QSO catalogs
    qso_cat = QSOcat(GetSDSSCat(cats=['DR7', 'DR12'], discard_FIRST=True, z_DR12='Z_PIPE'), zbins)

    embed()

