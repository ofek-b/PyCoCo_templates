#!/usr/bin/env python
# coding: utf-8

# # Two-Dim Gaussian Interpolation
# ### Sub-steps:
# **1)** *Build the grid time x wavelength using flux calibrated spectra, UV photometry and optical photometry (only where spectrophotometry is sparse)*
# 
# **2)** *Model the data using 2dim GPs and determine the 2dim interpolated surface*
# 
# **3)** *Predict UV extension for your sparse spectrophotometry*
# 
# **4)** *Predict additional spectra at the phases where spectrophotometry is sparse (FROM sparse spectrophotometry ---> TO spectrophotometry with high and homogeneous cadence)*

# In[1]:


import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import interpolate

import Codes_Scripts.GP2dim_utils as GP2dim
import Codes_Scripts.excludefilt
from Codes_Scripts.config import *


def main(SN_NAME):
    exclude_filt, Bessell_V_like, swift_V_like, Bessell_B_like, swift_B_like = Codes_Scripts.excludefilt.main(SN_NAME)

    # In[3]:

    CSP_SNe = []  # OFEK
    # ['SN2004fe', 'SN2005bf', 'SN2006V', 'SN2007C', 'SN2007Y',
    #            'SN2009bb', 'SN2008aq', 'SN2006T', 'SN2004gq', 'SN2004gt',
    #            'SN2004gv', 'SN2006ep', 'SN2008fq', 'SN2006aa']

    pre_bump = ['SN2011dh', 'SN1993J', 'SN2008D', 'SN2011fu', 'SN2006aj', 'SN1987A', 'SN2013df']

    exclude_filt = ['H', 'J', 'K', 'Ks', 'KS', 'Y']

    convert2mjd = (lambda x: float(
        x.replace('_REmangled_spec.txt', '').replace('_REmangled_spec_FL.txt', '').replace('_mangled_spec.txt', '')))

    # In[5]:

    def calc_lam_eff(wls, transmission):
        return (integrate.trapz(transmission * wls, wls) / integrate.trapz(transmission, wls))

    # In[6]:

    class FullMangledSeries_Class():
        """Class to load and mangle a single spectrum:
        """

        def __init__(self, snname, type_=None, spec_file=None,
                     mode='extend_spectra', verbose=False, DELTA=50.):
            """
            mode: 'extend_spectra'  or 'extrapolate_spectra'
            """
            ## Initialise the class variables
            self.snname = snname
            self.SNtype = type_
            self.mode = mode
            self.get_mangledspec_list()
            self.DELTA = DELTA
            self.verbose = verbose
            self.create_extended_spec_folder()
            self.path_fit_phot = join(OUTPUT_DIR, snname, 'fitted_phot_%s.dat' % snname)

        def get_mangledspec_list(self, extend_spectra=True, extrapolate_spectra=False, verbose=False):
            if self.mode == 'extend_spectra':
                mypath = OUTPUT_DIR + '/%s/mangled_spectra' % self.snname
                onlyfiles = [f for f in os.listdir(mypath) if
                             os.path.isfile(os.path.join(mypath, f)) & ('mangled_spec' in f) & ('.txt' in f)]
                self.mangledspec_list = onlyfiles
                self.mangled_file_path = mypath + '/'
            elif self.mode == 'extrapolate_spectra':
                mypath = OUTPUT_DIR + '/%s/mangled_spectra' % self.snname
                onlyfiles = [f for f in os.listdir(mypath)
                             # if os.path.isfile(os.path.join(mypath, f))&('REmangled_spec' in f)&('.txt' in f)&('_FL' not in f)]
                             if os.path.isfile(os.path.join(mypath, f)) & ('mangled_spec' in f) & ('.txt' in f)]
                self.mangledspec_list = onlyfiles
                self.mangled_file_path = mypath + '/'
            return onlyfiles

        def load_mangledfile(self, file):
            mangled_spec = np.genfromtxt(self.mangled_file_path + file, dtype=None, encoding="utf-8",
                                         names=['wls', 'flux', 'fluxerr', 'mang_mask'])
            return mangled_spec

        def load_manglingfile(self, mjd):
            # if not hasattr(self, "results_mainpath"):
            #    self.check_mangling_file()
            mangling_file = OUTPUT_DIR + '/%s/fitted_phot4mangling_%s.dat' % (self.snname, self.snname)
            if not os.path.isfile(mangling_file):
                raise Exception("I need the file with fitted photometry in order to mangle a spectrum")
            else:
                phot4mangling = pd.read_csv(mangling_file, sep='\t')
                # print (phot4mangling)#self.phot4mangling =
                specmjd = mjd  # float(self.spec_file.replace('mangled_spec_','').replace('.txt',''))
                self.phot4mangling = (phot4mangling[phot4mangling['spec_mjd'] == specmjd])
                if len(self.phot4mangling) < 1:
                    raise Exception(""" ### ERROR: 
    I looked in the file with the PHOTOMETRY for MANGLING 
    (i.e. fitted_phot4mangling_SNNAME.dat).
    I was loading the photometry to mangle/extend the spectrum you are currently loading
    in the GRID. I found NO photometry for it... Maybe you should re run GP fit or check your list of spec.""")

                elif len(self.phot4mangling) > 1:
                    raise Exception(""" ###  TRICKY ERROR: 
    I looked in the file with the PHOTOMETRY for MANGLING 
    (i.e. fitted_phot4mangling_SNNAME.dat).
    I was loading the photometry to mangle/extend the spectrum you are currently loading
    in the GRID. I found two spectra at the exact same MJD and this is problem when
    I try to build the TIMExWLS grid with this.
    Check the phot4mangling.txt file and check if you're MJDs are correct and 
    have the right decimals.""")
                self.avail_filters = [col.replace('_fitflux', '') for col in phot4mangling.columns if
                                      col[-8:] == '_fitflux']

        def create_extended_spec_folder(self):
            save_plot_path = join(OUTPUT_DIR, '%s/TwoDextended_spectra' % SN_NAME)
            if not os.path.exists(save_plot_path):
                os.makedirs(save_plot_path)
            else:
                os.system('rm -rf %s' % save_plot_path)
                os.makedirs(save_plot_path)

            self.save_plot_path = save_plot_path

        def running_mean_std(self, x, y, delta_fix=500.):
            # x = xnan[~np.isnan(ynan)]
            # y = ynan[~np.isnan(ynan)]

            total_bins = int((x.max() - x.min()) / delta_fix)
            bins = np.linspace(x.min(), x.max(), total_bins)
            try:
                delta = bins[1] - bins[0]
                idx = np.digitize(x, bins)
                running_median_x = np.array([np.mean(x[idx == k]) for k in np.arange(1, total_bins, 1)])
                running_mean = np.array([np.mean(y[idx == k]) for k in np.arange(1, total_bins, 1)])
                running_std = np.array([np.std(y[idx == k]) for k in np.arange(1, total_bins, 1)])
            except IndexError:
                running_median_x = np.array([np.mean(x)])
                running_mean = np.array([np.mean(y)])
                running_std = np.array([np.std(y)])

            clean_running_median_x = np.copy(running_median_x)
            clean_running_median_x[running_mean < 0.] = np.nan
            clean_running_mean = np.copy(running_mean)
            clean_running_mean[running_mean < 0.] = np.nan
            clean_running_std = np.copy(running_std)
            clean_running_std[running_mean < 0.] = np.nan

            return clean_running_median_x, clean_running_mean, clean_running_std

        def get_filt_transmission(self, filter_name):

            if 'swift' in filter_name:
                filt_transm = np.genfromtxt(FILTER_PATH + '/SWIFT/%s.dat' % filter_name, dtype=None, encoding="utf-8",
                                            names=['wls', 'flux'])
            else:
                filt_transm = np.genfromtxt(FILTER_PATH + '/GeneralFilters/%s.dat' % filter_name, dtype=None,
                                            encoding="utf-8", names=['wls', 'flux'])
            wls = filt_transm['wls']
            transmission = filt_transm['flux']
            return wls, transmission

        def lam_eff(self, filter_name):
            wls, transmission = self.get_filt_transmission(filter_name)
            return (integrate.trapz(transmission * wls, wls) / integrate.trapz(transmission, wls))

        def load_phot_for_extention(self, file, anchor=False):
            data_spec_mangled = self.load_mangledfile(file)
            mjd = convert2mjd(file)
            self.load_manglingfile(mjd)
            all_fitted_phot_list = []
            fitted_phot_list = []
            fitted_photerr_list = []
            wls_eff = []
            filters4extention = []
            for filt in self.avail_filters:
                lam_eff_value = self.lam_eff(filt)
                fitted_phot = self.phot4mangling['%s_fitflux' % filt].values
                fitted_phot_err = self.phot4mangling['%s_fitfluxerr' % filt].values
                all_fitted_phot_list.append(fitted_phot[0])
                inrange = self.phot4mangling['%s_inrange' % filt].values
                # if (lam_eff_value>max(data_spec_mangled['wls'])+200)|(lam_eff_value<min(data_spec_mangled['wls']-200)):
                if (lam_eff_value > max(data_spec_mangled['wls'])) | (lam_eff_value < min(data_spec_mangled['wls'])):
                    if inrange:
                        if self.verbose: print(filt, lam_eff_value, fitted_phot, mjd)
                        fitted_phot_list.append(fitted_phot[0])
                        fitted_photerr_list.append(fitted_phot_err[0])
                        wls_eff.append(lam_eff_value)
                        filters4extention.append(filt)

            fitted_phot_list = np.array(fitted_phot_list)[np.argsort(wls_eff)]
            fitted_photerr_list = np.array(fitted_photerr_list)[np.argsort(wls_eff)]
            filters4extention = np.array(filters4extention)[np.argsort(wls_eff)]
            wls_eff = np.sort(wls_eff)

            self.phot4extention = {'mjd': mjd, 'wls_eff': wls_eff, 'phot': fitted_phot_list,
                                   'phot_err': fitted_photerr_list,
                                   'names': filters4extention}
            return (self.phot4extention)

        def grid_all_spectraltimeseries(self):
            ungrid_data_wls = []
            ungrid_data_flux = []
            ungrid_data_fluxerr = []
            mjd = []

            DELTA = self.DELTA  # 70.
            for f in self.mangledspec_list:
                spec = self.load_mangledfile(f)
                if self.verbose: print(f, len(spec))
                # smoothed_wls, smoothed_flux, smoothed_flux_err = \
                #        self.running_mean_std(spec['wls'], spec['flux'], delta_fix=DELTA)

                smoothed_wls = spec['wls'][~np.isnan(spec['flux'])]
                smoothed_flux = spec['flux'][~np.isnan(spec['flux'])]
                smoothed_flux_err = spec['fluxerr'][~np.isnan(spec['flux'])]

                ungrid_data_wls.append(smoothed_wls)
                ungrid_data_flux.append(smoothed_flux)
                ungrid_data_fluxerr.append(smoothed_flux_err)
                mjd.append(convert2mjd(f))
            grid_wls = np.arange(1590., 11050., DELTA)
            grid_mjd = np.array(mjd)
            grid_all = pd.DataFrame()
            grid_all_err = pd.DataFrame()

            fig = plt.figure(figsize=(11, 4))  # (15,11))

            for ii in range(len(ungrid_data_flux)):
                grid_flux = np.ones(len(grid_wls))
                grid_fluxerr = np.ones(len(grid_wls))

                minwls = np.min(ungrid_data_wls[ii][~np.isnan(ungrid_data_wls[ii])])
                maxwls = np.max(ungrid_data_wls[ii][~np.isnan(ungrid_data_wls[ii])])
                data_mask = (grid_wls >= minwls) & (grid_wls <= maxwls)
                flux_interp = np.interp(grid_wls[data_mask], ungrid_data_wls[ii], ungrid_data_flux[ii])
                flux_interp_err = np.interp(grid_wls[data_mask], ungrid_data_wls[ii], ungrid_data_fluxerr[ii])
                plt.plot(ungrid_data_wls[ii], ungrid_data_flux[ii], lw=0.5, color='k',
                         label='origina spectrophotometry')
                plt.plot(grid_wls[data_mask], flux_interp, '.r', ms=3, label='"discretized" spectophotometry')
                plt.fill_between(grid_wls[data_mask], flux_interp - flux_interp_err, flux_interp + flux_interp_err,
                                 alpha=0.2, color='r')
                plt.ylabel('Flux')
                plt.xlabel('Wavelength')
                grid_flux[data_mask] = flux_interp
                grid_flux[~data_mask] = np.nan
                grid_fluxerr[data_mask] = flux_interp_err
                grid_fluxerr[~data_mask] = np.nan
                grid_all[str(grid_mjd[ii])] = grid_flux
                grid_all_err[str(grid_mjd[ii])] = grid_fluxerr
            plt.title(self.snname + ' Red: rebbined spectra for grid, Black: original unsmoothed spec')
            plt.show()
            plt.close(fig)
            grid_all = grid_all.set_index(grid_wls)
            grid_all.columns = grid_mjd
            grid_all_err = grid_all_err.set_index(grid_wls)
            grid_all_err.columns = grid_mjd
            self.grids = [grid_wls, grid_mjd, grid_all, grid_all_err]

            return grid_wls, grid_mjd, grid_all, grid_all_err

        def band_flux_modified(self, filter_name, file):

            spec_flux = self.load_mangledfile(file)

            if 'swift' in filter_name:
                filt_transm = np.genfromtxt(FILTER_PATH + '/SWIFT/%s.dat' % filter_name, dtype=None, encoding="utf-8",
                                            names=['wls', 'flux'])
            elif self.snname in CSP_SNe:
                filt_transm = np.genfromtxt(FILTER_PATH + '/Site3_CSP/%s.txt' % filter_name, dtype=None,
                                            encoding="utf-8",
                                            names=['wls', 'flux'])
            else:
                filt_transm = np.genfromtxt(FILTER_PATH + '/GeneralFilters/%s.dat' % filter_name, dtype=None,
                                            encoding="utf-8", names=['wls', 'flux'])

            lam_eff = calc_lam_eff(filt_transm['wls'], filt_transm['flux'])
            filt_transm_interp_func = interpolate.interp1d(filt_transm['wls'], filt_transm['flux'], kind='linear')
            cut_spec = [(spec_flux['wls'] > min(filt_transm['wls'])) & (spec_flux['wls'] < max(filt_transm['wls']))]
            cut_ext_spec = spec_flux[cut_spec]
            filt_transm_interp = filt_transm_interp_func(cut_ext_spec['wls'])
            #
            raw_phot = integrate.trapz(filt_transm_interp * cut_ext_spec['flux'],
                                       cut_ext_spec['wls']) / integrate.trapz(
                filt_transm['flux'], filt_transm['wls'])
            return lam_eff, raw_phot

        def extend_grid_all_spectraltimeseries(self):
            if not hasattr(self, 'grids'):
                self.grid_all_spectraltimeseries()

            grid_notext = (self.grids[2]).copy()
            grid_notext_err = (self.grids[3]).copy()

            for f in self.get_mangledspec_list():
                self.load_phot_for_extention(f)
                phot4ext = self.phot4extention
                for ind in range(len(phot4ext['wls_eff'])):  # np.where(phot4ext['wls_eff']<3500.)[0]:
                    # if th raw associated with the wls doesnt exist create it
                    phot_cut = self.band_flux_modified(phot4ext['names'][ind], f)[1]
                    UVwls = phot4ext['wls_eff'][ind]
                    if phot4ext['names'][ind] in ['swift_UVW1', 'swift_UVW2', 'swift_UVM2']:
                        phot_perc = (100. * phot_cut / (phot4ext['phot'][ind]))
                    else:
                        phot_perc = 0.
                    if self.verbose: print('UV synth versus obs phot %.2f' % phot_perc)
                    if self.verbose: print('UV synth - obs phot %.2E' % (phot4ext['phot'][ind] - phot_cut))
                    if UVwls not in grid_notext.index:
                        grid_notext.loc[UVwls] = np.nan * np.ones(grid_notext.shape[1])
                        grid_notext_err.loc[UVwls] = np.nan * np.ones(grid_notext_err.shape[1])
                    if (phot_perc > 1) & (phot_perc < 99):
                        grid_notext.loc[UVwls][phot4ext['mjd']] = phot4ext['phot'][ind] - phot_cut
                    else:
                        grid_notext.loc[UVwls][phot4ext['mjd']] = phot4ext['phot'][ind]
                    grid_notext_err.loc[UVwls][phot4ext['mjd']] = phot4ext['phot_err'][ind]

            LC_fit = self.open_LCfit_file()
            filters_LC = [i for i in LC_fit.columns[1:] if '_err' not in i]
            mjds_LC = LC_fit['MJD'].values
            min_mjds_LC = []
            max_mjds_LC = []
            for band in filters_LC:
                mjd_filt = mjds_LC[~np.isnan(LC_fit[band].values)]
                if not len(mjd_filt):  # OFEK: added condition
                    continue
                min_mjds_LC.append(min(mjd_filt))
                max_mjds_LC.append(max(mjd_filt))
                if self.lam_eff(band) not in grid_notext.index:
                    grid_notext.loc[self.lam_eff(band)] = np.full(len(grid_notext.columns), np.nan)
                    grid_notext_err.loc[self.lam_eff(band)] = np.full(len(grid_notext.columns), np.nan)

            if self.mode == 'extrapolate_spectra':
                def fill_gaps(min_mjd_phot, max_mjd_phot, spec_mjd, gap_size=2., fill_cadence=2.):
                    f = np.concatenate(([min_mjd_phot], spec_mjd, [max_mjd_phot]))
                    gaps_mjd = f - np.concatenate(([min_mjd_phot], f[:-1]))
                    extention = []
                    for gap, offset in zip(gaps_mjd[gaps_mjd > gap_size], f[gaps_mjd > gap_size]):
                        step = (gap / round(gap / fill_cadence, 0))
                        N = int(round(gap / fill_cadence, 0))
                        for i in range(N - 1):
                            extention.append(offset - (i + 1) * step)
                    return extention

                if (self.snname in pre_bump):
                    early_extrap = np.linspace(min(min_mjds_LC), min(min_mjds_LC) + 5., 15)
                    mjds_extention = np.sort(np.concatenate([early_extrap,
                                                             fill_gaps(min(min_mjds_LC), max(max_mjds_LC),
                                                                       grid_notext.columns)]))
                else:
                    early_extrap = np.linspace(min(min_mjds_LC), min(min_mjds_LC) + 5., 7)
                    mjds_extention = np.sort(np.concatenate([early_extrap,
                                                             fill_gaps(min(min_mjds_LC), max(max_mjds_LC),
                                                                       grid_notext.columns)]))

                if (max(mjds_extention - min(mjds_extention)) > 200.):
                    mjds_extention = mjds_extention[(mjds_extention - min(mjds_extention)) < 200.]
                    if self.verbose:
                        print('Max phase roughly %.2f.' % max((mjds_extention - min(mjds_extention))))
                        print('Chopping this to 200 days after first detection')
                mjds_grid = np.sort(np.concatenate([mjds_extention, grid_notext.columns]))

                for m in range(len(mjds_extention)):
                    if mjds_extention[m] not in grid_notext.columns:
                        grid_notext[mjds_extention[m]] = np.full(len(grid_notext.index), np.nan)
                        grid_notext_err[mjds_extention[m]] = np.full(len(grid_notext_err.index), np.nan)
                # grid_notext.columns = mjds_grid
                for band in filters_LC:
                    no_nan = [(~np.isnan(LC_fit[band].values)) & (~np.isnan(LC_fit[band + '_err'].values))]
                    mjd_filt = mjds_LC[no_nan]
                    if not len(mjd_filt):  # OFEK: added condition
                        continue
                    flux_filt = (LC_fit[band].values)[no_nan]
                    fluxerr_filt = (LC_fit[band + '_err'].values)[no_nan]
                    mask_grid = (mjds_grid > min(mjd_filt)) & (mjds_grid < max(mjd_filt))
                    LC_val = np.full(len(mjds_grid), np.nan)
                    LC_val[mask_grid] = np.interp(mjds_grid[mask_grid], mjd_filt, flux_filt)
                    LCerr_val = np.full(len(mjds_grid), np.nan)
                    LCerr_val[mask_grid] = np.interp(mjds_grid[mask_grid], mjd_filt, fluxerr_filt)
                    for m in range(len(mjds_grid)):
                        if (mjds_grid[m] in mjds_extention) & (m % 3 == 0):
                            grid_notext[mjds_grid[m]].loc[self.lam_eff(band)] = LC_val[m]
                            grid_notext_err[mjds_grid[m]].loc[self.lam_eff(band)] = LCerr_val[m]

            grid_ext = grid_notext.sort_index(inplace=False)
            grid_ext_err = grid_notext_err.sort_index(inplace=False)

            grid_ext_full = grid_ext.sort_index(axis=1, inplace=False)
            grid_ext_err_full = grid_ext_err.sort_index(axis=1, inplace=False)
            self.extended_grid = grid_ext_full
            return grid_ext_full.index, grid_ext_full.columns, grid_ext_full, grid_ext_err_full

        def get_filter_LC(self):
            LC_fit = self.open_LCfit_file()
            filters_LC = [i for i in LC_fit.columns[1:] if '_err' not in i]
            self.avail_filters = filters_LC
            return filters_LC

        def open_LCfit_file(self):
            # if not hasattr(self, "results_mainpath"):
            #    self.check_mangling_file()
            LC_file = join(OUTPUT_DIR, '%s/fitted_phot_%s.dat' % (self.snname, self.snname))
            if not os.path.isfile(LC_file):
                print("I need the file with fitted photometry in order to mangle a spectrum")
            else:
                fittphot = pd.read_csv(LC_file, sep='\t')
            return fittphot

        def get_spec_mjd(self):
            return np.array([convert2mjd(f) for f in self.get_mangledspec_list()])

    # In[7]:

    ## Set some important parameters:
    kernel_wls_scale = 0.05
    kernel_time_scale = 0.3

    ## Set your prior
    PRIOR_file = '/prior_SE.txt'

    # In[8]:

    ## First initialize the class FullMangledSeries_Class
    spec_class = FullMangledSeries_Class(snname=SN_NAME, mode='extrapolate_spectra',
                                         DELTA=30., verbose=False)
    # Load all the mangled spectra
    raw_numbers, raw_numbers_err, off_xa, off_ya, grid_ext_columns = GP2dim.prepare_grid(SN_NAME, spec_class, )

    # Trasform data to LOG, then rescale them to force them to be between 0,1
    # Then reshape the grid to feed the 2dGP
    y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm = GP2dim.transform2LOG_reshape(spec_class,
                                                                                              raw_numbers,
                                                                                              raw_numbers_err,
                                                                                              off_xa, off_ya)

    # Make some useful plots that helps to check the data you are using to interpolate the surface
    # These are saved in OUTPUT_DIR/SNname/TwoDextended_spectra
    GP2dim.make_plots(spec_class, y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm)

    # Set the prior
    Xprior, yprior = GP2dim.setPRIOR(SN_NAME, spec_class, PRIOR_file=PRIOR_file, PRIOR_folder=PRIORs_PATH)

    # In[9]:

    # Do the 2dim GP interpolation
    x1_fill, x2_fill, mu_fill, std_fill = GP2dim.run_2DGP_GRID(spec_class, y_data_nonan, y_data_nonan_err, x1_data_norm,
                                                               x2_data_norm, kernel_wls_scale, kernel_time_scale,
                                                               grid_ext_columns,
                                                               prior=True, points=Xprior, values=yprior)

    # Make some useful plots that helps to check the results
    # These are saved in OUTPUT_DIR/SNname/TwoDextended_spectra
    GP2dim.make_results_plots(spec_class, x1_fill, x2_fill, mu_fill, std_fill)
    mu_fill_conv, std_fill_conv, y_data_conv = GP2dim.transform_back_andPlot(spec_class, x1_fill, x2_fill,
                                                                             mu_fill, std_fill, y_data_nonan)

    # In[20]:

    mu_fill_conv, std_fill_conv, y_data_conv = GP2dim.transform_back_andPlot(spec_class, x1_fill, x2_fill,
                                                                             mu_fill, std_fill, y_data_nonan)

    # In[21]:

    GP2dim.save_plots_files(spec_class, grid_ext_columns, y_data_conv, x1_fill, x2_fill,
                            mu_fill_conv, std_fill_conv)
    spec_class.mode = 'extend_spectra'
    GP2dim.save_plots_files(spec_class, grid_ext_columns, y_data_conv, x1_fill, x2_fill,
                            mu_fill_conv, std_fill_conv)

    # In[ ]:
