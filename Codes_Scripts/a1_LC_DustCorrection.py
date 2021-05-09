#!/usr/bin/env python
# coding: utf-8

# # Correct Photometry for dust extinction

# Correct broad-band photometry for Host and MW extinction (if this has not been already done.. 99% of the time the photometry is published before dust corrections).
# 
# Assumptions: R_V always equal 3.1

# In[1]:
import json
import os

import numpy as np

import Codes_Scripts.excludefilt
from Codes_Scripts.config import *


def main(SN_NAME):
    exclude_filt, Bessell_V_like, swift_V_like, Bessell_B_like, swift_B_like = Codes_Scripts.excludefilt.main(SN_NAME)

    # In[3]:

    CSP_SNe = []  # OFEK
    # ['SN2004fe', 'SN2005bf', 'SN2006V', 'SN2007C', 'SN2007Y',
    #            'SN2009bb', 'SN2008aq', 'SN2006T', 'SN2004gq', 'SN2004gt',
    #            'SN2004gv', 'SN2006ep', 'SN2008fq', 'SN2006aa']

    with open(FILTERMAGSYS_PATH, 'r') as f:
        magsysdict = json.load(f)

    class SNPhotometryClass():
        """Class with photometry for each object:
                - load the photometry from the DATA folder
                - get the phootmetry in each filter
                - plot the raw photometry
                - fit the photometry using GP
        """

        def __init__(self, lc_path, snname, verbose=False):
            """
            """
            self.lc_data_path = lc_path + '/'
            self.snname = snname
            self.set_data_directory(verbose)

        def set_data_directory(self, verbose):
            """
            Set a new data directory path.
            Enables the data directory to be changed by the user.
            """
            SNphotometry_PATH = os.path.join(self.lc_data_path, '%s.dat' % self.snname)

            try:
                if verbose: print('Looking for Photometry for %s in%s' % (self.snname, SNphotometry_PATH))
                if os.path.isfile(SNphotometry_PATH):
                    if verbose: print('Got it!')
                    self.sn_rawphot_file = SNphotometry_PATH
                    pass
                else:
                    if not os.path.isdir(self.lc_data_path):
                        print('I cant find the directory with photometry. Check %s' % self.lc_data_path)
                        pass
                    else:
                        print('I cant find the file with photometry. Check %s' % SNphotometry_PATH)
                        pass

            except Exception as e:
                print(e)

        def load(self, verbose=False):
            """
            Loads a single photometry file.
            with ('MJD', 'flux', 'flux_err', 'filter')

            Parameters
            - verbose
            ----------
            Returns
            - photometry in all filters
            -------
            """
            if verbose: print('Loading %s' % self.sn_rawphot_file)
            try:
                lc_file = pd.read_csv(self.sn_rawphot_file,
                                      dtype=None, encoding="utf-8")
                mask_filt = np.array([f not in exclude_filt for f in lc_file['band']])
                lc_no_badfilters = lc_file[mask_filt]
                mask_filt = np.array([~np.isnan(f) for f in lc_no_badfilters['Flux']])
                self.phot = lc_no_badfilters[mask_filt]
                self.avail_filters = np.unique(self.phot['band'])
                if verbose: print('Photometry loaded')

            except Exception as e:
                print(e)
                print('Are you sure you gave me the right format? Check documentation in case.')

        def get_availfilter(self, verbose=False):
            """
            get available filter for this SN
            """
            # if photometry is not already loaded, load it!
            if (not hasattr(self, "phot")) | (not hasattr(self, "avail_filters")):
                self.load()
            return self.avail_filters

        def get_singlefilter(self, single_filter, verbose=False):
            """
            Loads from photometry file just 1 filter photometry.
            with ('MJD', 'flux', 'flux_err', 'filter')

            Parameters
            - verbose
            ----------
            Returns
            - photometry in all filters
            -------
            """
            # if photometry is not already loaded, load it!
            if not hasattr(self, "phot"):
                self.load()

            if not (isinstance(single_filter, str)):
                print('Single filter string please')
                return None

            if single_filter not in self.avail_filters:
                if verbose: print('Looks like the filter you are looking for is not available')
                return None

            filt_index = self.phot['band'] == single_filter
            return self.phot[filt_index]

        def corr_dust_singlefilter(self, filter_name):
            if not hasattr(self, "corr_factors"):
                self.corr_factors = {}
            corr_factors_dict = self.corr_factors

            self.get_dust()
            self.get_redshift()
            RV = RV_dict[self.SNType]

            if 'swift' in filter_name:
                w, t = w, t = wtf.loadFilter(FILTER_PATH + '/SWIFT/%s.dat' % filter_name)
            elif self.snname in CSP_SNe:
                w, t = w, t = wtf.loadFilter(FILTER_PATH + '/Site3_CSP/%s.txt' % filter_name)
            else:
                w, t = w, t = wtf.loadFilter(FILTER_PATH + '/GeneralFilters/%s.dat' % filter_name)

            w_SNframe = w / (1. + self.redshift)

            if magsysdict[filter_name] == 'Vega':
                band = wtf.Band_Vega(w_SNframe, t)
            elif magsysdict[filter_name] == 'AB':
                band = wtf.Band_AB(w_SNframe, t)
            ext_corr_Host = (1. / band.extinction(self.Hostebv, 'CCM', r_v=RV)).value
            print(filter_name, 'RV', RV, 'Host dust correction %.3f' % ext_corr_Host)

            if magsysdict[filter_name] == 'Vega':
                band = wtf.Band_Vega(w, t)
            elif magsysdict[filter_name] == 'AB':
                band = wtf.Band_AB(w, t)
            ext_corr_MW = (1. / band.extinction(self.MWebv, 'CCM')).value

            corr_factors_dict[filter_name] = ext_corr_Host * ext_corr_MW
            self.corr_factors = corr_factors_dict

        def get_dust(self):
            if self.snname not in info_objects.Name.values:
                raise Exception('This SN is not in the info.dat file')
            else:
                info_singleobj = info_objects[info_objects.Name == self.snname]
                self.MWebv = info_singleobj['EBV_MW'].values[0]
                Host_ebv = info_singleobj['EBV_host'].values[0]
                self.Hostebv = Host_ebv
                self.SNType = (info_singleobj['Type'].values[0])

        def get_redshift(self):
            info_singleobj = info_objects[info_objects.Name == self.snname]
            self.redshift = info_singleobj['z'].values[0]

        def correct_final_LC(self, name_file=None):
            for ff in self.get_availfilter():
                self.corr_dust_singlefilter(ff)

            lc_file = pd.DataFrame(self.phot)
            corr_dust_array = [self.corr_factors[f] for f in lc_file['band']]
            lc_file['Flux_corr'] = corr_dust_array * lc_file['Flux'].values
            lc_file['Flux_corr_err'] = corr_dust_array * lc_file['Flux_err'].values

            return lc_file

    # ## Dust correction

    # In[6]:

    # If you are assuming RV = 3.1
    RV_dict = {'Ic': 3.1, 'Ic-BL': 3.1, 'Ib': 3.1, 'Ibc': 3.1, 'IIb': 3.1, 'II': 3.1, 'IIn': 3.1,
               'Ia': 3.1}  # OFEK: added Ia, Ibc

    # If you want to do something more sophisticated and set a different R_V for different SN types set this:
    # RV_dict = {'Ic':4.3 , 'Ic-BL':4.3, 'Ib':2.6, 'IIb':1.1, 'II':3.1, 'IIn':3.1}

    # In[10]:

    snname = SN_NAME

    # In[11]:

    sn_phot = SNPhotometryClass(lc_path=join(PHOTOMETRY_PATH, '1_LCs_flux_raw'), snname=snname, verbose=True)
    sn_phot.load()

    # In[12]:

    sn_phot.get_availfilter()
    df = sn_phot.correct_final_LC()

    df_to_print = df[['MJD', 'band', 'Flux_corr', 'Flux_corr_err', 'FilterSet', 'Instr']]
    df_to_print.to_csv(join(PHOTOMETRY_PATH, '2_LCs_dust_corrected') + '/%s.dat' % snname, na_rep='nan', index=False,
                       header=['#MJD', 'band', 'Flux', 'Flux_err', 'FilterSet', 'Instr'])

    # In[ ]:
