#!/usr/bin/env python
# coding: utf-8

# # HO FATTO UN CHANGE STUPIDO in LOCAL/PUBLIC

# # Convert MAGS (Vega or AB mag system) to Flux You can use the customized package what_the_flux (written by Chris
# Frohmaier, with a small contribution from Maria Vincenzi).
# 
# Here you can find a quick example.
# 
# Remember that:
# 1) 90% of the times Core Collapse SNe are presented in the standard filter system (not in natural
# filter system). This means that photometry is presented as it would have been observed by the "standard"
# Bessell/SDSS filter set. The "standard" Bessell/SDSS filter sets are stored in inputs/Filters/GeneralFilters. The
# main exception is CSP data, usually presented in natural system. Therfore, for CSP objects we use
# inputs/Filters/Site3_CSP.
# 
# 2) Magnitudes in Bessell filters (UBVRI) are usually calculated in the Vega system and calibrated to Landolt standards
# 
# 3) Magnitudes in SDSS filters (ugriz) are usually calculated in the AB system and calibrated to Smith 2002 standards
# 
# 4) Magnitudes in Space Telescope SWIFT filters (W2,M2,W1,U,B,V) are usually calculated in the Vega system. See
# Peter Brown's paper for calibration of the instrument.
# 

# In[1]:
import json

import Codes_Scripts.excludefilt
from Codes_Scripts.config import *


def main(SN_NAME):
    exclude_filt, Bessell_V_like, swift_V_like, Bessell_B_like, swift_B_like = Codes_Scripts.excludefilt.main(SN_NAME)

    INPUT_LC_PATH = join(PHOTOMETRY_PATH, "0_LCs_mags_raw", "%s_mag.dat") % SN_NAME
    OUTPUT_LC_PATH = join(PHOTOMETRY_PATH, "1_LCs_flux_raw", "%s.dat") % SN_NAME
    # FILTERS_PATH = join(COCO_PATH , "Inputs/Filters")

    with open(FILTERMAGSYS_PATH, 'r') as f:
        magsysdict = json.load(f)

    # In[10]:

    CSP_SNe = []  # OFEK
    # ['SN2004fe', 'SN2005bf', 'SN2006V', 'SN2007C', 'SN2007Y',
    #            'SN2009bb', 'SN2008aq', 'SN2006T', 'SN2004gq', 'SN2004gt',
    #            'SN2004gv', 'SN2006ep', 'SN2008fq', 'SN2006aa']

    # In[11]:

    fout = open(OUTPUT_LC_PATH, 'w')
    fout.write('# From Mags to Flux, not MW/Host dust corrected\n')
    fout.close()

    phot_mags = pd.read_csv(INPUT_LC_PATH, comment='#')
    filter_set = 'GeneralFilters' if SN_NAME not in CSP_SNe else 'Site3_CSP'

    fluxes = []
    fluxes_err = []
    filter_set_list = []

    for index, row in phot_mags.iterrows():
        w, t = wtf.loadFilter(join(FILTERS_PATH, '%s/%s.dat') % (filter_set, row['band']))
        # OFEK: capital is Vega, noncapital is AB
        if row['band'] not in magsysdict:
            letter = row['band'].split('_')[1][0]
            if letter.islower():
                magsysdict[row['band']] = 'AB'
            else:
                magsysdict[row['band']] = 'Vega'
        magsys = magsysdict[row['band']]
        if magsys == 'Vega':
            band = wtf.Band_Vega(w, t)
        elif magsys == 'AB':
            band = wtf.Band_AB(w, t)
        else:
            raise Exception('Unknown magsystem ' + magsys)

        flux_withMW = wtf.mag2flux(band, row['Mag']).value
        flux_err_withMW = wtf.ERRmag2ERRflux(band, row['Mag'], row['Mag_err']).value
        fluxes.append(flux_withMW)
        fluxes_err.append(flux_err_withMW)
        filter_set_list.append(filter_set)

    phot_mags['Flux'] = fluxes
    phot_mags['Flux_err'] = fluxes_err
    phot_mags['FilterSet'] = filter_set_list
    phot_mags[['MJD', 'Mag', 'Mag_err', 'Flux', 'Flux_err', 'band', 'Instr', 'FilterSet']].to_csv(OUTPUT_LC_PATH,
                                                                                                  index=False)

    with open(FILTERMAGSYS_PATH, 'w') as f:
        json.dump(magsysdict, f, indent=2)
