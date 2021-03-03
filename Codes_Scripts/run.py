import json
import os
from glob import glob
from os.path import isfile, basename

from astropy.coordinates import SkyCoord
from astropy.time import Time

from Codes_Scripts.config import *

fallbackepxrange = 20  # days

def format_oscphot(oscphot_path):
    dfosc = pd.read_csv(oscphot_path)

    if SN_NAME not in info_objects['Name'].values:
        info_objects.append(pd.Series(name=SN_NAME))
        info_objects.at[SN_NAME, 'Name'] = SN_NAME
    dfoscF = dfosc[dfosc['upperlimit'] != 'T']
    upperlim = min(dfoscF['time'].values)
    nondet = dfosc[(dfosc['upperlimit'] == 'T') * (dfosc['time'] < upperlim)]
    lowerlim = max(nondet['time'].values) if not nondet.empty else upperlim - fallbackepxrange
    if pd.isna(info_objects.at[SN_NAME, 'MJD_exp_low']):
        info_objects.at[SN_NAME, 'MJD_exp_low'] = lowerlim
    if pd.isna(info_objects.at[SN_NAME, 'MJD_exp_up']):
        info_objects.at[SN_NAME, 'MJD_exp_up'] = upperlim
    info_objects.to_csv(DATAINFO_FILEPATH, sep=' ', index=False)

    dfosc = dfoscF[['time', 'magnitude', 'e_magnitude', 'band', 'instrument', 'telescope', 'source']]
    df = pd.DataFrame(columns='MJD,Mag,Mag_err,band,Instr,source'.split(','))
    if not isfile(FILTERDICT_PATH):
        fdct = {}
    else:
        with open(FILTERDICT_PATH, 'rb') as f:
            fdct = json.load(f)

    idx = 0
    minuslist = []
    for _, row in dfosc.iterrows():
        oscfilt = []
        for x in ['band', 'instrument', 'telescope']:
            oscfilt += [row[x]] if not pd.isna(row[x]) else ['']
        if ' / '.join(oscfilt) not in fdct:
            band = input('OSC band, instrument telescope: ' + str(oscfilt) + ' -- Enter band: ')
            # empty skips this time, "-" ignores this trio always
            if not band:
                continue
            fdct[' / '.join(oscfilt)] = band
            with open(FILTERDICT_PATH, 'w') as f:
                json.dump(fdct, f, indent=2)

        band = fdct[' / '.join(oscfilt)]
        if band != '-':
            df.loc[idx] = {'MJD': row['time'], 'Mag': row['magnitude'], 'Mag_err': row['e_magnitude'], 'band': band,
                           'Instr': row['instrument'], 'source': row['source']}
            idx += 1
        else:
            minuslist.append(' / '.join(oscfilt))

    df.to_csv(oscphot_path.rstrip('.osc'), index=False)
    os.system('rm ' + oscphot_path)

    print('Filters unknown:')
    for x in set(minuslist):
        print(x)


def format_marshallphot(oscphot_path):
    dfosc = pd.read_csv(oscphot_path)
    dfosc['jdobs'] = dfosc['jdobs'] - 2400000.5

    if SN_NAME not in info_objects['Name'].values:
        info_objects.append(pd.Series(name=SN_NAME))
        info_objects.at[SN_NAME, 'Name'] = SN_NAME
    dfoscF = dfosc[dfosc['magpsf'] <= dfosc['limmag']]
    upperlim = min(dfoscF['jdobs'].values)
    nondet = dfosc[(dfosc['magpsf'] > dfosc['limmag']) * (dfosc['jdobs'] < upperlim)]
    lowerlim = max(nondet['jdobs'].values) if not nondet.empty else upperlim - fallbackepxrange
    if pd.isna(info_objects.at[SN_NAME, 'MJD_exp_low']):
        info_objects.at[SN_NAME, 'MJD_exp_low'] = lowerlim
    if pd.isna(info_objects.at[SN_NAME, 'MJD_exp_up']):
        info_objects.at[SN_NAME, 'MJD_exp_up'] = upperlim
    info_objects.to_csv(DATAINFO_FILEPATH, sep=' ', index=False)

    dfosc = dfoscF[['jdobs', 'magpsf', 'sigmamagpsf', 'filter', 'instrument', 'refsys']]
    df = pd.DataFrame(columns='MJD,Mag,Mag_err,band,Instr,source'.split(','))
    if not isfile(FILTERDICT_PATH):
        fdct = {}
    else:
        with open(FILTERDICT_PATH, 'rb') as f:
            fdct = json.load(f)

    idx = 0
    for _, row in dfosc.iterrows():
        oscfilt = []
        for x in ['filter', 'instrument']:
            oscfilt += [row[x]] if not pd.isna(row[x]) else ['']
        if ' / '.join(oscfilt) not in fdct:
            band = input('Marshall band, instrument telescope: ' + str(oscfilt) + ' -- Enter band: ')
            # empty skips this time, "-" ignores this trio always
            if not band:
                continue
            fdct[' / '.join(oscfilt)] = band
            with open(FILTERDICT_PATH, 'w') as f:
                json.dump(fdct, f, indent=2)

        band = fdct[' / '.join(oscfilt)]
        if band != '-':
            df.loc[idx] = {'MJD': row['jdobs'], 'Mag': row['magpsf'], 'Mag_err': row['sigmamagpsf'], 'band': band,
                           'Instr': row['instrument'], 'source': 'ZTF partnership data'}
            idx += 1


    df.to_csv(oscphot_path.rstrip('.osc'), index=False)
    os.system('rm ' + oscphot_path)




def get_photometry(source):
    oscphot_path = join(PHOTOMETRY_PATH, "0_LCs_mags_raw", SN_NAME + '_mag.dat.osc')
    photpath = oscphot_path.rstrip('.osc')
    if not isfile(oscphot_path.rstrip('.osc')):
        if not isfile(oscphot_path):
            if source == 'osc':
                link = 'https://api.sne.space/%s/photometry/time+magnitude+e_magnitude+upperlimit+band+instrument' \
                       '+telescope+source?format=csv&time&magnitude' % SN_NAME
            elif source == 'marshall':
                link = "http://skipper.caltech.edu:8080/cgi-bin/growth/print_lc.cgi?name=" + SN_NAME
            else:
                raise Exception('marshall or osc only')
            os.system('wget -q "%s" -O "%s"' % (link, oscphot_path))
            has_oscphot = False
            if isfile(oscphot_path):
                with open(oscphot_path, 'r') as f:
                    if len(f.readlines()) > 1:
                        has_oscphot = True
            if not has_oscphot:
                raise Exception('Could not get photometry csv from the source.')
        if source == 'osc':
            format_oscphot(oscphot_path)
        elif source == 'marshall':
            format_marshallphot(oscphot_path)

    df = pd.read_csv(photpath)
    print('Filters obtained:')
    print(df.band.value_counts())
    # print('Have:', join('Photometry', "0_LCs_mags_raw", SN_NAME + '_mag.dat.osc'))


def get_spectroscopy(supyfitspecdir):
    specdir = join(DATASPEC_PATH, '1_spec_original', SN_NAME)

    havewisecsv = False
    if isfile(join(supyfitspecdir, 'wiserep_spectra.csv')):
        wisecsv = pd.read_csv(join(supyfitspecdir, 'wiserep_spectra.csv'), delimiter=',')
        havewisecsv = True

    os.system('mkdir -p ' + specdir)
    if not havewisecsv:
        wisecsv = []
    for sf in glob(join(supyfitspecdir, '*')):
        bsnm = basename(sf)
        if bsnm != 'photometry':
            os.system('cp ' + sf + ' ' + join(specdir, bsnm))
            if not havewisecsv:
                date = bsnm.split('_')[1]
                jd = Time({'year': int(date[:4]), 'month': int(date[4:6]), 'day': int(date[6:8])}).jd
                wisecsv.append([bsnm, jd, pd.NA, pd.NA, pd.NA])

    if not havewisecsv:
        wisecsv = pd.DataFrame(wisecsv, columns=['Ascii file', 'JD', 'Redshift', 'Obj. RA', 'Obj. DEC'])

    # list
    speclistapath = join(DATASPEC_PATH, '1_spec_lists_original', SN_NAME + '.list')
    lista = []
    existingmjd = []
    for _, row in wisecsv.iterrows():
        redshift = row['Redshift']
        ra = row['Obj. RA']
        dec = row['Obj. DEC']

        mjd = float(row['JD']) - 2400000.5
        if existingmjd:
            while abs(min(existingmjd, key=lambda x: abs(
                    x - mjd)) - mjd) < 0.01:  # PyCoCo throws an error if there are two spectra with the same mjd...
                mjd = mjd + 0.01 if min(existingmjd, key=lambda x: abs(x - mjd)) <= mjd else mjd - 0.01
        existingmjd.append(mjd)

        listarow = (join(specdir, row['Ascii file']), SN_NAME, str(mjd), str(redshift))
        lista.append(listarow)

    with open(speclistapath, 'w') as f:
        for listarow in lista:
            f.write('\t'.join(listarow) + '\n')

    # info
    if SN_NAME not in info_objects['Name'].values:
        info_objects.append(pd.Series(name=SN_NAME))
        info_objects.at[SN_NAME, 'Name'] = SN_NAME
    if not pd.isna(ra) and not pd.isna(dec):
        c = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
        ra_, dec_ = c.to_string('hmsdms').replace('h', ':').replace('m', ':').replace('d', ':').replace('s', '').split()
    else:
        ra_, dec_ = pd.NA, pd.NA
    info_objects.at[SN_NAME, 'RA'] = ra_
    info_objects.at[SN_NAME, 'Dec'] = dec_
    info_objects.at[SN_NAME, 'z'] = redshift

    info_objects.to_csv(DATAINFO_FILEPATH, sep=' ', index=False)


def runpycoco():
    import Codes_Scripts.a0_1_Smooth_spectra as module
    del module
    import Codes_Scripts.a0_toFlux as module
    del module
    import Codes_Scripts.a1_LC_DustCorrection as module
    del module
    import Codes_Scripts.a2_LC_modelRising as module
    del module
    import Codes_Scripts.a3_LC_modelExpDecay as module
    del module
    import Codes_Scripts.a4_LCfit as module
    del module
    import Codes_Scripts.a5_Mangle_spectra as module
    del module
    import Codes_Scripts.a6_TwoDim_UVExtend_Extrapolate as module
    del module
    import Codes_Scripts.a7_Rimangle as module
    del module



if __name__ == '__main__':
    # get_spectroscopy(SN_SUPYFITDIR)
    # get_photometry('osc')
    runpycoco()
