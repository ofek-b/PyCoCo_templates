import os
from glob import glob
from os.path import basename, isdir

from Codes_Scripts.config import *


def names():
    paths = glob(join(PHOTOMETRY_PATH, "0_LCs_mags_raw", '*'))
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = ['MJD', 'Mag', 'Mag_err', 'band', 'Instr', 'source']
            df.to_csv(p, sep=',', index=False)
            # dct=dict(MJD=[], Mag=[], Mag_err=['e_mag'], band=['Filter'], Instr=['Telescope','Tel'], source=[])
        except:
            print(p)


def split_phot_file(path='/home/ofekb/Desktop/ptfs.txt'):
    df = pd.read_csv(path, delim_whitespace=True)
    ans = [pd.DataFrame(y) for x, y in df.groupby('ID', as_index=False)]
    for df_ in ans:
        df_ = df_.replace('g', 'ZTF_g').replace('r', 'ZTF_r').replace('i', 'ZTF_i')
        df_['source'] = ['Lunnan_2017' for _ in df_.iterrows()]
        df_[['MJD', 'mag', 'e_mag', 'Filter', 'Tel', 'source']].to_csv(
            join(PHOTOMETRY_PATH, "0_LCs_mags_raw", df_.ID.to_list()[0] + '_mag.dat'), sep=',', index=False)


def sn20116hnk(path='/home/ofekb/Desktop/16hnk_ph', snname='SN2016hnk', source='2019A&A...630A..76G'):
    files = glob(join(path, '*'))
    df_all = pd.DataFrame()
    for f in files:
        filt = basename(f).split('.out_')[-1]
        df = pd.read_csv(f, delim_whitespace=True, skiprows=3)
        df['source'] = [source for _ in df.iterrows()]
        df['Filter'] = [filt for _ in df.iterrows()]
        df = df[['MJD', 'Mag', 'DMag', 'Filter', 'Telescope', 'source']]
        df_all = pd.concat([df_all, df])
    df_all.to_csv(join(PHOTOMETRY_PATH, "0_LCs_mags_raw", snname + '_mag.dat'), sep=',', index=False)


def make_clones(snname, supyfitspecdir, specfile2keep):
    wisecsv = pd.read_csv(join(supyfitspecdir, 'wiserep_spectra.csv'), delimiter=',')
    specfiles = [f for f in glob(join(supyfitspecdir, '*')) if
                 basename(f) not in [specfile2keep, 'wiserep_spectra.csv']]
    exfile_path = join(OUTPUT_DIR, snname, 'exclude_filt')
    vvbb_path = join(OUTPUT_DIR, snname, 'VVBB')
    for f in specfiles:
        wisecsv2 = wisecsv.loc[(wisecsv['Ascii file'] == specfile2keep) | (wisecsv['Ascii file'] == basename(f))]
        mjd1 = int(round( wisecsv.loc[wisecsv['Ascii file'] == specfile2keep]['JD'] - 2400000.5))
        mjd2= int(round( wisecsv.loc[wisecsv['Ascii file'] == basename(f)]['JD'] - 2400000.5))
        dir2 = supyfitspecdir.rstrip('/') + '_%s_%s' % (mjd1, mjd2)
        snname2 = snname + '_%s_%s' % (mjd1, mjd2)
        outdir2 = join(OUTPUT_DIR, snname2)
        if isdir(dir2):
            continue

        os.system('mkdir -p %s' % dir2)
        os.system('cp %s %s' % (join(supyfitspecdir, specfile2keep), dir2))
        os.system('cp %s %s' % (f, dir2))
        os.system('mkdir -p %s' % outdir2)
        os.system('cp %s %s' % (exfile_path, outdir2))
        os.system('cp %s %s' % (vvbb_path, outdir2))
        wisecsv2.to_csv(join(dir2, 'wiserep_spectra.csv'), sep=',', index=False)

        if snname2 not in info_objects['Name'].values:
            info_objects.append(pd.Series(name=snname2))
            info_objects.at[snname2, 'Name'] = snname2
            for col in ['z', 'RA', 'Dec', 'FullType', 'Type', 'Rich_Type', 'EBV_MW', 'EBV_host', 'Ref_Host',
                        'MJD_exp_low', 'MJD_exp_up']:
                info_objects.at[snname2, col] = info_objects.at[snname, col]
            info_objects.to_csv(DATAINFO_FILEPATH, sep=' ', index=False)

        origph, ph2 = join(PHOTOMETRY_PATH, "0_LCs_mags_raw", snname + '_mag.dat'), join(PHOTOMETRY_PATH,
                                                                                         "0_LCs_mags_raw",
                                                                                         snname2 + '_mag.dat')
        os.system('cp %s %s' % (origph, ph2))


if __name__ == '__main__':
    make_clones(snname='SN2005cf',
                supyfitspecdir=join(environ['HOME'], 'DropboxWIS/my_spectra_collections/Ia-norm/2005cf'),
                specfile2keep='2005cf_2005-06-12_00-00-00_SSO-2.3m_DBS_SUSPECT.dat')
