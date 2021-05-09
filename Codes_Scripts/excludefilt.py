from os import system

from Codes_Scripts.config import *


def main(SN_NAME):
    system('mkdir -p ' + join(OUTPUT_DIR, SN_NAME))

    exclude_filt = open(join(OUTPUT_DIR, SN_NAME, 'exclude_filt'), 'r').readlines()
    exclude_filt = [x.strip('\n').strip() for x in exclude_filt]

    # OFEK: variable Bessell_V_like replaced the literal "Bessell_V" throughout the code. Same for "Bessell_B", "swift_B/V"
    fourfilters = open(join(OUTPUT_DIR, SN_NAME, 'VVBB'), 'r').readlines()
    fourfilters = [x.strip('\n').strip() for x in fourfilters]

    Bessell_V_like = fourfilters[0]  # set zero time for comparbility with prior (only option)
    # & early extrap (must have pre- and post-peak) 1st priority
    swift_V_like = fourfilters[1]  # early extrap (must have pre- and post-peak) 2nd priority
    Bessell_B_like = fourfilters[2]  # set zero time for comparbility with prior, 1st priority
    swift_B_like = fourfilters[3]  # set zero time for comparbility with prior, 2nd priority

    return exclude_filt, Bessell_V_like, swift_V_like, Bessell_B_like, swift_B_like
