import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import scipy.stats
def contains_mask(list1, list2):
    return [element in list2 for element in list1]

def load_dump_and_sim(fraction, maria_cuts = False):
    dumpfile = r"C:\Users\cmeldorf\Desktop\BIASCOR_DATA\PIP_COLE_D5YR_BIASCOR_BIASCORSIM_DES.DUMP\PIP_COLE_D5YR_BIASCOR_BIASCORSIM_DES.DUMP"
    simfile = r"C:\Users\cmeldorf\Desktop\BIASCOR_DATA\FITOPT000.FITRES\FITOPT000.FITRES"
    dumpdf = pd.read_csv(dumpfile, skiprows = 5, sep = '\s+', nrows= 31821607//fraction)
    simdf = pd.read_csv(simfile, skiprows = 11, sep = '\s+', nrows= 2384818//fraction)

    if maria_cuts:
        #Applying Maria's cuts
        simdf = simdf[simdf['SIM_c'] > -0.3]
        simdf = simdf[simdf['SIM_c'] < 0.3]
        simdf = simdf[simdf['x1'] > -3]
        simdf = simdf[simdf['x1'] < 3]
        simdf = simdf[simdf['HOST_ZSPEC'] > 0.025]
        simdf = simdf[simdf['x1ERR'] < 1]
        simdf = simdf[simdf['PKMJDERR'] < 2]
        simdf = simdf[simdf['FITPROB'] > 0.001]
        simdf = simdf[simdf['HOST_NMATCH'] > 0]

        # I am pretty sure I don't want to apply the cuts to dumpdf

    return dumpdf, simdf

def rate_error(a,b):
    print('Changed the def of this function without zeros')
    a_err = np.sqrt(a)
    b_err = np.sqrt(b)
    a_frac = a_err / a
    b_frac = b_err / b
    tot_err = np.sqrt((a_frac**2) + (b_frac**2)) * (a / b)
    return tot_err

def calculate_Vmax_correction(hostlib, max_z, min_z, max_i):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

def calculate_volume(minz, maxz, area):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    rsurv = cosmo.luminosity_distance(maxz)
    survey_sphere = 4/3 * np.pi * rsurv**3
    r_inner = cosmo.luminosity_distance(minz)
    inner_sphere = 4/3 * np.pi * r_inner**3
    survey_sphere -= inner_sphere

    survey_volume = survey_sphere * area
    V = survey_volume / 41252.96125
    return V

