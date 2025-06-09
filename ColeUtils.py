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

def calc_Vmax(max_z, max_i, measured_z, measured_i, mass, bins, min_z = 0):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    cut = (measured_i < max_i) & (measured_z < max_z) & (measured_z > min_z) 
    measured_z = measured_z[cut]
    measured_i = measured_i[cut]
    mass = mass[cut]
    print('Z cutoff', max_z)
    print('Largest z in measured_z', np.max(measured_z))
    measured_z = np.asarray(measured_z)
    measured_i = np.asarray(measured_i)
    rsurv_gal = cosmo.luminosity_distance(max_z).value
    print('Luminosity distance at max z', rsurv_gal)
    robs = cosmo.luminosity_distance(measured_z).value
    print('Luminosity distance at measured z', robs[:10])
    print('measured_i', measured_i[:10])
    mobs = measured_i
    msurv = max_i
    distance_modulus = 10 ** ((msurv - mobs)/5) 
    rmax = robs * distance_modulus
    print('rmax', rmax[:10])
    print(robs/rmax)
    Vmax_correction = np.empty_like(rmax)
    Vmax_correction = (rsurv_gal / rmax)**3
    Vmax_correction[Vmax_correction < 1] = 1
    print('inf redshifts:')
    print(measured_z[np.where(Vmax_correction == np.inf)])
    print('and robs')
    print(robs[np.where(Vmax_correction == np.inf)])
    print('>1 fraction', np.size(Vmax_correction[np.where(Vmax_correction > 1)]) / np.size(Vmax_correction))
    print('max and min', np.max(Vmax_correction), np.min(Vmax_correction))

    Vmax_binned, bin_edges = scipy.stats.binned_statistic(mass, Vmax_correction, bins=bins, statistic='sum')[:2]
    counts, bin_edges = scipy.stats.binned_statistic(mass, Vmax_correction, bins=bins, statistic='count')[:2]
    print('mean vmax correction', np.mean(Vmax_correction))
    print(Vmax_binned/counts)


    return Vmax_binned, bin_edges, Vmax_correction, counts

from matplotlib import rcParams
import matplotlib as mpl
def update_rcParams(key, val):
    if key in rcParams:
        rcParams[key] = val

def LaurenNicePlots():
    update_rcParams('font.size', 10)
    update_rcParams('font.family', 'serif')
    update_rcParams('xtick.major.size', 8)
    update_rcParams('xtick.labelsize', 'large')
    update_rcParams('xtick.direction', "in")
    update_rcParams('xtick.minor.visible', True)
    update_rcParams('xtick.top', True)
    update_rcParams('ytick.major.size', 8)
    update_rcParams('ytick.labelsize', 'large')
    update_rcParams('ytick.direction', "in")
    update_rcParams('ytick.minor.visible', True)
    update_rcParams('ytick.right', True)
    update_rcParams('xtick.minor.size', 4)
    update_rcParams('ytick.minor.size', 4)
    update_rcParams('xtick.major.pad', 10)
    update_rcParams('ytick.major.pad', 10)
    update_rcParams('legend.numpoints', 1)
    update_rcParams('mathtext.fontset', 'cm')
    update_rcParams('mathtext.rm', 'serif')
    update_rcParams('axes.labelsize', 'x-large')
    update_rcParams('lines.marker', 'None')
    update_rcParams('lines.markersize', 1)
    update_rcParams('lines.markeredgewidth', 1.0)
    update_rcParams('lines.markeredgecolor', 'auto')

    cycle_colors = ['navy', 'maroon','darkorange', 'darkorchid', 'darkturquoise', 'darkmagenta', '6FADFA','7D7D7D','black']
    # cycle_colors = ['9F6CE6','FF984A','538050','6FADFA','7D7D7D','black']
    cycle_markers = ['o','^','*','s','X','d', '1','2', '3']
    # cycle_colors = ['darkorchid','darkorange','darkturquoise']
    # cycle_markers = ['o','^','*']
    #+ mpl.cycler(marker=cycle_markers)
    update_rcParams('axes.prop_cycle', mpl.cycler(color=cycle_colors) )