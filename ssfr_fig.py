import scipy
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt
from ColeUtils import contains_mask, load_dump_and_sim, calc_Vmax, calculate_volume, LaurenNicePlots
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
MAX_Z = 0.75
MIN_Z = 0.2
max_r = 24.5

LaurenNicePlots()

# Load real data

hostgals = fits.open(r"C:\Users\cmeldorf\Downloads\DES-SN5YR_DES_HEAD.FITS\DES-SN5YR_DES_HEAD.FITS")[1].data
cosmo_sne = pd.read_csv(r"C:\Users\cmeldorf\Downloads\DES-SN5YR_HD+MetaData.csv")[:1635]


#hostlib = pd.read_table(r"C:\Users\cmeldorf\Downloads\DES-SN5YR_DES.HOSTLIB\DES-SN5YR_DES.HOSTLIB", comment = '#', skiprows = 57, sep = '\s+')

hostlib = pd.read_table(r"C:\Users\cmeldorf\Downloads\corrected_deduped_with_magauto_logsfr_upto_4.HOSTLIB+HOSTNBR\corrected_deduped_with_magauto_logsfr_upto_4.HOSTLIB+HOSTNBR", skiprows = 2, comment = '#', sep = '\s+')


mask = contains_mask(hostgals['SNID'], cosmo_sne['CID'].values)
hostgals = hostgals[mask]
hostgals = hostgals[hostgals['SNID'].argsort()]
snid_array = [int(element) for element in hostgals['SNID']]
cid_array = [int(element) for element in cosmo_sne['CID'].values[:1635]]
np.testing.assert_array_equal(snid_array, cid_array)
cosmo_fields = pd.read_csv(r"C:\Users\cmeldorf\Downloads\DES_5Y.csv")
np.testing.assert_array_equal(cosmo_fields['CIDint'].values[:1635], cid_array)
cosmo_fields = cosmo_fields['FIELD'][:1635]
cosmo_sne['FIELD'] = cosmo_fields
cosmo_fields[(cosmo_fields == 'C3') | (cosmo_fields == 'X3')] = 1
cosmo_fields[cosmo_fields !=1] = 0



print('cutting real data to z < %.2f and z > %.2f' % (MAX_Z, MIN_Z))


cut = (cosmo_sne['zHD'] < MAX_Z) & (cosmo_sne['zHD'] > MIN_Z) & (hostgals['HOSTGAL_MAG_r'] < max_r)
hostgals = hostgals[cut]
cosmo_sne = cosmo_sne[cut]
cosmo_fields = cosmo_fields[cut]
hostlib = hostlib[(hostlib['ZTRUE'] < MAX_Z) & (hostlib['ZTRUE'] > MIN_Z) & (hostlib['r_obs_auto'] < max_r)]

real_array = np.array([cosmo_sne['zHD'], hostgals['HOSTGAL_MAG_r'], cosmo_sne['c'], cosmo_fields, cosmo_sne['x1']]).T


# Getting Field information
fields = np.array(['C1', 'C2', 'C3', 'X1', 'X2', 'X3', 'S1', 'S2', 'E1', 'E2'])
field_ras = np.array([54.2743,  54.2743, 52.6484, 34.4757, 35.6645, 36.4500,42.8200, 41.1944, 7.8744 , 9.5000,])
field_decs = np.array([-27.1116, -29.0884, -28.1000, -4.9295, -6.4121, -4.6000, 0.0000, -0.9884, -43.0096,  -43.9980])  


poly_dict = np.load('poly_dict.npy', allow_pickle=True).item()

############ Load spec efficiency data ############
file_path = r"C:\Users\cmeldorf\Desktop\SEARCHEFF_zHOST_DES-SN5YR.DAT"
df = pd.DataFrame(columns=['Year', 'Field', 'r_obs_auto', 'obs_gr_auto', 'HOST_EFF'])

with open(file_path, 'r') as file:
    year = 'test'
    field = 'test'
    for i, line in enumerate(file):   
        if i < 20:
            continue
        line = line.split()
        if 'Year' in line: 
            year = line[-1]
        if 'FIELDLIST:' in line:
            field = line[-1]  
        if 'HOSTEFF:' in line:
            new_rows = pd.DataFrame({'Year': year, 'Field': field, 'r_obs_auto': float(line[1]), 'obs_gr_auto': float(line[2]), 'HOST_EFF': float(line[3])}, index = [0])
            df = pd.concat([df, new_rows], ignore_index=True)
spec_df =df

spec_df = spec_df.groupby(['Field', 'r_obs_auto'])
host_spec_effs = spec_df['HOST_EFF'].mean().reset_index()


############ Calculate corrections ############
corrections = []


for f, c, x1, z, mr in zip(cosmo_sne['FIELD'], cosmo_sne['c'], cosmo_sne['x1'], cosmo_sne['zHD'], hostgals['HOSTGAL_MAG_r']):
    f_reduced = f.split('+')[-1]
    c_index = -0.3 if c < 0 else 0.0
    x_index = -3 if x1 < 0 else 0
    p = poly_dict[f_reduced + '_' + str(c_index) + '_' + str(x_index)]

    host_spec_effs_cut = host_spec_effs[[(f_reduced in i) for i in host_spec_effs['Field'].values]]
    interp = scipy.interpolate.interp1d(host_spec_effs_cut['r_obs_auto'], host_spec_effs_cut['HOST_EFF'], kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    #plt.plot(np.linspace(18, 24.5, 100), interp(np.linspace(18, 24.5, 100)), linestyle = '--')
    #plt.axvline(mr, color='C0', linestyle='--', label='Host mag r = %.2f' % mr)
    host_eff = interp(mr)
    #plt.axhline(host_eff, color='C1', linestyle='--', label='Host Eff = %.2f' % host_eff)
    #plt.show()
    #break
    if host_eff == 0:
        print('Host Eff is zero, skipping')
        corrections.append(np.nan)
        continue

    correction = 1/(p(z) * host_eff)
    if correction < 1:
        print('Correction is less than 1, boosting to 1')
    corrections.append(correction)
    

sullivan_style_corrections = np.ones_like(corrections)
sullivan_style_corrections[(cosmo_sne['zHD'] > 0.2) & (cosmo_sne['zHD'] < 0.4)] = 1/0.331
sullivan_style_corrections[(cosmo_sne['zHD'] > 0.4) & (cosmo_sne['zHD'] < 0.6)] = 1/0.310
sullivan_style_corrections[(cosmo_sne['zHD'] > 0.6) & (cosmo_sne['zHD'] < 0.75)] = 1/0.248
#### Plot Rates #####
#Plot cell
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
from ColeUtils import calculate_volume

start = np.min(cosmo_sne['PKMJD'][cosmo_sne['PKMJD']>56000])
stop = np.max(cosmo_sne['PKMJD'][cosmo_sne['PKMJD']<56800])
dur = stop - start
total_time = dur * 5 / 365.25

max_z_deep = np.max(cosmo_sne['zHD'][real_array[:,3] == 1])
min_z_deep = np.min(cosmo_sne['zHD'][real_array[:,3] == 1])

max_z_shallow = np.max(cosmo_sne['zHD'][real_array[:,3] == 0])
min_z_shallow = np.min(cosmo_sne['zHD'][real_array[:,3] == 0])

V_SN_deep = calculate_volume(min_z_deep, max_z_deep, 27 * 2 /10)
V_SN_shallow = calculate_volume(min_z_shallow, max_z_shallow, 27 * 8/10)

V_G = calculate_volume(MIN_Z, MAX_Z, 2.7)  #Volume of the entire DES-SN survey

V_SN = np.empty_like(real_array[:,3])
V_SN[real_array[:,3] == 1] = V_SN_deep
V_SN[real_array[:,3] == 0] = V_SN_shallow


volume_factor = (V_G/V_SN).astype(float)
print('Volume Factor', volume_factor)
print('Total Time', total_time)

bins = np.arange(-11, -8.0, 0.5)


for i in hostlib.columns:
    if 'SFR' in i:
        sfr_col_name = i
        print('Found SFR column:', sfr_col_name)
        break


hostlib['LOGsSFR'] = np.log10( 10**hostlib[sfr_col_name] / 10**hostlib['LOGMASS'] )
'''
Vmax, Vmaxbins, Vmax_correction, counts = calc_Vmax(max_z = MAX_Z, max_r = max_r, 
                                            measured_z = hostlib['ZTRUE'],
                                             measured_i = hostlib['r_obs_auto'], 
                                             statistic = hostlib['LOGsSFR'], 
                                             bins = bins,
                                             min_z = MIN_Z)
'''

# Calculate Vmax correction for hostlib #######################################
import kcorrect
from ColeUtils import VVmax
#Initialize kcorrect with the desired filter responses
responses = ['decam_g', 'decam_r', 'decam_i', 'decam_z']
print('Initializing kcorrect with responses:', responses)
kc = kcorrect.kcorrect.Kcorrect(responses=responses)

catalogue = hostlib
redshift = catalogue.iloc[:]['ZTRUE']
mags = catalogue.iloc[:][['g_obs_auto', 'r_obs_auto', 'i_obs_auto', 'z_obs_auto']]
magerrs = catalogue.iloc[:][['g_obs_auto', 'r_obs_auto', 'i_obs_auto', 'z_obs_auto']]/10
maggies = 10 ** (-0.4 * mags)
maggie_err = 10 ** (-0.4 * magerrs)
ivar = maggie_err ** (-2)
ivar = ivar.values
ivar[ivar > 1e+18] = 1e+18

# "coeffs" is a [5]-array with coefficients multiplying each template
coeffs = kc.fit_coeffs(redshift=redshift, maggies=maggies, ivar=ivar)

# "k" is a [5]-array with the K-corrections in magnitude units
k = kc.kcorrect(redshift=redshift, coeffs=coeffs)


abs_mags = kc.absmag(redshift=redshift, maggies=maggies, ivar=ivar, coeffs=coeffs)[:,1]  

Vmax_correction = VVmax(redshifts=redshift, observed_mag_r=catalogue['r_obs_auto'], 
             abs_r_mag=abs_mags, zmin_survey=0.2, zmax_survey=0.6)

Vmax = scipy.stats.binned_statistic(catalogue['LOGsSFR'], Vmax_correction,
                                    bins=bins, statistic='sum')[0]

##################### END VMAX SECTION ########################

cosmo_sne['HOST_LOGsSFR'] = np.log10(10**(hostgals['HOSTGAL_LOGSFR']) / (10**cosmo_sne['HOST_LOGMASS']))  # Convert to log SFR


total_mass = scipy.stats.binned_statistic(hostlib['LOGsSFR'], (10**hostlib['LOGMASS']) * Vmax_correction, 
                                        bins = bins, statistic = 'sum')[0]


N_SN_no_corr = np.histogram(cosmo_sne['HOST_LOGsSFR'], bins = bins)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

N_SN = scipy.stats.binned_statistic(cosmo_sne['HOST_LOGsSFR'], corrections * volume_factor,  bins = bins, statistic = 'sum')[0]

#N_SN_sullivan = scipy.stats.binned_statistic(cosmo_sne['HOST_LOGsSFR'], sullivan_style_corrections,  bins = bins, statistic = 'sum')[0]

N_SN_no_corr_per_mass = N_SN_no_corr[0] / total_mass
N_SN_per_mass = N_SN / total_mass


#plt.scatter(cosmo_sne['HOST_LOGsSFR'], np.log10(corrections), s = 1, alpha = 0.5, c=cosmo_sne['HOST_LOGMASS'], cmap='viridis', label='Hostlib')
#plt.colorbar(label='Correction Factor')
#plt.ylim(-12, -8)
#plt.xlim(8, 12)
#plt.show()

plt.subplot(1, 3, 1)
plt.yscale('log')
norm0 = np.sum(bin_width * N_SN_no_corr[0])
plt.hist(cosmo_sne['HOST_LOGsSFR'], bins = bins, label = 'Observed SNe', ls = '--', histtype='step')

s_logmass = [6.606674394677709, 7.0536352473199555, 6.183551697080299, 7.804529397263261, 7.184743553272747, 7.303933297301057, 7.631704405914994, 8.215733189204231, 8.388557493088582, 8.41835492909566, 8.573301733825245, 8.740167100479313, 8.9904644629965, 8.912991404363662, 8.632897293303317, 8.746126037709596, 8.370679306469901, 8.513706861811091, 8.609058107062605, 8.78784272310507, 8.912991404363662, 9.073897146323532, 9.139451643031887, 8.9904644629965, 8.907033154597297, 8.859356157043708, 8.936829215676541, 9.08581639571193, 9.181168328427361, 9.2407625129776, 9.276519573678875, 9.276519573678875, 9.270559948984678, 9.34207407038723, 9.520857998965779, 9.461263126951623, 9.401669629865303, 9.556615059667056, 9.34207407038723, 9.431465690944547, 9.461263126951623, 9.419547129020065, 9.449344565027143, 9.556615059667056, 9.634088118299891, 9.687723365619846, 9.610250306987012, 9.717520801626925, 9.765196424252682, 9.902264354899671, 9.926102166212551, 9.878426543586794, 10.06912972155374, 10.25387464975432, 10.194279090276249, 10.003575912309303, 10.122764968873696, 10.20619765220073, 10.20619765220073, 10.122764968873696, 10.599523257523021, 10.963051426838234, 10.873659118817002, 10.694874502774535, 10.6233610688359, 10.647198880148778, 10.6233610688359, 10.516090574195987, 10.414779704250272, 10.313468146840643, 10.301549584916163, 11.05244304739555, 11.082240483402627, 10.927294366136959, 10.11680671910733, 10.390942580401312, 10.343265582847721, 10.247914337596205, 10.271752148909085, 10.593563632828822, 10.814064246802847, 10.74255081286421, 10.641239255454579, 10.498212387577308, 9.961859226913827, 9.395709317707187, 9.353992632311712, 9.711561176932726, 9.592372120368331, 10.17044127896337, 10.110846406949214, 9.568533621591536, 9.210965076970522, 10.569725821515943, 10.533968760814666, 10.408820079556074, 10.414779704250272, 10.432657890868953, 10.6233610688359, 10.641239255454579, 10.885579055669316, 10.879618743511198, 10.814064246802847, 10.796186060184166, 11.034564860776872, 10.933253990831156, 10.933253990831156, 11.058402672089748, 11.314660346765049, 11.278903286063773, 11.231227663438014, 11.421930841404963, 11.38021553093732, 11.612634019371908, 11.618593644066106, 11.386173780703686, 11.231227663438014, 11.183552040812257]
s_logsfr = [-3.3746541144800384, -3.2424226149733806, -2.5426985765916332, -1.831954981762445, -1.5399440449893191, -1.4517898179136353, -0.7630847004068024, -0.4104677921040678, -1.049585858956208, -1.104682370048359, -2.2837453558020404, -1.1377407216044173, -1.1212113869332565, -0.9118448990120935, -0.8347099105975881, -0.7685944786305221, -0.7190074279758285, -0.4876027805185741, -0.03581208869271668, -0.30027476991976654, -0.3223135650283848, -0.45454474674877865, -0.18457292287053306, -0.06336034423879155, -0.03581208869271668, 0.23416037075805418, 0.31129535917256046, 0.1680443032184633, 0.1900830983270816, 0.07438061570532284, 0.09641909302767848, 0.09090963259022011, 0.019284422399435286, -0.030302628255258313, 0.8347106256166792, 0.7134990003437247, 0.6418737901529399, 0.6473832505903969, 0.4049590466857018, 0.4269975240080566, 0.31129535917256046, 0.23416037075805418, 0.1680443032184633, 0.24517960941923223, 0.20110233698825963, 0.1460061436823712, 0.12947712679747303, 0.34986253559355074, 0.2176313538731569, 0.5426998477366842, 1.1652892154901897, 1.1212122608454784, 1.2975208738899768, 1.319559351212332, 0.8787882158339149, 0.8236917047417642, 0.7630857332121557, 0.7741049718733333, 0.6969699834588274, 0.6418737901529399, 1.1542701357221432, 1.5730027937782056, 1.3140497318817435, 1.187328010598807, 1.0550966699852822, 0.9614326646858782, 0.8236917047417642, 0.8126724660805866, 0.8126724660805866, 0.8126724660805866, 0.7355374776660804, 0.7079890632268739, 0.34986253559355074, 0.08539985436650044, 0.5316807679686377, 0.48760381332392777, 0.37190117180903703, 0.3388429791461105, 0.25068891096355816, 0.08539985436650044, -0.024792850031538638, -0.09090859978486687, -0.20661108240662607, -0.3498621383607228, -0.19559184374544802, -0.13498555442957683, -0.23966879839015798, -0.37740975833427237, -0.4435255080876006, -0.6749304733311186, -0.7410462230844477, -1.1432498642556115, -0.9118448990120935, -3.0110182853023884, -3.181816961230036, -3.264461727868262, -3.4077121482498334, -3.5234143130853295, -3.264461727868262, -3.4187310691247497, -3.347105858933963, -3.4187310691247497, -3.4848471366643405, -3.5895303806249204, -3.517905170434135, -3.044075683499659, -3.1983456603286706, -3.0716239390457343, -2.950411995986518, -3.0936624163680895, -3.044075683499659, -3.055095239947099, -3.121210671914165, -3.4022023700261137, -3.5399436477564894, -3.6391171134933513, -3.5785108241774797, -3.4077121482498334]

s_logmass = np.array(s_logmass)
s_logsfr = np.array(s_logsfr)

plt.hist(np.log10((10**s_logsfr)/(10**s_logmass)), bins = bins, label = 'Sullivan', histtype='step', ls='--')
#
#plt.yscale('log')
norm1 = np.sum(bin_width * N_SN)
plt.bar(bin_centers, N_SN, width = bin_width, label = 'Corrected SNe', alpha = 0.5, color = 'C2')
#plt.bar(bin_centers, N_SN_sullivan, width = bin_width, label = 'Using a sullivan style correction', alpha = 0.5, color = 'C0')
#print(N_SN/norm0, 'N_SN/norm')
norm2 = np.sum(bin_width * Vmax)
#plt.bar(bin_centers, counts/norm2, width = bin_width, label = 'uncorrected galaxies', alpha = 0.5, color = 'C0')
#
#print(Vmax/norm2, 'Vmax/norm')
#plt.plot(bin_centers, (N_SN/norm1) / (Vmax/norm2), label = 'rate', color = 'C1')
#plt.plot(bin_centers, total_mass, label = 'total mass', color = 'C4')

plt.xlabel('log sSFR')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 3, 2)
plt.yscale('log')
plt.hist(hostlib['LOGsSFR'], bins = bins, label = 'Hostlib', alpha = 0.5)
plt.bar(bin_centers, Vmax, width = bin_width, label = 'Vmax corrected galaxies', alpha = 0.5, color = 'C3')
plt.xlabel('log sSFR')

#mass_bins = np.linspace(8, 12, 10)
#plt.hist(hostlib['LOGMASS'], bins = mass_bins, label = 'Hostlib', histtype='step', ls='--')
#Vmax_mass = scipy.stats.binned_statistic(catalogue['LOGMASS'], Vmax_correction,
#                                    bins=mass_bins, statistic='sum')[0]
#plt.bar((mass_bins[:-1] + mass_bins[1:]) / 2, Vmax_mass, width = mass_bins[1] - mass_bins[0], label = 'Vmax corrected galaxies', alpha = 0.5, color = 'C3')
#plt.xlabel('log Stellar Mass')

plt.ylabel('Density')
plt.legend()

plt.subplot(1, 3, 3)


mean_correction = scipy.stats.binned_statistic(cosmo_sne['HOST_LOGsSFR'], corrections * volume_factor,  bins = bins, statistic = 'mean')[0]
rate_error = np.sqrt(N_SN_no_corr[0]) * (mean_correction / total_mass)  * (1 / Vmax) * (1 / total_time)
xerr = (bins[1] - bins[0]) / 2


plt.errorbar(bin_centers, ((N_SN_per_mass / Vmax) * 1/total_time), yerr =rate_error, xerr = xerr, label = 'Inferred SN Rate')
plt.ylabel(r'Rate (SNe / stellar mass / yr)')
plt.xlabel('log sSFR')
plt.yscale('log')
plt.legend()
plt.show()