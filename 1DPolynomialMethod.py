#SFR from hostlib
#KDE?
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
MAX_Z = 0.6
MIN_Z = 0.2
max_i = 24.5

LaurenNicePlots()

# Load real data

hostgals = fits.open(r"C:\Users\cmeldorf\Downloads\DES-SN5YR_DES_HEAD.FITS\DES-SN5YR_DES_HEAD.FITS")[1].data
cosmo_sne = pd.read_csv(r"C:\Users\cmeldorf\Downloads\DES-SN5YR_HD+MetaData.csv")[:1635]

hostlib = pd.read_table(r"C:\Users\cmeldorf\Downloads\DES-SN5YR_DES.HOSTLIB\DES-SN5YR_DES.HOSTLIB", comment = '#', skiprows = 57, sep = '\s+')
#hostlib = pd.read_table(r"C:\Users\cmeldorf\Downloads\corrected_deduped_with_magauto_logsfr_upto_4.HOSTLIB+HOSTNBR\corrected_deduped_with_magauto_logsfr_upto_4.HOSTLIB+HOSTNBR", skiprows = 2, comment = '#', sep = '\s+')


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


cut = (cosmo_sne['zHD'] < MAX_Z) & (cosmo_sne['zHD'] > MIN_Z) & (hostgals['HOSTGAL_MAG_r'] < max_i)
hostgals = hostgals[cut]
cosmo_sne = cosmo_sne[cut]
cosmo_fields = cosmo_fields[cut]
hostlib = hostlib[(hostlib['ZTRUE'] < MAX_Z) & (hostlib['ZTRUE'] > MIN_Z) & (hostlib['r_obs_auto'] < max_i)]

print(np.shape(hostlib), 'hostlib shape')
real_array = np.array([cosmo_sne['zHD'], hostgals['HOSTGAL_MAG_r'], cosmo_sne['c'], cosmo_fields, cosmo_sne['x1']]).T
real_array = real_array[(cosmo_sne['zHD'] < MAX_Z) & (cosmo_sne['zHD'] > MIN_Z)]

plt.subplot(2,3,1)
loghist, _, _ =plt.hist(hostlib['LOGMASS'], bins = np.linspace(7, 12, 50), histtype = 'step', label = 'Hostlib Masses', color = 'C0', density = True)
print('Hostlib Masses', np.shape(hostlib['LOGMASS']))
plt.yscale('log')

x = [7.047789818077926, 7.2102748136431805, 7.353644267876959, 7.444444260682686, 7.568697897942348, 7.731182617864964, 7.912783706046972, 8.156511337216173, 8.357228242372296, 8.562724239593267, 8.744324776489998, 8.94026314086655, 9.169653495841207, 9.422939311140103, 9.657108758179609, 9.881720572374695, 10.163679837492126, 10.46953401164852, 10.75149327676595, 10.937873457012806, 11.066905635052041, 11.200716905156122, 11.339307267325053, 11.420549627286361]
y = [1132.540765804203, 2143.292994929668, 3018.074830946204, 4965.307582469544, 6270.435393930947, 10808.95843950722, 16197.933716284108, 19828.839972178033, 21102.029679612348, 21102.029679612348, 18061.62933265445, 14526.544191823114, 12241.57508354335, 9693.640935300697, 7918.600975101953, 6468.607034015782, 5366.975513727014, 4665.721857553848, 3471.6886712951996, 2045.5531516076105, 1031.6050532848835, 405.60982495523245, 149.8566984255766, 44.529645577932406]

w_bin_width = np.mean(np.diff(x))

area = np.sum(np.array(y) * w_bin_width)

plt.scatter(x, np.array(y)/area,  color = 'C1', s = 1, label = 'Wiseman et. al. Hostlib Masses')



plt.legend()
plt.xlim(8,12)

plt.subplot(2,3,4)
from scipy import interpolate
f = interpolate.interp1d(x, y/area, kind='linear', fill_value='extrapolate')
bins = np.linspace(7, 12, 50)
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.plot(bin_centers, loghist - f(bin_centers), label = 'This Work - Wiseman', color = 'C2')
plt.axhline(0, color = 'C1', linestyle = '--', label = '1')

plt.subplot(2,3,2)
bins = np.arange(8, 12, 0.25)
#a, b, _ = plt.hist(hostgals['HOSTGAL_LOGMASS'], density = True, bins = bins)

a,b = np.histogram(hostgals['HOSTGAL_LOGMASS'], density = True, bins = bins)
b_centers = (b[:-1] + b[1:]) / 2
plt.plot(b_centers, a, label = 'Hostgal Masses', color = 'C0')

x = [8.127837336112362, 8.371564967281563, 8.643966599554576, 8.902030955633045, 9.145758586802245, 9.379928585127026, 9.633213849140645, 9.876941480309846, 10.130227295608742, 10.397849284531631, 10.632019282856412, 10.85185200498665, 11.13859036216893, 11.367980717143586, 11.630825267857457]
y = [-0.0016259631403337978, 0.0520325086337578, 0.0650406826850605, 0.10894309425996965, 0.20975609146109098, 0.2520325398956665, 0.31707322258072684, 0.37398380820693883, 0.4422763233869402, 0.5593496084914847, 0.5691057155835301, 0.6357723614089242, 0.31544716565466663, 0.09430895706833332, 0.011382117125242404]
print(np.size(y))
plt.plot(x,y,  color = 'C2', label = 'Wiseman et. al.', marker = 'o')
plt.legend()

plt.subplot(2,3,5)
plt.plot(b_centers, a - y, label = 'This Work - Wiseman', color = 'C0')
plt.axhline(0, color = 'C1', linestyle = '--', label = '1')


# Getting Field information
fields = np.array(['C1', 'C2', 'C3', 'X1', 'X2', 'X3', 'S1', 'S2', 'E1', 'E2'])
field_ras = np.array([54.2743,  54.2743, 52.6484, 34.4757, 35.6645, 36.4500,42.8200, 41.1944, 7.8744 , 9.5000,])
field_decs = np.array([-27.1116, -29.0884, -28.1000, -4.9295, -6.4121, -4.6000, 0.0000, -0.9884, -43.0096,  -43.9980])  


poly_dict = np.load('poly_dict.npy', allow_pickle=True).item()
print(poly_dict.keys())

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


############ Calculate corrections ############
corrections = []


for f, c, x1, z, mr in zip(cosmo_sne['FIELD'], cosmo_sne['c'], cosmo_sne['x1'], cosmo_sne['zHD'], hostgals['HOSTGAL_MAG_r']):
    f_reduced = f.split('+')[-1]
    c_index = -0.3 if c < 0 else 0.0
    x_index = -3 if x1 < 0 else 0
    p = poly_dict[f_reduced + '_' + str(c_index) + '_' + str(x_index)]

    spec_df_cut = spec_df[[(f_reduced in i) for i in spec_df['Field'].values]]
    closest = np.argmin(np.abs(spec_df_cut['r_obs_auto'] - mr))
    host_eff = spec_df_cut['HOST_EFF'].iloc[closest]
    
    if host_eff == 0:
        print('Host Eff is zero, skipping')
        corrections.append(np.nan)
        continue

    correction = 1/(p(z) * host_eff)



    corrections.append(correction)
    
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
print(V_SN)

volume_factor = (V_G/V_SN).astype(float)
print('Volume Factor', volume_factor)
print('Total Time', total_time)


bins = np.arange(8, 12, 0.25)

Vmax, Vmaxbins, Vmax_correction, counts = calc_Vmax(max_z = MAX_Z, max_i = max_i, 
                                            measured_z = hostlib['ZTRUE'],
                                             measured_i = hostlib['i_obs_auto'], 
                                             mass = hostlib['LOGMASS'], 
                                             bins = bins,
                                             min_z = MIN_Z)
#Vmax_correction[Vmax_correction > 1e4] = 1e4  #Cutting out extreme outliers

N_SN_no_corr = np.histogram(cosmo_sne['HOST_LOGMASS'], bins = bins)
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.subplot(2,3,3)
N_SN = scipy.stats.binned_statistic(cosmo_sne['HOST_LOGMASS'], correction * volume_factor,  bins = bins, statistic = 'sum')[0] 

plt.plot(bin_centers[1:], ((N_SN / Vmax) * 1/total_time)[1:], label = 'Inferred SNe', color = 'C1')


plt.yscale('log')
x = [8.366934932446702, 8.620967291793225, 8.874999651139749, 9.116935630167202, 9.370968687234226, 9.62499965113975, 9.872984866907489, 10.127015830813011, 10.387096729179323, 10.63508054950606, 10.883064369832798, 11.125000348860251, 11.372984169186989, 11.645161447872372]
y = [-3.5696322907036198, -3.511605428268646, -3.3326885718235055, -3.163442859117528, -2.9361702839765758, -2.7669245712705974, -2.6460348561686864, -2.457446855984384, -2.259187712060919, -2.147968861789915, -1.9690522842530296, -1.891682576523221, -1.9158605753252544, -2.0125725705333872]
x = np.array(x)
y = np.array(y)
y = 10**y

plt.plot(x, y, label = 'Wiseman et. al.', marker = 'o', ms = 1, color = 'r')
plt.legend()
plt.show()