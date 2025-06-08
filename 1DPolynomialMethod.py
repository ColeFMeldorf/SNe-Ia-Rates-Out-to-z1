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

real_array = np.array([cosmo_sne['zHD'], hostgals['HOSTGAL_MAG_r'], cosmo_sne['c'] - 0.0, cosmo_fields, cosmo_sne['x1']]).T
print('SUBTRACTING 0.1 FROM C')
print(real_array.shape)

print(cosmo_sne['zHD'].shape)
print('cutting real data to z < %.2f and z > %.2f' % (MAX_Z, MIN_Z))
real_array = real_array[(cosmo_sne['zHD'] < MAX_Z) & (cosmo_sne['zHD'] > MIN_Z)]
real_array.shape

hostgals = hostgals[(cosmo_sne['zHD'] < MAX_Z) & (cosmo_sne['zHD'] > MIN_Z)]
cosmo_sne = cosmo_sne[(cosmo_sne['zHD'] < MAX_Z) & (cosmo_sne['zHD'] > MIN_Z)]
hostlib = hostlib[(hostlib['ZTRUE'] < MAX_Z) & (hostlib['ZTRUE'] > MIN_Z)]


# Getting Field information
fields = np.array(['C1', 'C2', 'C3', 'X1', 'X2', 'X3', 'S1', 'S2', 'E1', 'E2'])
field_ras = np.array([54.2743,  54.2743, 52.6484, 34.4757, 35.6645, 36.4500,42.8200, 41.1944, 7.8744 , 9.5000,])
field_decs = np.array([-27.1116, -29.0884, -28.1000, -4.9295, -6.4121, -4.6000, 0.0000, -0.9884, -43.0096,  -43.9980])  
#c = SkyCoord(ra=dumpdf['RA'].values*u.degree, dec=dumpdf['DEC'].values*u.degree)
#catalog = SkyCoord(ra=field_ras*u.degree, dec=field_decs*u.degree)
#idx, d2d, d3d = c.match_to_catalog_sky(catalog)
#print(idx)
#reduced_dump_fields = np.array([fields[i] for i in idx])
#print(reduced_dump_fields)
#deepfields = np.where((fields == 'X3') | (fields == 'C3'))[0]
#deep_bool = np.zeros_like(idx)
#deep_bool[np.where((idx == deepfields[0]) | (idx == deepfields[1]))[0]] = 1

# c_cuts = [(-0.3, 0.0), (0.0, 0.3)]
# x_cuts = [(-3, 0), (0,3)]

# bins = np.linspace(0, 1.2, 16)



# reduced_sim_fields = np.array([i.split('+')[-1] for i in simdf['FIELD']])
# fields = np.unique(np.array(reduced_sim_fields))

plt.figure(figsize = (5,10))

poly_dict = {}

'''
for i,f in enumerate(fields):
  plt.subplot(5,2,i+1)
  print(f)
  plt.text(0.5, 0.6, f)
  for c_cut in c_cuts:
      for x_cut in x_cuts:
          simdf_cut = simdf.iloc[np.where(
                                    (simdf['SIM_c'] > c_cut[0]) 
                                  & (simdf['SIM_c'] < c_cut[1])
                                  & (simdf['SIM_x1'] > x_cut[0])
                                  & (simdf['SIM_x1'] < x_cut[1])
                                  & (reduced_sim_fields == f))]
          
          dumpdf_cut = dumpdf.iloc[np.where(
                                    (dumpdf['SALT2c'] > c_cut[0]) 
                                  & (dumpdf['SALT2c'] < c_cut[1])
                                  & (dumpdf['SALT2x1'] > x_cut[0])
                                  & (dumpdf['SALT2x1'] < x_cut[1])
                                  & (reduced_dump_fields == f))]
          print(np.size(simdf_cut), np.size(dumpdf_cut))
          dump_counts = np.histogram(dumpdf_cut['GENZ'], bins = bins)[0]
          sim_counts = np.histogram(simdf_cut['zHD'], bins = bins)[0]
          bin_centers = bins[:-1]/2 + bins[1:]/2

          w = sim_counts + dump_counts

          error = np.sqrt(1/sim_counts + 1/dump_counts)

          eff = np.nan_to_num(sim_counts/dump_counts)
          eff[eff>=1 ] = 1

          #plt.errorbar(bin_centers, (sim_counts/dump_counts), yerr = (sim_counts/dump_counts) * error)
          p = np.poly1d(np.polyfit(bin_centers, eff, deg = 4, w = w))

          

          zz = np.linspace(bin_centers[0], bin_centers[-1],100)
          plt.plot(zz, p(zz))
          plt.ylim(0, 1)

          poly_dict[f + '_' + str(c_cut[0]) + '_' + str(x_cut[0])] = p
          print(eff)
          print(p)

plt.show()
'''
poly_dict = np.load('poly_dict.npy', allow_pickle=True).item()
print(poly_dict.keys())

############ Load spec efficiency data ############
file_path = r"C:\Users\cmeldorf\Desktop\SEARCHEFF_zHOST_DES-SN5YR.DAT"
#searcheff  = pd.read_csv(file, delimiter='   ', skiprows = 23)

df = pd.DataFrame(columns=['Year', 'Field', 'r_obs_auto', 'obs_gr_auto', 'HOST_EFF'])

with open(file_path, 'r') as file:
    year = 'test'
    field = 'test'
    for i, line in enumerate(file):
        
        if i < 20:
            continue
        line = line.split()
        #print(line)

        if 'Year' in line: 
            year = line[-1]
            #print(line)
            #print(year)
        if 'FIELDLIST:' in line:
            field = line[-1]
            #print(line)
            #print(field)
        
        if 'HOSTEFF:' in line:
            #print(year, field)
            new_rows = pd.DataFrame({'Year': year, 'Field': field, 'r_obs_auto': float(line[1]), 'obs_gr_auto': float(line[2]), 'HOST_EFF': float(line[3])}, index = [0])
            df = pd.concat([df, new_rows], ignore_index=True)
spec_df =df


############ Calculate corrections ############
corrections = []


for f, c, x1, z, mr in zip(cosmo_sne['FIELD'], cosmo_sne['c'], cosmo_sne['x1'], cosmo_sne['zHD'], hostgals['HOSTGAL_MAG_r']):
    print(f,c,x1,z)
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


N_SN = scipy.stats.binned_statistic(cosmo_sne['HOST_LOGMASS'], correction * volume_factor,  bins = bins, statistic = 'sum')[0] 



plt.plot(bin_centers, ((N_SN / Vmax) * 1/total_time), label = 'Inferred SNe', color = 'C1')

# corrected_N_SN = ((N_SN / Vmax) * 1/total_time) * mass_correction
# plt.plot(bin_centers, corrected_N_SN, label = 'Inferred SNe', color = 'C3')


plt.yscale('log')
#plt.ylim(1e-4, 10**(-1))
x = [8.366934932446702, 8.620967291793225, 8.874999651139749, 9.116935630167202, 9.370968687234226, 9.62499965113975, 9.872984866907489, 10.127015830813011, 10.387096729179323, 10.63508054950606, 10.883064369832798, 11.125000348860251, 11.372984169186989, 11.645161447872372]
y = [-3.5696322907036198, -3.511605428268646, -3.3326885718235055, -3.163442859117528, -2.9361702839765758, -2.7669245712705974, -2.6460348561686864, -2.457446855984384, -2.259187712060919, -2.147968861789915, -1.9690522842530296, -1.891682576523221, -1.9158605753252544, -2.0125725705333872]
x = np.array(x)
y = np.array(y)
y = 10**y

plt.plot(x, y, label = 'Wiseman et. al.', marker = 'o', ms = 1, color = 'r')
plt.legend()
plt.show()