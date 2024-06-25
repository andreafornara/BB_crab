# %%
from bqht import bqht # https://gitlab.cern.ch/bi/head-tail/bqht
import numpy as np
from matplotlib import pyplot as plt
import scipy.constants as cst
import matplotlib
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from scipy.interpolate import interp1d
import time
import nafflib
ht = bqht.BQHT()
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import pandas as pd
from MD_functions import *

#Calibration factor from au to um
au_to_um = -1000/0.21
def func(x, m, c):
        return m * x + c

# %%
# Decide which beam to study
B1 = True
if(B1):
    slicings = [0, 2000, 2000, 6000]
    compare_file = '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_185547.h5'
    filestocompare = [
        #+170, first set
        '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_185552.h5',
        '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_185621.h5',
        # 0
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_185757.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_185810.h5',
        # -170, first set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_185903.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_185935.h5',
        # +170, second set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_190238.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_190312.h5',
        # -170, second set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_190532.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_190542.h5',
        # +170, third set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_190837.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_190843.h5',
        # -170, third set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_191250.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_191255.h5',
        # Dump
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_191625.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B1/LHC.BQHT.B1_20240513_191633.h5'
    ]
else:
    slicings = [0, 900, 2000, 6000]
    compare_file = '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_185022.h5'
    filestocompare = [
        #+170, first set
        '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_185552.h5',
        '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_185621.h5',
        # 0
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_185757.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_185810.h5',
        # -170, first set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_185903.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_185935.h5',
        # +170, second set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_190238.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_190312.h5',
        # -170, second set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_190532.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_190542.h5',
        # +170, third set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_190837.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_190843.h5',
        # -170, third set
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_191250.h5',
        # '/afs/cern.ch/work/a/afornara/public/HT_DATA_MD_20240513/LHC.BQHT.B2/LHC.BQHT.B2_20240513_191255.h5',
    ]
sigmas, deltas, times = unpack_HT_data(compare_file)

print('--------------------')
for ii in range(0,len(filestocompare)):
    sigmas_temp, deltas_temp, times_temp = unpack_HT_data(filestocompare[ii])
    print('Single file length:',len(sigmas_temp))
    sigmas = np.concatenate((sigmas, sigmas_temp), axis = 0)
    deltas = np.concatenate((deltas, deltas_temp), axis = 0)
    times = np.concatenate((times, times_temp), axis = 0)


start_at = 1
end_at = len(sigmas)
print('The compare file starts at turn: ', start_at)
print('The compare file ends at turn: ', end_at)

#check if in compare_file string there is B1 or B2
B1 = False
if 'B1' in compare_file:
    B1 = True
    print('Studying B1')
else:
    print('Studying B2')

# %%
# Decide which portion of the file to use: if the file starts with no beam start = 910
start_at = 1
end_at = len(sigmas)

sigmas, deltas, times = sigmas[start_at:end_at], deltas[start_at:end_at], times[start_at:end_at]
# Perform the average on the signals before and after the collision and calculate the standard deviation
avg_sigma_0_signal = np.mean(sigmas[slicings[0]:slicings[1]], axis = 0)
avg_delta_0_signal = np.mean(deltas[slicings[0]:slicings[1]], axis = 0)
avg_sigma_1_signal = np.mean(sigmas[slicings[2]:slicings[3]], axis = 0)
avg_delta_1_signal = np.mean(deltas[slicings[2]:slicings[3]], axis = 0)
avg_time_0_signal = np.mean(times[slicings[0]:slicings[1]], axis = 0)
avg_time_1_signal = np.mean(times[slicings[2]:slicings[3]], axis = 0)
avg_normal_delta_0_signal = np.mean(deltas[slicings[0]:slicings[1]]/sigmas[slicings[0]:slicings[1]]*au_to_um, axis = 0)
avg_normal_delta_1_signal = np.mean(deltas[slicings[2]:slicings[3]]/sigmas[slicings[2]:slicings[3]]*au_to_um, axis = 0)
std_normal_delta_0_signal = np.std(deltas[slicings[0]:slicings[1]]/sigmas[slicings[0]:slicings[1]]*au_to_um, axis = 0)/np.sqrt(len(deltas[slicings[0]:slicings[1]]))
std_normal_delta_1_signal = np.std(deltas[slicings[2]:slicings[3]]/sigmas[slicings[2]:slicings[3]]*au_to_um, axis = 0)/np.sqrt(len(deltas[slicings[2]:slicings[3]]))
std_total_signal = np.sqrt(std_normal_delta_0_signal**2+std_normal_delta_1_signal**2)
diff_avg_normal_delta_signal = avg_normal_delta_1_signal-avg_normal_delta_0_signal

# We now find the gaussians in the signals for the synchronization via maximization of the product of the gaussians
sigma_gaussians, sigma_parameters, all_maxes, all_maxes_from_fit, avg_standard_deviation = find_gaussians(sigmas, deltas, times)
# all_shifted_sigma, all_shifted_delta, all_shifted_times are the interpolated signals shifted to the optimal shift
all_shifted_sigma, all_shifted_delta, all_shifted_times, optimal_shift = align_via_gaussians(sigmas, deltas, times, sigma_parameters, B1 = True)
# Perform the average on the shifted signals and calculate the standard deviation

avg_sigma_0 = np.mean(all_shifted_sigma[slicings[0]:slicings[1]], axis = 0)
avg_delta_0 = np.mean(all_shifted_delta[slicings[0]:slicings[1]], axis = 0)
avg_normal_delta_0 = avg_delta_0/avg_sigma_0*au_to_um
avg_time_0 = np.mean(all_shifted_times[slicings[0]:slicings[1]], axis = 0)
std_normal_delta_0 = np.std(all_shifted_delta[slicings[0]:slicings[1]]/all_shifted_sigma[slicings[0]:slicings[1]]*au_to_um, axis = 0)
avg_sigma_1 = np.mean(all_shifted_sigma[slicings[2]:slicings[3]], axis = 0)
avg_delta_1 = np.mean(all_shifted_delta[slicings[2]:slicings[3]], axis = 0)
avg_normal_delta_1 = avg_delta_1/avg_sigma_1*au_to_um
avg_time_1 = np.mean(all_shifted_times[slicings[2]:slicings[3]], axis = 0)
std_normal_delta_1 = np.std(all_shifted_delta[slicings[2]:slicings[3]]/all_shifted_sigma[slicings[2]:slicings[3]]*au_to_um, axis = 0)
std_total = np.sqrt(std_normal_delta_0**2/len(all_shifted_delta[slicings[0]:slicings[1]])+std_normal_delta_1**2/len(all_shifted_delta[slicings[2]:slicings[3]]))
# Interpolate the signals and perform the difference on the interpolated signals
tts = np.linspace(np.min(avg_time_0), np.max(avg_time_0), len(avg_normal_delta_0))
interp_avg_normal_delta_0 = interp1d(tts, avg_normal_delta_0, kind = 'linear')
interp_avg_normal_delta_1 = interp1d(tts, avg_normal_delta_1, kind = 'linear')
diff_avg_normal_delta = (interp_avg_normal_delta_1(avg_time_0)-interp_avg_normal_delta_0(avg_time_0))

# %%
# Now plotting the difference signal and fitting it with a line to retrieve crabbing
fig, ax = plt.subplots(2, 1, figsize = (10, 10))

#find the peak in the avg_sigma_0_signal
peak = np.argmax(avg_sigma_0_signal)

#Now only fit the points around the peak
start = 5
end = 5

popt, pcov = curve_fit(func, avg_time_0_signal[peak-start:peak+end], diff_avg_normal_delta_signal[peak-start:peak+end], sigma = std_total_signal[peak-start:peak+end])
ax[0].errorbar(avg_time_0_signal[peak-start:peak+end], diff_avg_normal_delta_signal[peak-start:peak+end], yerr = std_total_signal[peak-start:peak+end], fmt = 'o', color = 'black', label = 'Difference Signal')
ax[0].plot(avg_time_0_signal[peak-start:peak+end], func(avg_time_0_signal[peak-start:peak+end], *popt), color = 'green', label = f'Fit, crabbing = {-popt[0]*0.33:.0f}'+r'$\mu$m')
ax[0].plot(avg_time_0_signal[peak-start:peak+end], avg_sigma_0_signal[peak-start:peak+end]*100-50, color = 'red', label = 'Sum Signal as reference')
# ax.errorbar(avg_time_0_signal[], diff_avg_normal_delta_signal, yerr = std_total_signal, fmt = 'o', color = 'black', label = 'Difference Signal')
# ax.plot(avg_time_0_signal, func(avg_time_0_signal, *popt), color = 'red', label = f'Fit, slope = {popt[0]*0.33:.2f}')
ax[0].legend()
ax[0].grid()

ax[0].set_ylabel(r' $\Delta\frac{\Delta(z)}{\Sigma}$ [um]', fontsize = 10, labelpad = 20, rotation = 0)
ax[0].set_xlabel('Time [ns]')
ax[0].set_title('Real signals fit')

# Same thing for the interpolated signals
peak = np.argmax(avg_sigma_0)
start = 3
end = 4
popt, pcov = curve_fit(func, avg_time_0[peak-start:peak+end], diff_avg_normal_delta[peak-start:peak+end], sigma = std_total[peak-start:peak+end])
ax[1].errorbar(avg_time_0[peak-start:peak+end], diff_avg_normal_delta[peak-start:peak+end], yerr = std_total[peak-start:peak+end], fmt = 'o', color = 'black', label = 'Difference Signal')
ax[1].plot(avg_time_0[peak-start:peak+end], func(avg_time_0[peak-start:peak+end], *popt), color = 'green', label = f'Fit, crabbing = {-popt[0]*0.33:.0f}'+r'$\mu$m')
ax[1].plot(avg_time_0[peak-start:peak+end], avg_sigma_0[peak-start:peak+end]*100-50, color = 'red', label = 'Sum Signal as reference')
ax[1].legend()
ax[1].grid()
ax[1].set_xlabel('Time [ns]')
ax[1].set_ylabel(r' $\Delta\frac{\Delta(z)}{\Sigma}$ [um]', fontsize = 10, labelpad = 20, rotation = 0)
ax[1].set_title('Interpolated signals fit')
# %%
