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

from MD_functions import *

#Calibration factor from au to um
au_to_um = -1000/0.21
def func(x, m, c):
        return m * x + c

# %%
compare_file = '/eos/user/a/afornara/HT_DATA_2024_05_13_MD/LHC.BQHT.B1/LHC.BQHT.B1_20240513_184534.h5'
# sigmas, deltas, times are the original signals
sigmas, deltas, times = unpack_HT_data(compare_file)
# Decide which portion of the file to use: if the file starts with no beam start = 910
start_at = 1
end_at = len(sigmas)
sigmas, deltas, times = sigmas[start_at:end_at], deltas[start_at:end_at], times[start_at:end_at]

# We find the averages of the signals before and after the collision and calculate the standard deviation
# 0:900 before collision, 1200:2000 after collision

avg_sigma_0_signal = np.mean(sigmas[0:900], axis = 0)
avg_delta_0_signal = np.mean(deltas[0:900], axis = 0)
avg_sigma_1_signal = np.mean(sigmas[1200:2000], axis = 0)
avg_delta_1_signal = np.mean(deltas[1200:2000], axis = 0)
avg_time_0_signal = np.mean(times[0:900], axis = 0)
avg_time_1_signal = np.mean(times[1200:2000], axis = 0)
avg_normal_delta_0_signal = np.mean(deltas[0:900]/sigmas[0:900]*au_to_um, axis = 0)
avg_normal_delta_1_signal = np.mean(deltas[1200:2000]/sigmas[1200:2000]*au_to_um, axis = 0)
std_normal_delta_0_signal = np.std(deltas[0:900]/sigmas[0:900]*au_to_um, axis = 0)/np.sqrt(len(deltas[0:900]))
std_normal_delta_1_signal = np.std(deltas[1200:2000]/sigmas[1200:2000]*au_to_um, axis = 0)/np.sqrt(len(deltas[1200:2000]))
std_total_signal = np.sqrt(std_normal_delta_0_signal**2+std_normal_delta_1_signal**2)
diff_avg_normal_delta_signal = avg_normal_delta_1_signal-avg_normal_delta_0_signal

# We now find the gaussians in the signals for the synchronization via maximization of the product of the gaussians
sigma_gaussians, sigma_parameters, all_maxes, all_maxes_from_fit, avg_standard_deviation = find_gaussians(sigmas, deltas, times)
# all_shifted_sigma, all_shifted_delta, all_shifted_times are the interpolated signals shifted to the optimal shift
all_shifted_sigma, all_shifted_delta, all_shifted_times, optimal_shift = align_via_gaussians(sigmas, deltas, times, sigma_parameters, B1 = True)
# We do the same for the interpolated signals
avg_sigma_0 = np.mean(all_shifted_sigma[0:900], axis = 0)
avg_delta_0 = np.mean(all_shifted_delta[0:900], axis = 0)
avg_normal_delta_0 = avg_delta_0/avg_sigma_0*au_to_um
avg_time_0 = np.mean(all_shifted_times[0:900], axis = 0)
std_normal_delta_0 = np.std(all_shifted_delta[0:900]/all_shifted_sigma[0:900]*au_to_um, axis = 0)
avg_sigma_1 = np.mean(all_shifted_sigma[1200:2000], axis = 0)
avg_delta_1 = np.mean(all_shifted_delta[1200:2000], axis = 0)
avg_normal_delta_1 = avg_delta_1/avg_sigma_1*au_to_um
avg_time_1 = np.mean(all_shifted_times[1200:2000], axis = 0)
std_normal_delta_1 = np.std(all_shifted_delta[1200:2000]/all_shifted_sigma[1200:2000]*au_to_um, axis = 0)
std_total = np.sqrt(std_normal_delta_0**2/len(all_shifted_delta[0:900])+std_normal_delta_1**2/len(all_shifted_delta[1200:2000]))
# Interpolate the signals and perform the difference on the interpolated signals
tts = np.linspace(np.min(avg_time_0), np.max(avg_time_0), len(avg_normal_delta_0))
interp_avg_normal_delta_0 = interp1d(tts, avg_normal_delta_0, kind = 'linear')
interp_avg_normal_delta_1 = interp1d(tts, avg_normal_delta_1, kind = 'linear')
diff_avg_normal_delta = (interp_avg_normal_delta_1(avg_time_0)-interp_avg_normal_delta_0(avg_time_0))

# %%
# Plotting both the real and the interpolated signals
fig, ax = plt.subplots(2, 1, figsize = (10, 15))
# Plotting just a few signals to see the difference
for ii in range(0, 100):
    ax[0].plot(times[ii], sigmas[ii]*150, 'o', color = 'red', alpha = 0.1)
    ax[0].plot(times[ii+1200], sigmas[ii+1200]*150, 'o', color = 'red', alpha = 0.1)

    ax[0].plot(times[ii], deltas[ii]/sigmas[ii]*au_to_um, 'o', color = 'green', alpha = 0.1)
    ax[0].plot(times[1200+ii], deltas[1200+ii]/sigmas[1200+ii]*au_to_um, 'o', color = 'blue', alpha = 0.1)
ax[0].plot(times[0], sigmas[0]*150, 'o', color = 'red', alpha = 0.1, label = 'Sum Signal')
ax[0].plot(times[0], deltas[0]/sigmas[0]*au_to_um, 'o', color = 'green', alpha = 0.1, label = 'Difference Signal Before Collision')
ax[0].plot(times[1200], deltas[1200]/sigmas[1200]*au_to_um, 'o', color = 'blue', alpha = 0.1, label = 'Difference Signal After Collision')
ax[0].grid()
ax[0].set_xlabel('Time [ns]')
ax[0].set_ylabel('Signal [a.u.]')
ax[0].legend()
ax[0].set_ylim(-500, 300)
ax[0].set_xlim(-0.8,0.8)
ax[0].set_title('Real signals')

for ii in range(0, 100):
    ax[1].plot(all_shifted_times[ii], all_shifted_sigma[ii]*150, 'o', color = 'red', alpha = 0.1)
    ax[1].plot(all_shifted_times[ii+1200], all_shifted_sigma[ii+1200]*150, 'o', color = 'red', alpha = 0.1)

    ax[1].plot(all_shifted_times[ii], all_shifted_delta[ii]/all_shifted_sigma[ii]*au_to_um, 'o', color = 'green', alpha = 0.1)
    ax[1].plot(all_shifted_times[1200+ii], all_shifted_delta[1200+ii]/all_shifted_sigma[1200+ii]*au_to_um, 'o', color = 'blue', alpha = 0.1)
ax[1].plot(all_shifted_times[0], all_shifted_sigma[0]*150, 'o', color = 'red', alpha = 0.1, label = 'Sum Signal')
ax[1].plot(all_shifted_times[0], all_shifted_delta[0]/all_shifted_sigma[0]*au_to_um, 'o', color = 'green', alpha = 0.1, label = 'Difference Signal Before Collision')
ax[1].plot(all_shifted_times[1200], all_shifted_delta[1200]/all_shifted_sigma[1200]*au_to_um, 'o', color = 'blue', alpha = 0.1, label = 'Difference Signal After Collision')
ax[1].grid()
ax[1].set_xlabel('Time [ns]')
ax[1].set_ylabel('Signal [a.u.]')
ax[1].legend()
ax[1].set_ylim(-300, 200)
ax[1].set_xlim(-0.8,0.8)
ax[1].set_title('Interpolated signals')


# %%
# Now plotting the difference signal and fitting it with a line to retrieve crabbing
fig, ax = plt.subplots(2, 1, figsize = (10, 10))

#find the peak in the avg_sigma_0_signal
peak = np.argmax(avg_sigma_0_signal)

#Now only fit the points around the peak
start = 3
end = 4

popt, pcov = curve_fit(func, avg_time_0_signal[peak-start:peak+end], diff_avg_normal_delta_signal[peak-start:peak+end], sigma = std_total_signal[peak-start:peak+end])
ax[0].errorbar(avg_time_0_signal[peak-start:peak+end], diff_avg_normal_delta_signal[peak-start:peak+end], yerr = std_total_signal[peak-start:peak+end], fmt = 'o', color = 'black', label = 'Difference Signal')
ax[0].plot(avg_time_0_signal[peak-start:peak+end], func(avg_time_0_signal[peak-start:peak+end], *popt), color = 'green', label = f'Fit, crabbing = {-popt[0]*0.33:.0f}'+r'$\mu$m')
ax[0].plot(avg_time_0_signal[peak-start:peak+end], avg_sigma_0_signal[peak-start:peak+end]*500-500, color = 'red', label = 'Sum Signal as reference')
# ax.errorbar(avg_time_0_signal[], diff_avg_normal_delta_signal, yerr = std_total_signal, fmt = 'o', color = 'black', label = 'Difference Signal')
# ax.plot(avg_time_0_signal, func(avg_time_0_signal, *popt), color = 'red', label = f'Fit, slope = {popt[0]*0.33:.2f}')
ax[0].legend()
ax[0].grid()
ax[0].set_title('Real signals fit')
ax[0].set_xlabel('Time [ns]')
ax[0].set_ylabel(r' $\Delta\frac{\Delta(z)}{\Sigma}$ [um]', fontsize = 10, labelpad = 20, rotation = 0)

# Same thing for the interpolated signals
peak = np.argmax(avg_sigma_0)
start = 3
end = 4
popt, pcov = curve_fit(func, avg_time_0[peak-start:peak+end], diff_avg_normal_delta[peak-start:peak+end], sigma = std_total[peak-start:peak+end])
ax[1].errorbar(avg_time_0[peak-start:peak+end], diff_avg_normal_delta[peak-start:peak+end], yerr = std_total[peak-start:peak+end], fmt = 'o', color = 'black', label = 'Difference Signal')
ax[1].plot(avg_time_0[peak-start:peak+end], func(avg_time_0[peak-start:peak+end], *popt), color = 'green', label = f'Fit, crabbing = {-popt[0]*0.33:.0f}'+r'$\mu$m')
ax[1].plot(avg_time_0[peak-start:peak+end], avg_sigma_0[peak-start:peak+end]*500-500, color = 'red', label = 'Sum Signal as reference')
ax[1].legend()
ax[1].grid()
ax[1].set_title('Interpolated signals fit')
ax[1].set_xlabel('Time [ns]')
ax[1].set_ylabel(r' $\Delta\frac{\Delta(z)}{\Sigma}$ [um]', fontsize = 10, labelpad = 20, rotation = 0)

# %%
