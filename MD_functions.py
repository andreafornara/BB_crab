# %%
import sys
sys.path.append('/afs/cern.ch/work/a/afornara/public/Head_Tail_LHC')
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
from scipy import signal
from scipy.optimize import curve_fit
from scipy.optimize import minimize

ht = bqht.BQHT()

# %%
#Create a function that interpolates the signal and returns the interpolated signal
def interpolate_signal(signal, time):
    interpolator = interp1d(time,signal, kind = 'linear')
    times_interpolated = np.linspace(-9,14,2000)
    signal_interpolated= interpolator(times_interpolated)
    return signal_interpolated, times_interpolated, interpolator

def find_time_shift_via_roll(time, interpolator, interpolator_0):
    signal_0 = interpolator_0(time)
    products = []
    rollers = np.linspace(-200,200,401)
    for jj in range(len(rollers)):
        signal = interpolator(np.roll(time,int(rollers[jj])))
        product = signal_0*signal
        products.append(np.sum(product))
    # print(products)
    index = np.argmax(np.array(products))
    return index

def unpack_HT_data(filename, total_number_of_turns = 2000, first_bunch = 0, last_bunch = 0):
    print('Unpacking data from file: ', filename)
    htfile = ht.open_file(filename)
    htfile.optimise_overlap()
    sigmas = []
    deltas = []
    times = []
    for i in range(total_number_of_turns):
        x, y = htfile.vertical.sigma[i, first_bunch:last_bunch]
        sigma = y
        sigmas.append(sigma)
        x, y = htfile.vertical.delta[i, first_bunch:last_bunch]
        delta = y
        deltas.append(delta)
        mid_index = np.argmax(sigma)
        mid_time = x[mid_index]
        time_ns = (x-mid_time)*1E9
        # time_ns = x*1E9
        times.append(time_ns)
    sigmas = np.array(sigmas)
    deltas = np.array(deltas)
    times = np.array(times)
    # print('Using new time definition')
    return sigmas, deltas, times

def interpolate_all_signals(sigmas,deltas,times,start_at = 0, end_at = 2000, sigmas_0 = 0, deltas_0 = 0, times_0 = 0, first_signal = True):
    interpolator_sigmas = []
    interpolator_deltas = []
    interpolator_times = []
    for i in range(start_at, end_at):
        if(i == start_at and not(first_signal)):
            print('Using sigmas_0 and deltas_0 as reference')
            signal_interpolated_0, times_interpolated_0, sigma_interpolator_0 = interpolate_signal(sigmas_0,times_0)
            delta_interpolated_0, times_interpolated_0, delta_interpolator_0 = interpolate_signal(deltas_0,times_0)
            index_0 = find_time_shift_via_roll(times_interpolated_0, sigma_interpolator_0, sigma_interpolator_0)
        if(i == start_at and first_signal):
            print('Using first signal as reference')
            signal_interpolated_0, times_interpolated_0, sigma_interpolator_0 = interpolate_signal(sigmas[i],times[i])
            delta_interpolated_0, times_interpolated_0, delta_interpolator_0 = interpolate_signal(deltas[i],times[i])
            index_0 = find_time_shift_via_roll(times_interpolated_0, sigma_interpolator_0, sigma_interpolator_0)
        
        signal_interpolated, times_interpolated, sigma_interpolator = interpolate_signal(sigmas[i],times[i])
        delta_interpolated, times_interpolated, delta_interpolator = interpolate_signal(deltas[i],times[i])
        index = find_time_shift_via_roll(times_interpolated, sigma_interpolator, sigma_interpolator_0) - index_0
        interpolator_sigmas.append(sigma_interpolator(np.roll(times_interpolated_0, index)))
        interpolator_deltas.append(delta_interpolator(np.roll(times_interpolated_0, index)))
        interpolator_times.append(times_interpolated)
        if(i % 100 == 0):
            print('Interpolating signal number: ', i)
    interpolator_sigmas = np.array(interpolator_sigmas)
    interpolator_deltas = np.array(interpolator_deltas)
    interpolator_times = np.array(interpolator_times)
    return interpolator_sigmas, interpolator_deltas, interpolator_times

def create_average_signal_0(sigmas,deltas, start_at = 0, end_at = -1):
    sigmas_0 = np.mean(sigmas[start_at:end_at],axis = 0)
    deltas_0 = np.mean(deltas[start_at:end_at],axis = 0)
    return sigmas_0, deltas_0

def find_start(sigmas):
    for i in range(len(sigmas)):
        if np.max(sigmas[i]) > 1E-1:
            return i+1

def moving_average(data, window_size):
    """
    Calculate the moving average of a 1D array.
    Parameters:
        data (array_like): Input data array.
        window_size (int): Size of the moving window.
    Returns:
        array_like: Moving average of the input data.
    """
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gaussian(sigmas, times):
    popt, pcov = curve_fit(gaussian, times, sigmas, p0 = [1, 0, 1])
    return gaussian(times, *popt), popt

def fit_all_gaussians(sigmas, times):
    all_gaussians = []
    all_parameters = []
    for ii in range(len(sigmas)):
        gaussian_fit, parameters = fit_gaussian(sigmas[ii], times[ii])
        all_gaussians.append(gaussian_fit)
        all_parameters.append(parameters)
    return np.array(all_gaussians), np.array(all_parameters)

def find_gaussians(sigmas, deltas, times):
    sigma_gaussians = []
    sigma_parameters = []
    all_maxes = []
    all_maxes_from_fit = []
    avg_standard_deviation = 0
    for ii in range(len(sigmas)):
        cut = np.where((times[ii] > -2.0) & (times[ii] < 2.0))
        gaussian_fit, parameters = fit_gaussian(sigmas[ii][cut], times[ii][cut])
        #find the maximum in sigmas[ii][cut]
        all_maxes.append(np.max(sigmas[ii][cut]))
        all_maxes_from_fit.append(np.max(gaussian_fit))
        curr_times = np.linspace(-2.0, 2.0, len(sigmas[ii][cut]))
        sigma_gaussians.append(gaussian(curr_times, *parameters))
        avg_standard_deviation += parameters[2]
        sigma_parameters.append(parameters)
    return sigma_gaussians, sigma_parameters, all_maxes, all_maxes_from_fit, avg_standard_deviation


def align_via_gaussians(sigmas, deltas, times, sigma_parameters, align_with_other = False, other_signal = 0, B1 = True):
    if(B1):
        t_shift_sigma_delta = 0.12
    else:
        t_shift_sigma_delta = 0.07
    all_shifted_sigma = []
    all_shifted_delta = []
    all_shifted_times = []
    optimal_shifts = []
    print('Warning: using fixed length of 30 for the interpolation.')
    for ii in range(len(sigmas)):
        cut = np.where((times[ii] > -1.5) & (times[ii] < 1.5))
        curr_times = np.linspace(-1.52, 1.52, 30)
        # curr_times = np.linspace(-1.5, 1.5, len(sigmas[ii]))
        if(align_with_other == False):
            sigma_0 = gaussian(curr_times, *sigma_parameters[0])
        else:
            sigma_0 = other_signal
        sigma_ii = gaussian(curr_times, *sigma_parameters[ii])
        def objective_function(shift):
            """Objective function to maximize the product of the two shifted arrays."""
            shifted_times_ii = curr_times + shift
            shifted_sigma_ii = gaussian(shifted_times_ii, *sigma_parameters[ii])
            product = np.sum(sigma_0 * shifted_sigma_ii)  # Calculate product
            return -product  # Minimize the negative product for maximization
        initial_guess = 0.1
        result = minimize(objective_function, initial_guess, method='Powell', bounds=[(-0.2, 0.2)])
        optimal_shift = result.x[0]
        shifted_times_ii = curr_times + optimal_shift
        interp_delta_ii = interp1d(times[ii]-optimal_shift+t_shift_sigma_delta, deltas[ii], kind = 'linear')
        interp_sigma_ii = interp1d(times[ii]-optimal_shift, sigmas[ii], kind = 'linear')
        shifted_sigma_ii = interp_sigma_ii(shifted_times_ii)
        shifted_delta_ii = interp_delta_ii(shifted_times_ii)
        all_shifted_sigma.append(shifted_sigma_ii)
        all_shifted_delta.append(shifted_delta_ii)
        all_shifted_times.append(shifted_times_ii)
        optimal_shifts.append(optimal_shift)
    # print(np.shape(all_shifted_sigma), np.shape(all_shifted_delta), np.shape(all_shifted_times), np.shape(optimal_shifts))
    all_shifted_sigma = np.array(all_shifted_sigma)
    all_shifted_delta = np.array(all_shifted_delta)
    all_shifted_times = np.array(all_shifted_times)
    optimal_shift = np.array(optimal_shift)
    return all_shifted_sigma, all_shifted_delta, all_shifted_times, optimal_shifts