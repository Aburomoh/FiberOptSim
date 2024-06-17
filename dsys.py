import cmath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *  # Generally, avoid using wildcard imports if possible
from numpy.fft import fft, ifft, fftshift, ifftshift
import scipy.integrate as integrate
from scipy import special as sp
from scipy.linalg import dft
import tensorflow as tf
from tensorflow import keras as K

def source(L):
    """
    Generates a stream of binary bits of length L as a numpy array.
    
    Parameters:
        L (int): The length of the bit array to generate.
    
    Returns:
        numpy.ndarray: An array of binary bits.
    """
    return np.random.randint(2, size=L)

def cos_m(theta, eps=1e-8):
    """
    Computes the cosine of theta, but replaces values close to zero with a small epsilon.
    
    Parameters:
        theta (float or array-like): The angle(s) in radians.
        eps (float): The tolerance threshold to replace zeros.
    
    Returns:
        float or numpy.ndarray: The cosine of theta or array of cosine values.
    """
    result = np.cos(theta)
    if abs(result) < eps:
        result = eps
    return result

def read_signal_files(filepaths):
    """
    Helper function to read data from files and convert to numpy arrays.
    """
    data = []
    with open(filepaths, 'r') as file:
        lines = file.readlines()
        data = np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])
    return data

def dataset_read(signalBeforeHardDecision_XI, signalBeforeHardDecision_XQ,
                 signalBeforeHardDecision_YI, signalBeforeHardDecision_YQ,
                 signalBeforeHardDecision_Ref_XI, signalBeforeHardDecision_Ref_XQ,
                 signalBeforeHardDecision_Ref_YI, signalBeforeHardDiscovery_Decision_Ref_YQ):
    """
    Reads and processes signal trace data from transmitter and receiver sides.

    Parameters:
        signalBeforeHardDecision_XI, ...: file paths to the input signal files.
        signalBeforeHardDecision_Ref_XI, ...: file paths to the reference signal files.

    Returns:
        Tuple of arrays: Processed signals (c_X_cap, c_Y_cap, c_X_SEND, c_Y_SEND).
    """
    # Read and process transmitter signals
    num_XI_CAP = read_signal_files(signalBeforeHardDecision_XI)
    num_XQ_CAP = read_signal_iles(signalBeforeHardDecision_XQ)
    num_YI_CAP = read_signal_files(signalBeforeHardDecision_YI)
    num_YQ_CAP = read_signal_files(signalBeforeHardDecision_YQ)

    # Read and process reference signals
    num_XI_SEND = read_signal_files(signalBeforeHardDecision_Ref_XI)
    num_XQ_SEND = read_signal_files(signalBeforeHardDecision_Ref_XQ)
    num_YI_SEND = read_signal_files(signalBeforeHardDecision_Ref_YI)
    num_YQ_SEND = read_signal_files(signalBeforeHardDecision_Ref_YQ)

    # Calculate complex values for captured and sent signals
    c_X_cap = num_XI_CAP + 1j * num_XQ_CAP
    c_Y_cap = num_YI_CAP + 1j * num_YQ_CAP
    c_X_SEND = num_XI_SEND + 1j * num_XQ_SEND
    c_Y_SEND = num_YI_SEND + 1j * num_YQ_SEND

    # Normalize captured signals based on sent signals
    c_X_cap *= np.sqrt(np.mean(np.abs(c_X_SEND)**2)) / np.sqrt(np.mean(np.abs(c_X_cap)**2))
    c_Y_cap *= np.sqrt(np.mean(np.abs(c_Y_SEND)**2)) / np.sqrt(np.mean(np.abs(c_Y_cap)**2))

    return c_X_cap, c_Y_cap, c_X_SEND, c_Y_SEND

def sin_m(theta, eps=1e-8):
    """
    Computes the sine of theta, but replaces values close to zero with a small epsilon.
    
    Parameters:
        theta (float or array-like): The angle(s) in radians.
        eps (float): The tolerance threshold to replace zeros.
    
    Returns:
        float or numpy.ndarray: The sine of theta or array of sine values.
    """
    result = np.sin(theta)
    if abs(result) < eps:
        result = eps
    return result

def QAM16(data, p):
    """
    Generates QAM-16 symbols from input binary data.

    Parameters:
        data (numpy array of int): Input binary data array.
        p (float): Power scaling factor for the constellation.

    Returns:
        numpy.ndarray: Array of QAM-16 symbols.
    """
    lim1 = len(data) // 4
    sym = np.zeros(lim1, dtype=complex)
    m = np.array([8, 4, 2, 1])

    # Mapping from decimal values to QAM-16 constellation points
    mapping = np.array([
        -3-3j, -3-1j, -3+3j, -3+1j,
        -1-3j, -1-1j, -1+3j, -1+1j,
        3-3j, 3-1j, 3+3j, 3+1j,
        1-3j, 1-1j, 1+3j, 1+1j
    ])

    for i in range(lim1):
        part = data[4*i:4*i+4]
        dec_val = np.dot(part, m)
        sym[i] = mapping[dec_val]

    # Apply power scaling factor
    sym = np.sqrt(p) * (np.sqrt(16) / np.sqrt(160)) * sym

    return sym

def QAM64(data, p):
    """
    Generates QAM-64 symbols from input binary data using NumPy for optimized computations.

    Parameters:
        data (numpy array of int): Input binary data array.
        p (float): Power scaling factor for the constellation.

    Returns:
        numpy.ndarray: Array of QAM-64 symbols.
    """
    lim1 = len(data) // 6
    sym = np.zeros(lim1, dtype=complex)
    m = np.array([32, 16, 8, 4, 2, 1])

    # Define real and imaginary parts mappings based on decimal value ranges
    real_map = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
    imag_map = np.array([-7j, -5j, -1j, -3j, 7j, 5j, 1j, 3j])

    for i in range(lim1):
        part = data[6*i:6*i+6]
        dec_val = np.dot(part, m)
        real_idx = dec_val // 8  # Real part index
        imag_idx = dec_val % 8   # Imaginary part index

        sym[i] = real_map[real_idx] + imag_map[imag_idx]

    # Apply power scaling factor
    sym *= np.sqrt(p) / np.sqrt(42)

    return sym


def deQAM16(rec_symbs, p):
    """
    Demodulates received symbols for 16-QAM based on the minimum distance.

    Parameters:
        rec_symbs (numpy array): Received symbols to be demodulated.
        p (float): Power scaling factor used in the constellation.

    Returns:
        numpy.ndarray: Demodulated symbols.
    """
    # Define the constellation points for 16-QAM
    scale = np.sqrt(p) / np.sqrt(10)
    real_parts = np.array([-3, -1, 1, 3])
    imag_parts = real_parts * 1j
    const = (real_parts[:, np.newaxis] + imag_parts).flatten() * scale

    # Calculate distances from each received symbol to each constellation point
    distances = np.abs(rec_symbs[:, np.newaxis] - const)

    # Find the constellation points with the minimum distance to each received symbol
    min_indices = np.argmin(distances, axis=1)
    demodulated_symbols = const[min_indices]

    return demodulated_symbols

def deQAM64(rec_symbs, p):
    """
    Demodulates received symbols for 64-QAM based on the minimum distance.

    Parameters:
        rec_symbs (numpy array): Received symbols to be demodulated.
        p (float): Power scaling factor used in the constellation.

    Returns:
        numpy.ndarray: Demodulated symbols.
    """
    # Define the constellation points for 64-QAM
    scale = np.sqrt(p) / np.sqrt(42)
    real_parts = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    imag_parts = real_parts * 1j
    const = (real_parts[:, np.newaxis] + imag_parts).flatten() * scale

    # Calculate distances from each received symbol to each constellation point
    distances = np.abs(rec_symbs[:, np.newaxis] - const)

    # Find the constellation points with the minimum distance to each received symbol
    min_indices = np.argmin(distances, axis=1)
    demodulated_symbols = const[min_indices]

    return demodulated_symbols

def mod(t, s, B, rof=0.0625):
    """
    Modulates a given array of symbols using a sinc pulse shaping filter.

    Parameters:
        t (numpy array): Time axis.
        s (numpy array of complex): Array of symbols to be modulated.
        B (float): Bandwidth.
        rof (float): Roll-off factor for the sinc function.

    Returns:
        numpy.ndarray: Modulated signal.
    """
    dt = t[1] - t[0]
    Nfft = len(t)
    sampling = len(t) // len(s)
    Ns = len(s)
    
    q = np.zeros(Nfft, dtype=complex)
    midpoint = Ns // 2
    for l in range(-midpoint, midpoint):
        indx = l + midpoint
        lb = max(0, sampling * (indx - 20))
        ub = min(sampling * (indx + 20), Nfft)
        t_index = slice(lb, ub)
        q[t_index] += s[indx] * rcosine(t[t_index] - l / B, B, rof)

    return q

def rcosine(t, B, rof):
    """
    Computes the raised cosine filter values using the sinc function.

    Parameters:
        t (numpy array): Time values where the filter is evaluated.
        B (float): Bandwidth.
        rof (float): Roll-off factor.

    Returns:
        numpy.ndarray: Values of the raised cosine filter.
    """
    eps = 1e-8  # Small threshold to avoid division by zero
    num = np.sinc(t * B) * np.cos(np.pi * rof * (t * B))  # Numerator of the rcosine function
    den = 1 - (2 * rof * t * B) ** 2  # Denominator of the rcosine function

    # Identify indices where the denominator is too close to zero
    bad_ind = np.argwhere(abs(den) < eps)

    # Safety check for boundary case
    # Avoid index error if the last element is a bad index
    if bad_ind.size > 0 and bad_ind[-1] == len(den) - 1:
        bad_ind = bad_ind[:-1]

    # Replace problematic values by borrowing from the next time step
    den[bad_ind] = den[bad_ind + 1]

    return num / den

def phase_noise(N_sym, linewidth, T_sym, samp_rate):
    """
    Generates cumulative phase noise for a given number of symbols.

    Parameters:
        N_sym (int): Number of symbols.
        linewidth (float): Linewidth of the noise.
        T_sym (float): Symbol time.
        samp_rate (int): Sample rate.

    Returns:
        numpy.ndarray: Cumulative phase noise.
    """
    noise_var = 2 * np.pi * linewidth * T6 * samp_rate
    noise = np.random.randn(N_sym * samp_rate)
    return np.cumsum(np.sqrt(noise_var) * noise)

def s2b(s):
    """
    Converts a stream of QAM-16 symbols to a bit stream.

    Parameters:
        s (numpy.ndarray): Array of QAM-16 symbols.

    Returns:
        numpy.ndarray: Bit stream.
    """
    l_s = len(s)
    dec = np.zeros(l_s)
    
    # Map symbols to decimal
    real_map = {-1: 4, 3: 8, 1: 12}
    imag_map = {-1: 1, 3: 2, 1: 3}
    
    for l in range(l_s):
        dec[l] = real_map.get(np.real(s[l]), 0) + imag_map.get(np.imag(s[l]), 0)
        
    # Convert decimal to binary
    r_data = np.zeros(4 * l_s)
    for l in range(l_s):
        r_data[4*l:4*l+4] = d2b(int(dec[l]), 4)
        
    return r_data

def s2bQAM64(s):
    """
    Converts a stream of QAM-64 symbols to a bit stream.

    Parameters:
        s (numpy.ndarray): Array of QAM-64 symbols.

    Returns:
        numpy.ndarray: Bit stream.
    """
    l_s = len(s)
    dec = np.zeros(l_s)
    
    # Define mapping for real and imaginary parts
    real_map = {3: 56, 1: 48, 5: 40, 7: 32, -3: 24, -1: 16, -5: 8}
    imag_map = {3: 7, 1: 6, 5: 5, 7: 4, -3: 3, -1: 2, -5: 1}
    
    for l in range(l_s):
        real_val = np.real(s[l])
        imag_val = np.imag(s[l])
        dec[l] = real_map.get(real_val, 0) + imag_map.get(imag_val, 0)
    
    # Convert decimal to binary
    r_data = np.zeros(6 * l_s)
    for l in range(l_s):
        r_data[6*l:6*l+6] = d2b(int(dec[l]), 6)
        
    return r_data

from scipy.special import erfc

def BER(Qfac):
    """
    Computes the Bit Error Rate from the Q-factor.

    Parameters:
        Qfac (float): Q-factor.

    Returns:
        float: Bit Error Rate.
    """
    return erfc(10**(Qfac/20) / np.sqrt(2)) / 2

def Qfac(BER):
    """
    Computes the Q-factor from the Bit Error Rate.

    Parameters:
        BER (float): Bit Error Rate.

    Returns:
        float: Q-factor.
    """
    return 20 * np.log10(np.sqrt(2) * erfcinv(2 * BER))

def simulate_dpol_fiber_propagation(signal, total_fiber_length, num_spans, num_segments_per_span, 
                                    nonlinearity_coeff, attenuation_db, dispersion_param, 
                                    sample_interval, tau_values, theta_values, phi_values):
    """
    Simulates the propagation of a dual-polarized signal through a fiber using the Split-Step Fourier Method (SSFM).

    Parameters:
        signal (tuple): Tuple of numpy arrays (x_pol_signal_time, y_pol_signal_time) representing the X and Y polarizations.
        total_fiber_length (float): Total length of the optical fiber.
        num_spans (int): Number of fiber spans.
        num_segments_per_span (int): Number of segments per span.
        nonlinearity_coeff (float): Nonlinear coefficient of the fiber.
        attenuation_db (float): Fiber attenuation in dB.
        dispersion_param (float): Dispersion parameter of the fiber.
        sample_interval (float): Time interval between samples.
        tau_values (array): Array of PMD values per segment.
        theta_values (array): Array of rotation angles for birefringence per segment.
        phi_values (array): Array of phase shifts per segment.

    Returns:
        tuple: Tuple of arrays (x_pol_signal_time, y_pol_signal_time) after propagation.
    """
    alpha_linear = (np.log(10) / 10) * (10 ** -3) * attenuation_db
    span_length = total_fiber_length / num_spans
    gain_db = attenuation_db * (span_length * 10 ** -3)
    gain_linear = 10 ** (gain_epoch_db / 10)
    x_pol_signal_time, y_pol_signal_time = signal
    num_samples = len(x_pol_signal_time)
    frequency_array = 2 * np.pi / (num_samples * sample_interval) * fftshift(np.arange(-num_samples / 2, num_samples / 2))
    segment_length = total_fiber_length / (num_segments_per_span * num_spans)
    dispersion_effect = np.exp(-(1j / 2) * dispersion_param * (frequency_array ** 2) * segment_length)  # Dispersion
    attenuation_effect = np.exp((-alpha_linear / 2) * segment_length)  # Attenuation

    for span_index in range(num_spans):
        for segment_index in range(num_segments_per_span):
            x_pol_signal_freq = fft(x_pol_signal_time)
            y_pol_signal_freq = fft(y_pol_signal_time)

            # Apply dispersion
            x_pol_signal_freq *= dispersion_effect
            y_pol_signal_freq *= dispersion_effect

            # Birefringence and phase shifts
            theta = theta_values[span_index * num_segments_per_span + segment_index]
            phi = phi_values[span_index * num_segments_per_span + segment_index]
            tau = tau_values[span_index * num_segments_per_span + segment_index]

            # PMD and phase shifts application
            x_pol_signal_freq *= np.exp(1j * (frequency_array * tau / 2 + phi / 2))
            y_pol_signal_freq *= np.exp(-1j * (frequency_array * tau / 2 + phi / 2))

            # Back to time domain and apply nonlinearity and loss
            x_pol_signal_time = ifft(x_pol_signal_freq) * attenuation_effect
            y_pol_signal_time = ifft(y_pol_signal_freq) * attenuation_effect

            # Effective nonlinearity
            x_pol_signal_time *= np.exp(1j * nonlinearity_coeff * (np.abs(x_pol_signal_time) ** 2 + (2 / 3) * np.abs(y_pol_signal_time) ** 2) * segment_length)
            y_pol_signal_time *= np.exp(1j * nonlinearity_coeff * (np.abs(y_pol_signal_time) ** 2 + (2 / 3) * np.abs(x_pol_signal_time) ** 2) * segment_length)

        # Gain compensation per span
        x_pol_signal_time *= np.sqrt(gain_linear)
        y_pol_signal_time *= np.sqrt(gain_linear)

    return x_pol_signal_time, y_pol_signal_time

def ssfm_dpol_nonsymm(signal, span_lengths, gamma, alpha_db, beta2, dt, tau_values, theta_values, phi_values, bandwidth, noise_figure_db, h, f0, num_samples, time_array):
    """
    Simulates dual-polarized signal propagation in a non-symmetric fiber configuration,
    alternating between single-mode fiber (SMF) and dispersion compensation fiber (DCF) spans.

    Parameters:
        signal (tuple): Initial x and y polarization signals as arrays.
        span_lengths (list): Lengths of each span in kilometers.
        gamma (list): Nonlinear coefficients for each span.
        alpha_db (float): Attenuation in dB/km.
        beta2 (list): Dispersion coefficients for each span.
        dt (float): Time step between samples.
        tau_values, theta_values, phi_values (list): PMD, rotation angles, and phase shifts for each span.
        bandwidth (float): Optical signal bandwidth.
        noise_figure_db (float): Optical amplifier noise figure in dB.
        h (float): Planck's constant.
        f0 (float): Central frequency of the optical signal.
        num_samples (int): Number of samples in the signal.
        time_array (np.array): Array representing the time domain for simulation.

    Returns:
        tuple: The x and y polarization signals after propagation through all spans.
    """
    alpha_linear = np.log(10) / 10 * alpha_db * 1e-3
    gain_db = alpha_db * span_lengths
    gain_linear = 10**(gain_db / 10)
    noise_factor = 10**(noise_figure_db / 10)
    var_noise = (gain_linear[0] * gain_linear[1] - 1) * h * f0 * noise_factor * bandwidth

    x_pol, y_pol = signal
    num_freq = 2 * np.pi / (num_samples * dt) * fftshift(np.arange(-num_samples / 2, num_samples / 2))
    step_size = 1e3

    attenuation = np.exp((-alpha_linear / 2) * step_size)
    for j in range(len(span_lengths) // 2):
        dis_s = np.exp(-(1j / 2) * beta2[2 * j] * (num_freq ** 2) * step_size)
        dis_d = np.exp(-(1j / 2) * beta2[2 * j + 1] * (num_freq ** 2) * step_size)
        for _ in range(span_lengths[2 * j]):  # SMF spans
            x_freq = fft(x_pol)
            y_freq = fft(y_pol)

            x_freq *= dis_s
            y_freq *= dis_s

            x_pol = ifft(x_freq) * attenuation
            y_pol = ifft(y_freq) * attenuation

            x_pol *= np.exp(1j * gamma[2 * j] * (np.abs(x_pol)**2 + (2/3) * np.abs(y_pol)**2) * step_size)
            y_pol *= np.exp(1j * gamma[2 * j] * (np.abs(y_pol)**2 + (2/3) * np.abs(x_pol)**2) * step_size)

        x_pol *= np.sqrt(gain_linear[2 * j])
        y_pol *= np.sqrt(gain_linear[2 * j])

        for _ in range(span_lengths[2 * j + 1]):  # DCF spans
            x_freq = fft(x_pol)
            y_freq = fft(y_pol)

            x_freq *= dis_d
            y_freq *= dis_d

            x_pol = ifft(x_freq) * attenuation
            y_pol = ifft(y_freq) * attenuation

        noise_x = np.sqrt(var_noise / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        noise_y = np.sqrt(var_noise / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

        x_pol = np.sqrt(gain_linear[2 * j + 1]) * x_pol + noise_x
        y_pol = np.sqrt(gain_linear[2 * j + 1]) * y_pol + noise_y

    return x_pol, y_pol


def ssfm_chromatic_kerr(signal, span_lengths, gamma, alpha_db, beta2, dt, B, NF_db, h, f0):
    """
    Simulates dual-polarized signal propagation in a fiber using SSFM, considering chromatic dispersion and Kerr effect,
    alternating between single-mode fiber (SMF) and dispersion compensation fiber (DCF) spans without PMD and birefringence.

    Parameters:
        signal (tuple): Tuple of numpy arrays for the X and Y polarizations.
        span_lengths (list): Number of segments for SMF and DCF for each span.
        gamma (list): Nonlinear coefficient for each span.
        alpha_db (float): Attenuation in dB/km.
        beta2 (list): Dispersion coefficients for each span.
        dt (float): Time interval between samples.
        B (float): Bandwidth of the optical signal.
        NF_db (float): Noise figure in dB.
        h (float): Planck's constant.
        f0 (float): Central frequency of the optical signal.

    Returns:
        tuple: X and Y polarization signals after propagation through all spans.
    """
    alpha_linear = np.log(10) / 10 * alpha_db * 1e-3
    gain_db = alpha_db * np.array(span_lengths)
    gain_linear = 10 ** (gain_db / 10)
    noise_factor = 10 ** (NF_db / 10)
    var = (gain_linear[0] * gain_linear[1] - 1) * h * f0 * noise_factor * B

    x_pol, y_pol = signal
    num_samples = len(x_pol)
    omega = 2 * np.pi / (num_samples * dt) * fftshift(np.arange(-num_samples / 2, num_samples / 2))
    step_size = 1e3
    attenuation = np.exp((-alpha_linear / 2) * step_size)

    for j in range(len(span_lengths) // 2):
        dispersion_s = np.exp(-(1j / 2) * beta2[2 * j] * (omega ** 2) * step_size)
        dispersion_d = np.exp(-(1j / 2) * beta2[2 * j + 1] * (omega ** 2) * step_size)

        # Process SMF segments
        for _ in range(span_lengths[2 * j]):
            x_freq = fft(x_pol)
            y_freq = fft(y_pol)

            x_freq *= dispersion_s
            y_freq *= dispersion_s

            x_pol = ifft(x_freq) * attenuation
            y_pol = ifft(y_freq) * attenuation

            x_pol *= np.exp(1j * 8 / 9 * gamma[2 * j] * (np.abs(x_pol)**2 + np.abs(y_pol)**2) * step_size)
            y_pol *= np.exp(1j * 8 / 9 * gamma[2 * j] * (np.abs(y_pol)**2 + np.abs(x_pol)**2) * step_size)

        x_pol *= np.sqrt(gain_linear[2 * j])
        y_pol *= np.sqrt(gain_linear[2 * j])

        # Process DCF segments
        for _ in range(span_lengths[2 * j + 1]):
            x_freq = fft(x_pol)
            y_freq = fft(y_pol)

            x_freq *= dispersion_d
            y_freq *= dispersion_d

            x_pol = ifft(x_freq) * attenuation
            y_pol = ifft(y_freq) * attenuation

        # Apply noise after DCF
        noise_x = np.sqrt(var / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        noise_y = np.sqrt(var / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        x_pol = np.sqrt(gain_linear[2 * j + 1]) * x_pol + noise_x
        y_pol = np.sqrt(gain_linear[2 * j + 1]) * y_pol + noise_y

    return x_pol, y_pol


def DBP1(signal, span_lengths, gamma, alpha_db, beta2, dt, B, NF_db, h, f0):
    alpha = np.log(10) / 10 * alpha_db * 1e-3
    gain_db = alpha_db * span_lengths
    gains = 10 ** (gain_db / 10)

    x, y = signal
    num_samples = len(x)
    omega = 2 * np.pi / (num_samples * dt) * fftshift(np.arange(-num_samples / 2, num_samples / 2))
    step_size = 1e3

    attenuation = np.exp((-alpha / 2) * step_size)
    num_spans = len(span_lengths) // 2

    for j in range(num_spans - 1, -1, -1):
        dispersion_correction = np.exp(-(1j / 2) * -beta2[j] * (omega ** 2) * step_size)

        x /= np.sqrt(gains[j])
        y /= np.sqrt(gains[j])

        for _ in range(span_lengths[j]):
            fx = fft(x / attenuation)
            fy = fft(y / attenuation)

            fx *= dispersion_correction
            fy *= dispersion_correction

            x = ifft(fx)
            y = ifft(fy)

            nonlinear_phase_shift = -8 / 9 * gamma[j] * (np.abs(x)**2 + np.abs(y)**2) * step_size
            x *= np.exp(1j * nonlinear_phase_shift)
            y *= np.exp(1j * nonlinear_phase_shift)

    return x, y

def DBP_nonsymm(signal, span_lengths, gamma, alpha_db, beta2, dt, TAU, THETA, PHI, B, NF_db, h, f0, N_s, t):
    """
    Performs digital back-propagation (DBP) for non-symmetric spans of fibers.
    This DBP starts from the receiver end to the transmitter end, alternating between DCF and SMF.

    Parameters:
        signal (tuple): Tuple of arrays representing the X and Y polarizations of the signal.
        span_lengths (list): Number of segments for each type of fiber within each span.
        gamma (list): Nonlinear coefficients for each fiber span.
        alpha_db (float): Fiber attenuation in dB.
        beta2 (list): Dispersion parameters for each fiber type.
        dt (float): Time interval between samples.
        TAU (list): PMD values for each segment, not used in this function.
        THETA (list): Birefringence angles for each segment, not used in this function.
        PHI (list): Phase shifts for each segment, not used in this function.
        B (float): Bandwidth of the signal.
        NF_db (float): Noise figure in dB.
        h (float): Planck's constant.
        f0 (float): Central frequency of the optical signal.
        N_s (int): Number of samples in the signal.
        t (array): Time array for the signal.

    Returns:
        tuple: X and Y polarization signals after digital back-propagation.
    """
    alpha = np.log(10) / 10 * alpha_db * 1e-3
    G_db = alpha_db * np.array(span_lengths)
    g = 10 ** (G_db / 10)
    nf = 10 ** (NF_db / 10)
    
    x, y = signal
    num_samples = int(len(x))
    omega = 2 * np.pi / (num_samples * dt) * fftshift(np.arange(-num_samples / 2, num_samples / 2))
    step_size = 1e3
    
    attenuation = np.exp((-alpha / 2) * step_size)
    num_spans = int(len(span_lengths) / 2)
    index_counter = int(np.sum(span_lengths))

    # Start DBP from the last span moving to the first span
    for j in range(num_spans):
        index = -2 * j - 1  # DCF first (odd indices)
        
        # Adjust signal by the gain of the last amplifier
        x /= np.sqrt(g[index])
        y /= np.sqrt(g[index])
        
        # DCF processing
        for _ in range(span_lengths[index]):
            index_counter -= 1
            fx = fft(x / attenuation)
            fy = fft(y / attenuation)

            # Apply dispersion correction
            dispersion_correction = np.exp((1j / 2) * beta2[index] * (omega ** 2) * step_size)
            fx *= dispersion_correction
            fy *= dispersion_correction

            x = ifft(fx)
            y = ifft(fy)
        
        # SMF processing
        index = -2 * j - 2  # SMF second (even indices)
        x /= np.sqrt(g[index])
        y /= np.sqrt(g[index])

        for _ in range(span_lengths[index]):
            index_counter -= 1
            x = x * np.exp(-1j * gamma[index] * (np.abs(x)**2 + (2 / 3) * np.abs(y)**2) * step_size)
            y = y * np.exp(-1j * gamma[index] * (np.abs(y)**2 + (2 / 3) * np.abs(x)**2) * step_size)

            fx = fft(x / attenuation)
            fy = fft(y / attenuation)

            dispersion_correction = np.exp((1j / 2) * beta2[index] * (omega ** 2) * step_size)
            fx *= dispersion_correction
            fy *= dispersion_correction

            x = ifft(fx)
            y = ifft(fy)

    return x, y


def DSP_nonsymm(signal, l_span, gamma, alpha_db, beta2, dt, TAU, THETA, PHI, B, NF_db, h, f0, N_s, t):
    """
    Simulates perfect digital signal processing (DSP) for optical signals, compensating for linear impairments
    only and ignoring nonlinear effects. The simulation alternates between different types of fibers.

    Parameters:
        signal (tuple): Tuple of arrays representing X and Y polarizations of the signal.
        l_span (list): Lengths of each span in kilometers.
        gamma (list): Nonlinear coefficients (unused in perfect DSP).
        alpha_db (float): Fiber attenuation in dB.
        beta2 (list): Dispersion parameters for each fiber type.
        dt (float): Time interval between samples.
        TAU (list): PMD values, ideally not significant in perfect DSP.
        THETA (list): Birefringence angles, ideally not significant in perfect DSP.
        PHI (list): Phase shifts, ideally not significant in perfect DSP.
        B (float): Bandwidth of the optical signal.
        NF_db (float): Noise figure in dB.
        h (float): Planck's constant.
        f0 (float): Central frequency of the optical signal.
        N_s (int): Number of samples in the signal.
        t (array): Time array for the signal.

    Returns:
        tuple: Corrected X and Y polarization signals after DSP.
    """
    alpha = np.log(10) / 10 * alpha_db * 1e-3
    G_db = alpha_db * np.array(l_span)
    gains = 10 ** (G_db / 10)
    
    x, y = signal
    num_samples = len(x)
    omega = 2 * np.pi / (num_samples * dt) * fftshift(np.arange(-num_samples / 2, num_samples / 2))
    step_size = 1e3
    
    attenuation = np.exp((-alpha / 2) * step_size)  # Apply attenuation compensation
    n_span = len(l_span) // 2  # Calculate number of spans
    c = sum(l_span)  # Counter for indexing phase, theta, tau adjustments

    x_freq = fft(x)
    y_freq = fft(y)

    # Iterate through each span in reverse order, applying corrections
    for j in range(n_span):
        index = -2 * j - 1  # Index for DCF spans

        # Process DCF spans first in each iteration
        for _ in range(l_span[index]):
            c -= 1
            
            # Apply phase, PMD and birefringence corrections even though it's ideal DSP (no actual effect)
            phi_correction = np.exp(-1j * PHI[c] / 2)
            theta_correction = np.exp(-1j * TAU[c] * omega / 2)
            birefringence_correction = cos_m(-THETA[c]) * x_freq + sin_m(-THETA[c]) * y_freq

            x_freq = birefringence_correction * phi_correction * theta_correction
            y_freq = birefringence_correction * phi_correction * theta_correction

        # Process SMF spans next
        index = -2 * j - 2
        for _ in range(l_span[index]):
            c -= 1
            
            # Apply phase, PMD and birefringence corrections
            phi_correction = np.exp(-1j * PHI[c] / 2)
            theta_correction = np.exp(-1j * TAU[c] * omega / 2)
            birefringence_correction = cos_m(-THETA[c]) * x_freq + sin_m(-THETA[c]) * y_freq

            x_freq = birefringence_correction * phi_correction * theta_correction
            y_freq = birefringence_correction * phi_correction * theta_correction

    # Convert frequencies back to time domain signals
    x_corrected = ifft(x_freq)
    y_corrected = ifft(y_freq)

    return x_corrected, y_corrected

def t2f(t):
    """
    Converts a time array into its corresponding frequency array assuming uniform spacing.

    Parameters:
        t (np.array): Array of time samples.

    Returns:
        np.array: Array of frequency samples.
    """
    dt = t[1] - t[0]  # Time interval between samples
    F = 1 / dt  # Sampling frequency
    f = np.linspace(-F / 2, F / 2, len(t), endpoint=False)  # Frequency array from -Nyquist to Nyquist
    return f


def DBP_dpol(signal, l_f, n_span, n_seg, gamma, alpha_db, beta2, dt):
    """
    Simulates digital back propagation (DBP) for a dual-polarized optical signal over a fiber link.
    This process compensates primarily for chromatic dispersion and nonlinear phase shifts due to the Kerr effect.

    Parameters:
        signal (tuple): Tuple containing the arrays of X and Y polarizations of the optical signal.
        l_f (float): Total fiber length.
        n_span (int): Number of spans in the fiber link.
        n_seg (int): Number of segments per span to simulate.
        gamma (float): Nonlinear coefficient of the fiber.
        alpha_db (float): Attenuation of the fiber in dB.
        beta2 (float): Dispersion parameter of the fiber.
        dt (float): Time interval between consecutive samples in the signal.

    Returns:
        tuple: X and Y polarization signals after compensation via DBP.
    """
    # Calculate linear loss coefficient
    alpha = (log(10) / 10) * (10 ** -3) * alpha_db
    # Length of each span
    l_span = l_f / n_span
    # Retrieve signal components
    signal_recx, signal_recy = signal[0], signal[1]
    # Number of samples in the signal
    l_s = int(len(signal_recx))
    # Frequency components corresponding to each sample
    w = 2 * np.pi / (l_s * dt) * fftshift(np.arange(-l_s / 2, l_s / 2))
    # Step size in each segment
    step_s = l_f / (n_seg * n_span)
    # Dispersion compensation factor for each step
    dis = np.exp(-(1j / 2) * beta2 * (w ** 2) * step_s)

    # Iterate through each span and each segment
    for j in range(n_span):
        for jj in range(n_seg):
            # Fourier transform of the signals
            signal_recfx = fft(signal_recx)
            signal_recfy = fft(signal_recy)
            # Apply dispersion compensation
            signal_recfx *= dis
            signal_recfy *= dis
            # Inverse Fourier transform to return to time domain
            signal_recx = ifft(signal_recfx)
            signal_recy = ifft(signal_recfy)
            # Apply nonlinear phase shift compensation
            signal_recx *= np.exp(1j * gamma * ((np.abs(signal_recx) ** 2) + (2 / 3) * (np.abs(signal_recy) ** 2)) * step_s)
            signal_recy *= np.exp(1j * gamma * ((np.abs(signal_recy) ** 2) + (2 / 3) * (np.abs(signal_recx) ** 2)) * step_s)

    return signal_recx, signal_recy

def cmd(signal, l_f, beta2, dt):
    """
    Compensates for chromatic dispersion in a dual-polarized signal.

    Parameters:
        signal (tuple): Tuple containing arrays for the X and Y polarizations of the optical signal.
        l_f (float): Length of the fiber over which dispersion has occurred.
        beta2 (float): Dispersion parameter of the fiber.
        dt (float): Sampling interval of the digital signal.

    Returns:
        tuple: The X and Y polarization signals after chromatic dispersion compensation.
    """
    x, y = signal
    num_samples = len(x)
    omega = 2 * np.pi / (num_samples * dt) * fftshift(np.arange(-num_samples / 2, num_samples / 2))
    dispersion_compensation = np.exp(-(1j / 2) * (-beta2) * (omega ** 2) * l_f)

    x_freq = fft(x) * dispersion_compensation
    y_freq = fft(y) * dispersion_compensation

    x_compensated = ifft(x_freq)
    y_compensated = ifft(y_freq)

    return x_compensated, y_compensated


def CMA_RDEM(signal, Nsps, N_s, N_taps, p):
    """
    Implements the Constant Modulus Algorithm (CMA) and Radius Directed Equalization (RDE) to 
    adaptively equalize the dual-polarization signals in optical communication.

    Parameters:
        signal (tuple): Tuple containing X and Y polarization components of the input signal.
        Nsps (int): Number of samples per symbol (oversampling rate).
        N_s (int): Total number of samples in the signal.
        N_taps (int): Number of taps in the adaptive equalizer filters.
        p (float): Power level used for signal normalization.

    Returns:
        tuple: Equalized X and Y polarization signals.
    """

    # Normalize the input signal by the square root of power level `p`
    s_inx = signal[0] / sqrt(p)
    s_iny = signal[1] / sqrt(p)
    
    # Initialize adaptive filter coefficients for dual-polarization signals
    F_1 = np.zeros(N_taps, dtype=np.complex_)  # Filter for X-input to X-output
    F_2 = np.zeros(N_taps, dtype=np.complex_)  # Filter for Y-input to X-output
    F_1[int(floor(N_taps / 2))] = 1 + 0j       # Start with impulse in the middle of the filter

    # Define adaptation rate and iteration parameters
    mu = 0.001     # Learning rate for adaptation
    itr = 10000    # Number of total iterations for adaptation
    th = 5000      # Threshold to switch from CMA to RDE

    # Radii defined for different constellation rings in RDE
    R1 = 0.4470035621347161
    R2 = 0.9995303606046092
    R3 = 1.3410107005462
    g1 = (R1 + R2) / 2  # Boundary between inner and middle ring
    g3 = (R2 + R3) / 2  # Boundary between middle and outer ring

    # CMA Loop for initial convergence
    for l in range(th):
        idx_s = l * Nsps
        s_x = s_inx[idx_s:idx_s + N_taps]
        s_y = s_iny[idx_s:idx_s + N_taps]
        st_outx = np.dot(F_1, s_x) + np.dot(F_2, s_y)
        e_x = 2 * (abs(st_outx)**2 - sqrt(2)) * st_outx  # Error signal based on CMA criterion
        F_1 -= mu * e_x * np.conjugate(s_x)  # Update F_1
        F_2 -= mu * e_x * np.conjugate(s_y)  # Update F_2

    # RDE Loop for fine-tuning the coefficients
    for l in range(th, itr):
        idx_s = l * Nsps
        s_x = s_inx[idx_s:idx_s + N_taps]
        s_y = s_iny[idx_s:idx_s + N_taps]
        st_outx = np.dot(F_1, s_x) + np.dot(F_2, s_y)

        # Adjust target radius based on the detected symbol region
        if abs(st_outx) < g1:
            Rs_x = R1**2
        elif abs(st_outx) > g3:
            Rs_x = R3**2
        else:
            Rs_x = R2**2

        e_x = 2 * (abs(st_outx)**2 - Rs_x) * st_outx  # Error signal based on RDE criterion
        F_1 -= mu * e_x * np.conjugate(s_x)
        F_2 -= mu * e_x * np.conjugate(s_y)

    # Output computation over the last symbols in the signal minus the number of taps
    limit = N_s - N_taps
    s_outx = np.zeros(limit, dtype=np.complex_)
    s_outy = np.zeros(limit, dtype=np.complex_)
    for l in range(limit):
        idx_s = l * Nsps
        s_x = s_inx[idx_s:idx_s + N_taps]
        s_y = s_iny[idx_s:idx_s + N_taps]
        s_outx[l] = np.dot(F_1, s_x) + np.dot(F_2, s_y)
        s_outy[l] = np.dot(-np.conjugate(np.flip(F_2)), s_x) + np.dot(np.conjugate(np.flip(F_1)), s_y)

    # Scale output signals back to the original power level
    sx_out = s_outx * sqrt(p)
    sy_out = s_outy * sqrt(p)

    return sx_out, sy_out

def CMA_RDE(signal, Nsps, N_s, N_taps, p):
    """
    Implements both Constant Modulus Algorithm (CMA) and Radius Directed Equalization (RDE) for dual-polarization 
    signals, adjusting filter taps to equalize the signal based on predetermined radius constraints and constant modulus.

    Parameters:
        signal (tuple): Tuple containing the X and Y polarization components of the input optical signal.
        Nsps (int): Number of samples per symbol, defining the oversampling rate of the signal.
        N_s (int): Total number of samples in the signal.
        N_taps (int): Number of taps in the adaptive equalizer filters.
        p (float): Power normalization factor used to scale the input signals.

    Returns:
        tuple: Returns equalized X and Y polarization signals, scaled back to the original power level.
    """
    # Normalize the input signal
    s_inx = signal[0] / sqrt(p)
    s_iny = signal[1] / sqrt(p)
    
    # Initialize the weights of the four FIR filters for dual-polarization
    F_1 = np.zeros(N_taps, dtype=np.complex_)  # Filter for X-input to X-output
    F_2 = np.zeros(N_taps, dtype=np.complex_)  # Filter for Y-input to X-output
    F_3 = np.zeros(N_taps, dtype=np.complex_)  # Filter for X-input to Y-output
    F_4 = np.zeros(N_taps, dtype=np.complex_)  # Filter for Y-input to Y-output
    F_1[int(floor(N_taps / 2))] = 1 + 0j  # Center tap initialization for X
    F_4[int(floor(N_taps / 2))] = 1 + 0j  # Center tap initialization for Y

    # Learning rate for filter updates
    mu = 0.001
    itr = 10000  # Total iterations for training
    th = 5000    # Threshold iteration to switch from CMA to RDE

    # CMA adaptation phase
    for l in range(th):
        idx_s = l * Nsps
        s_x = s_inx[idx_s:idx_s + N_taps]
        s_y = s_iny[idx_s:idx_s + N_taps]
        st_outx = np.vdot(F_1, s_x) + np.vdot(F_2, s_y)
        st_outy = np.vdot(F_3, s_x) + np.vdot(F_4, s_y)
        e_x = (1 - abs(st_outx)**2)
        e_y = (1 - abs(st_outy)**2)
        F_1 += mu * e_x * s_x * np.conjugate(st_outx)
        F_2 += mu * e_x * s_y * np.conjugate(st_outx)
        F_3 += mu * e_y * s_x * np.conjugate(st_outy)
        F_4 += mu * e_y * s_y * np.conjugate(st_outy)

    # Radii definitions for RDE
    R1 = 0.4470035621347161
    R2 = 0.9995303606046092
    R3 = 1.3410107005462
    g1 = (R1 + R2) / 2
    g3 = (R2 + R3) / 2

    # RDE adaptation phase
    for l in range(th, itr):
        idx_s = l * Nsps
        s_x = s_inx[idx_s:idx_s + N_taps]
        s_y = s_iny[idx_s:idx_s + N_taps]
        st_outx = np.vdot(F_1, s_x) + np.vdot(F_2, s_y)
        st_outy = np.vdot(F_3, s_x) + np.vdot(F_4, s_y)

        # Determine error based on distance from defined radii
        Rs_x = R1**2 if abs(st_outx) < g1 else R3**2 if abs(st_outx) > g3 else R2**2
        Rs_y = R1**2 if abs(st_outy) < g1 else R3**2 if abs(st_outy) > g3 else R2**2

        e_x = (Rs_x - abs(st_outx)**2)
        e_y = (Rs_y - abs(st_outy)**2)
        F_1 += mu * e_x * s_x * np.conjugate(st_outx)
        F_2 += mu * e_x * s_y * np.conjugate(st_outx)
        F_3 += mu * e_y * s_x * np.conjugate(st_outy)
        F_4 += mu * e_y * s_y * np.conjugate(st_outy)

    # Construct the output signals
    limit = N_s - N_taps
    s_outx = np.zeros((limit,), dtype=np.complex_)
    s_outy = np.zeros((limit,), dtype=np.complex_)
    for l in range(limit):
        idx_s = l * Nsps
        s_x = s_inx[idx_s:idx_s + N_taps]
        s_y = s_iny[idx_s:idx_s + N_taps]
        s_outx[l] = np.vdot(F_1, s_x) + np.vdot(F_2, s_y)
        s_outy[l] = np.vdot(F_3, s_x) + np.vdot(F_4, s_y)

    # Scale output signals back to original power level
    sx_out = s_outx * sqrt(p)
    sy_out = s_outy * sqrt(p)

    return sx_out, sy_out


def CMA_RDE2(signal, Nsps, N_s, N_taps, p):
    """
    Implements both Constant Modulus Algorithm (CMA) and Radius Directed Equalization (RDE) for
    dual-polarization signals, adjusting filter taps to equalize the signal based on predetermined
    radius constraints and constant modulus.

    Parameters:
        signal (tuple): Tuple containing the X and Y polarization components of the input optical signal.
        Nsps (int): Number of samples per symbol, defining the oversampling rate of the signal.
        N_s (int): Total number of samples in the signal.
        N_taps (int): Number of taps in the adaptive equalizer filters.
        p (float): Power normalization factor used to scale the input signals.

    Returns:
        tuple: Returns equalized X and Y polarization signals, scaled back to the original power level.
    """
    # Normalize the input signals
    s_inx = signal[0] / sqrt(p)
    s_iny = signal[1] / sqrt(p)

    # Initialize the filter weights for four FIR filters
    F_1 = np.zeros(N_taps, dtype=np.complex_)  # Filter for X-input to X-output
    F_2 = np.zeros(N_taps, dtype=np.complex_)  # Filter for Y-input to X-output
    F_3 = np.zeros(N_taps, dtype=np.complex_)  # Filter for X-input to Y-output
    F_4 = np.zeros(N_taps, dtype=np.complex_)  # Filter for Y-input to Y-output
    F_1[int(floor(N_taps / 2)) + 1] = 1 + 0j   # Initialize center tap offset for F_1
    F_4[int(floor(N_taps / 2)) + 1] = 1 + 0j   # Initialize center tap offset for F_4

    # Learning rate for the adaptive algorithm
    mu = 0.001

    # Number of iterations for the algorithms
    itr = 10000  # Total iterations
    th = 5000    # Threshold to switch from CMA to RDE

    # CMA phase to initially adjust filters based on the constant modulus criterion
    for l in range(th):
        idx_s = l * Nsps
        s_x = s_inx[idx_s:idx_s + N_taps]
        s_y = s_iny[idx_s:idx_s + N_taps]
        st_outx = np.dot(F_1, s_x) + np.dot(F_2, s_y)
        st_outy = np.dot(F_3, s_x) + np.dot(F_4, s_y)
        e_x = 2 * (abs(st_outx)**2 - 1) * st_outx
        e_y = 2 * (abs(st_outy)**2 - 1) * st_outy
        F_1 += mu * e_x * np.conjugate(s_x)
        F_2 += mu * e_x * np.conjugate(s_y)
        F_3 += mu * e_y * np.conjugate(s_x)
        F_4 += mu * e_y * np.conjugate(s_y)

    # Radii for RDE
    R1 = 0.4470035621347161
    R2 = 0.9995303606046092
    R3 = 1.3410107005462
    g1 = (R1 + R2) / 2
    g3 = (R2 + R3) / 2

    # RDE phase to further refine the filters based on target radii
    for l in range(th, itr):
        idx_s = l * Nsps
        s_x = s_inx[idx_s:idx_s + N_taps]
        s_y = s_iny[idx_s:idx_s + N_taps]
        st_outx = np.dot(F_1, s_x) + np.dot(F_2, s_y)
        st_outy = np.dot(F_3, s_x) + np.dot(F_4, s_y)

        # Adjust error calculations based on radial targets
        Rs_x = R1**2 if abs(st_outx) < g1 else R3**2 if abs(st_outx) > g3 else R2**2
        Rs_y = R1**2 if abs(st_outy) < g1 else R3**2 if abs(st_outy) > g3 else R2**2

        e_x = 2 * (Rs_x - abs(st_outx)**2) * st_outx
        e_y = 2 * (Rs_y - abs(st_outy)**2) * st_outy
        F_1 += mu * e_x * np.conjugate(s_x)
        F_2 += mu * e_x * np.conjugate(s_y)
        F_3 += mu * e_y * np.conjugate(s_x)
        F_4 += mu * e_y * np.conjugate(s_y)

    # Output signal construction
    limit = N_s - N_taps
    s_outx = np.zeros((limit,), dtype=np.complex_)
    s_outy = np.zeros((limit,), dtype=np.complex_)
    for l in range(limit):
        idx_s = l * 2
        s_x = s_inx[idx_s:idx_s + N_taps]
        s_y = s_iny[idx_s:idx_s + N_taps]
        s_outx[l] = np.dot(F_1, s_x) + np.dot(F_2, s_y)
        s_outy[l] = np.dot(F_3, s_x) + np.dot(F_4, s_y)

    # Scale output signals back to original power level
    sx_out = s_outx * sqrt(p)
    sy_out = s_outy * sqrt(p)

    return sx_out, sy_out#****************************************************************************************************

def pol_dmux(sym_r, sym_t):
    """
    Performs polarization demultiplexing by cross-correlating received and transmitted symbols
    to determine the best alignment, and hence effectively separate the mixed polarization states.

    Parameters:
        sym_r (tuple): Tuple containing arrays of received symbols for X and Y polarizations.
        sym_t (tuple): Tuple containing arrays of transmitted symbols for X and Y polarizations.

    Returns:
        tuple: Tuple containing the demultiplexed output symbols for X and Y polarizations.
    """
    # Unpack received and transmitted symbols for each polarization
    r_x, r_y = sym_r
    t_x, t_y = sym_t

    # Calculate the cross-correlation between received and transmitted symbols
    n = len(r_x) - 1
    ax = np.arange(-n, n + 1)
    cor_xx = np.abs(np.correlate(r_x, t_x, "full"))
    cor_xy = np.abs(np.correlate(r_x, t_y, "full"))
    cor_yx = np.abs(np.correlate(r_y, t_x, "full"))
    cor_yy = np.abs(np.correlate(r_y, t_y, "full"))

    # Find max correlation values and corresponding indices
    mcor_xx, i_xx = np.max(cor_xx), np.argmax(cor_xx)
    mcor_xy, i_xy = np.max(cor_xy), np.argmax(cor_xy)
    mcor_yx, i_yx = np.max(cor_yx), np.argmax(cor_yx)
    mcor_yy, i_yy = np.max(cor_yy), np.argmax(cor_yy)

    # Determine output symbols based on max correlation comparisons
    if mcor_xx > mcor_xy:
        sout_x = np.roll(r_x, -ax[i_xx])
    else:
        sout_x = np.roll(r_y, -ax[i_xy])
        
    if mcor_yy > mcor_yx:
        sout_y = np.roll(r_y, -ax[i_yy])
    else:
        sout_y = np.roll(r_x, -ax[i_yx])

    return sout_x, sout_y


def cpe(sym_r, sym_t, s_wind, p):
    """
    Performs Carrier Phase Estimation (CPE) and correction based on the QAM-16 demodulation
    of received symbols compared to transmitted symbols over multiple phase shifts.

    Parameters:
        sym_r (tuple): Received X and Y symbols (arrays).
        sym_t (tuple): Transmitted X and Y symbols (not used in current function logic).
        s_wind (int): Sliding window size for averaging the distances.
        p (float): Power or another parameter for the deQAM16 function.

    Returns:
        tuple: Corrected X and Y output symbols.
    """
    sr_x, sr_y = sym_r  # Unpack received symbols
    B = 32  # Number of angles to be tested.
    theta_t = np.linspace(-pi/4, pi/4, B)  # Equally spaced angles within -pi/4 to pi/4
    l_sr = len(sr_x)  # Length of received symbols array

    # Initialize distance arrays
    d_x = np.zeros((l_sr, len(theta_t)))
    d_y = np.zeros((l_sr, len(theta_t)))

    # Apply phase shifts and calculate distances
    for l in range(l_sr):
        srx_test = sr_x[l] * exp(-1j * theta_t)
        sry_test = sr_y[l] * exp(-1j * theta_t)
        srx_test_d = deQAM16(srx_test, p)
        sry_test_d = deQAM16(sry_test, p)
        d_x[l] = abs(srx_test - srx_test_d)**2
        d_y[l] = abs(sry_test - sry_test_d)**2

    # Average distances within sliding window
    dx_avg = np.zeros_like(d_x)
    dy_avg = np.zeros_like(d_y)
    half_wind = s_wind // 2
    for l in range(half_wind, l_sr - half_wind):
        for k in range(len(theta_t)):
            dx_avg[l, k] = mean(d_x[l-half_wind:l+half_wind+1, k])
            dy_avg[l, k] = mean(d_y[l-half_wind:l+half_wind+1, k])

    # Estimate best phase correction
    theta_estx = np.zeros(l_sr)
    theta_esty = np.zeros(l_sr)
    for l in range(half_wind, l_sr - half_wind):
        theta_estx[l] = theta_t[np.argmin(dx_avg[l])]
        theta_esty[l] = theta_t[np.argmin(dy_avg[l])]

    # Mitigate cycle slips by detecting large phase jumps
    phase_trx = diff(theta_estx)
    phase_try = diff(theta_esty)
    phase_shiftx = cumsum((phase_trx < -pi/4) * (-pi/2) + (phase_trx > pi/4) * (pi/2))
    phase_shifty = cumsum((phase_try < -pi/4) * (-pi/2) + (phase_try > pi/4) * (pi/2))

    # Correct the original symbols using the estimated phase
    thet_x = theta_estx - phase_shiftx
    thet_y = theta_esty - phase_shifty
   
#***************************************************************************
# #seconed cpe stage
    v_1x =s_outtx[lim1:lim2]
    v_2x =st_x[lim1:lim2]
    ratio_x= np.divide(v_1x,v_2x)
    anglex = cmath.phase(mean(ratio_x))
    ss_x=np.multiply(s_outtx, exp(-1j*anglex))
    v_1y =s_outty[lim1:lim2]
    v_2y =st_y[lim1:lim2]
    ratio_y= np.divide(v_1y,v_2y)
    angley = cmath.phase(mean(ratio_y))
    ss_y=np.multiply(s_outty, exp(-1j*angley))
    #*******************************************************************
    return ss_x,ss_y

def diff(seq):
    """
    Computes a discrete differentiation of a given sequence.

    Parameters:
        seq (array-like): The sequence for which the differentiation is to be computed.

    Returns:
        numpy.ndarray: The differentiated sequence with the same length as the input,
                       where the first element is the same as in the original sequence.
    """
    # Determine the length of the input sequence
    l_seq = len(seq)
    
    # Initialize an array of zeros with the same length as the input sequence
    # This array will store the differentiated sequence
    seq_d = np.zeros((1, l_seq))
    
    # Loop over the entire sequence to compute the difference between consecutive elements
    for l in range(l_seq):
        if l == 0:
            # For the first element, since there is no previous element, copy it as is
            seq_d[0, l] = seq[l]
        else:
            # For other elements, compute the difference with the previous element
            seq_d[0, l] = seq[l] - seq[l-1]
    
    # Return the array containing the differentiated values
    return seq_d


def d2b(x, k):
    """
    Converts an integer to a binary list of a fixed size 'k'.
    
    Parameters:
        x (int): The integer to convert to binary.
        k (int): The fixed size of the binary list (number of bits).
    
    Returns:
        list: A binary list with length 'k', padded with zeros if necessary.
    """
    # Convert integer x to binary string, remove the '0b' prefix, and convert to a list of characters
    bi = list(bin(x)[2:])
    
    # Convert list of character bits to floats (0 or 1)
    l = len(bi)
    for i in range(l):
        bi[i] = float(bi[i])
    
    # If the current length of the binary list isn't equal to k, pad with zeros
    if l != k:
        # Create a list of zeros of length (k-l) to pad the binary list
        biry = list(zeros(k-l))
        biry.extend(bi)
    else:
        # If no padding is needed, use the original list
        biry = bi
    
    return biry


def err_count(symb_est, symb_trs):
    """
    Counts the number of errors between estimated and transmitted symbols.
    
    Parameters:
        symb_est (array-like): Estimated symbols.
        symb_trs (array-like): Transmitted symbols.
    
    Returns:
        int: The total count of errors (absolute differences) between the estimated and transmitted symbols.
    """
    # Compute the element-wise absolute difference between the two symbol arrays
    # and sum these differences to get the total count of errors
    return np.sum(np.abs(symb_est != symb_trs))

def pha(v_1, v_2):
    """
    Calculates the angle (phase difference) between two complex vectors.
    
    Parameters:
        v_1 (array-like): First complex vector.
        v_2 (array-like): Second complex vector, must be non-zero wherever division occurs.
    
    Returns:
        numpy.ndarray: Array containing the phase differences for each corresponding element of v_1 and v_2.
    """
    l_v = len(v_1)
    theta = np.zeros((1, l_v))
    div = np.divide(v_1, v_2)  # Element-wise division of the two complex vectors
    
    for l in range(l_v):
        theta[0, l] = cmath.phase(div[l])  # Compute phase of each complex division result
    
    return theta

def noiseadd(B, alpha_db, NF_db, h, f0, N_s, t, l_span):
    """
    Adds noise to a fiber channel based on the given parameters and returns the noisy signal.
    
    Parameters:
        B (float): Bandwidth of the signal.
        alpha_db (float): Attenuation of the fiber in dB.
        NF_db (float): Noise figure of the amplifier in dB.
        h (float): Planck's constant.
        f0 (float): Central frequency of the signal.
        N_s (int): Number of samples in the signal.
        t (array-like): Time array or similar parameter not directly used in the function.
        l_span (float): Length of the fiber span.
    
    Returns:
        tuple: Two numpy.ndarrays representing the noise-added signals in X and Y polarizations.
    """
    alpha = (np.log(10) / 10) * (10**(-3)) * alpha_db
    G_db = alpha_db * l_span
    g = 10**(G_db / 10)
    nf = 10**(NF_db / 10)
    var = 10 * (g - 1) * h * f0 * (nf / 2) * B  # Calculate noise variance
    
    # Generate complex Gaussian noise for X and Y polarizations
    noise_x = np.sqrt(var / 2) * (np.random.randn(N_s) + 1j * np.random.randn(N_s))
    noise_y = np.sqrt(var / 2) * (np.random.randn(N_s) + 1j * np.random.randn(N_s))
    
    # Apply modulation or any processing required on noise signals
    noisex_added = mod(t, noise_x, B)
    noisey_added = mod(t, noise_y, B)
    
    return noisex_added, noisey_added

def AWGN(x, var, N_span):
    """
    Adds Additive White Gaussian Noise (AWGN) to the input signal, simulating noise addition across multiple spans.
    
    Parameters:
        x (array-like): The input signal to which noise is to be added.
        var (float): The variance of the Gaussian noise.
        N_span (int): The number of spans over which the noise is added.
    
    Returns:
        numpy.ndarray: The signal with added noise.
    """
    l_s = len(x)
    y = x.copy()
    
    for _ in range(N_span):  # Use N_span for the number of noise additions
        # Generate complex Gaussian noise and add it to the signal
        y += np.sqrt(var / 2) * (np.random.randn(l_s) + 1j * np.random.randn(l_s))
    
    return y

def data_prep(s_wind, inp_signal, s, N_blocks, N_s, N_d):
    """
    Prepares data for neural network training and validation by processing input signals and corresponding labels.
    
    Parameters:
        s_wind (int): Sliding window size used in data preparation.
        inp_signal (array): Input signal array from which features are extracted.
        s (array): Output signal array from which labels are extracted.
        N_blocks (int): Total number of data blocks (both training and validation).
        N_s (int): Number of samples per block.
        N_d (int): Number of desired output samples per example.
    
    Returns:
        tuple: Contains reshaped input and output data for both training and validation.
    """
    # Calculate the number of training and validation blocks
    tra_blocks = int(N_blocks / 2)
    val_blocks = int(N_blocks / 2)
    
    # Calculate the number of examples based on block sizes and the dimensions required
    N_ex_tra = (tra_blocks) * (N_s - (s_wind + N_d - 1))  # Number of training examples
    N_ex_val = (val_blocks) * (N_s - (s_wind + N_d - 1))  # Number of validation examples
    
    L_ex = (N_d + s_wind)  # Length of each example per polarization
    
    # Initialize arrays for neural network input and output
    inp_Nnet_x = np.zeros((N_ex_tra, L_ex)) + 0j
    out_Nnet_x = np.zeros((N_ex_tra, N_d)) + 0j
    inp_Nnet_y = np.zeros((N_ex_tra, L_ex)) + 0j
    out_Nnet_y = np.zeros((N_ex_tra, N_d)) + 0j
    
    val_inp_Nnet_x = np.zeros((N_ex_val, L_ex)) + 0j
    val_out_Nnet_x = np.zeros((N_ex_val, N_d)) + 0j
    val_inp_Nnet_y = np.zeros((N_ex_val, L_ex)) + 0j
    val_out_Nnet_y = np.zeros((N_ex_val, N_d)) + 0j
    
    # Populate training and validation data
    count = 0
    for k in range(N_blocks):
        for l in range(N_s - (s_wind + N_d - 1)):
            if k < tra_blocks:
                inp_Nnet_x[count] = inp_signal[k, 0, l:l + int(s_wind / 2) + N_d + int(s_wind / 2)]
                out_Nnet_x[count] = s[k, 0, int(s_wind / 2) + l:l + int(s_wind / 2) + N_d]
                inp_Nnet_y[count] = inp_signal[k, 1, l:l + int(s_wind / 2) + N_d + int(s_wind / 2)]
                out_Nnet_y[count] = s[k, 1, int(s_wind / 2) + l:l + int(s_wind / 2) + N_d]
            elif k >= tra_blocks:
                val_inp_Nnet_x[count] = inp_signal[k, 0, l:l + int(s_wind / 2) + N_d + int(s_wind / 2)]
                val_out_Nnet_x[count] = s[k, 0, int(s_wind / 2) + l:l + int(s_wind / 2) + N_d]
                val_inp_Nnet_y[count] = inp_signal[k, 1, l:l + int(s_wind / 2) + N_d + int(s_wind / 2)]
                val_out_Nnet_y[count] = s[k, 1, int(s_wind / 2) + l:l + int(s_wind / 2) + N_d]
            count += 1
    
    # Concatenate X and Y polarization inputs for training and validation
    inp_Nnet = np.concatenate((inp_Nnet_x, inp_Nnet_y), axis=1)
    out_Nnet = out_Nnet_x
    val_Nnet_inp = np.concatenate((val_inp_Nnet_x, val_inp_Nnet_y), axis=1)
    val_Nnet_out = val_out_Nnet_x
    
    # Reshape data for network input
    val_Nnet_inp_resh = np.zeros((N_ex_val, 2 * 2 * L_ex))
    val_Nnet_out_resh = np.zeros((N_ex_val, 2 * N_d))
    inp_Nnet_resh = np.zeros((N_ex_tra, 2 * 2 * L_ex))
    out_Nnet_resh = np.zeros((N_ex_tra, 2 * N_d))
    for l in range(N_ex_tra):
        inp_Nnet_resh[l] = resh_data(inp_Nnet[l], 'real')
        out_Nnet_resh[l] = resh_data(out_Nnet[l], 'real')
    for l in range(N_ex_val):
        val_Nnet_inp_resh[l] = resh_data(val_Nnet_inp[l], 'real')
        val_Nnet_out_resh[l] = resh_data(val_Nnet_out[l], 'real')
    
    return inp_Nnet_resh, out_Nnet_resh, val_Nnet_inp_resh, val_Nnet_out_resh

def resh_data(Data, mapping):
    """
    Reshapes data between complex and real formats. If 'mapping' is 'real', it converts a complex array
    to a real array by interleaving real and imaginary parts. If 'mapping' is 'complex', it converts an
    interleaved real array back into a complex array.
    
    Parameters:
        Data (numpy.ndarray): The data to be reshaped. Can be complex or real depending on the mode.
        mapping (str): A mode specifier; "real" for complex to real transformation, 
                       "complex" for real to complex transformation.
                       
    Returns:
        numpy.ndarray: The reshaped data array.
    """
    L_data = len(Data)
    count = 0
    if mapping == "real":
        # Initialize a new numpy array with double the length of the input data
        X = np.zeros(2 * L_data)
        for l in range(L_data):
            X[count] = np.real(Data[l])
            count += 1
            X[count] = np.imag(Data[l])
            count += 1
    elif mapping == "complex":
        # Initialize a new numpy array with half the length of the input data
        X = np.zeros(int(L_data / 2), dtype=complex)
        for l in range(int(L_data / 2)):
            X[l] = Data[count] + 1j * Data[count + 1]
            count += 2
    return X

def matchfil(s, B, t, Ns):
    """
    Implements a matched filter using a sinc function for a given signal.
    
    Parameters:
        s (numpy.ndarray): The input signal to be filtered.
        B (float): The bandwidth of the sinc filter.
        t (numpy.ndarray): The time vector, should be the same length as the signal.
        Ns (int): The number of samples in the filter response, typically should cover the range of interest.
    
    Returns:
        numpy.ndarray: The filtered signal.
    """
    # Determine the range of l values for the filter kernel
    l1 = -int(np.floor(Ns / 2))
    l2 = int(np.ceil(Ns / 2))
    
    # Time step between samples
    dt = t[1] - t[0]
    
    # Initialize the result vector with complex zeros
    R = np.zeros(Ns, dtype=complex)
    
    # Normalize the sinc function to maintain energy levels
    norm = np.linalg.norm(np.sinc(B * t))
    
    # Compute the matched filter output for each shift of the sinc function
    for l in range(l1, l2):
        # Generate the sinc function shifted by l
        bas = np.sinc(B * t - l)
        
        # Multiply the input signal with the shifted sinc function and integrate (sum) over time
        R[l + int(np.floor(Ns / 2))] = np.sum(s * (bas * B * dt / norm))
    
    return R


def DBP_DM(signal, l_span, gamma, alpha_db, beta2, dt, B, NF_db, h, f0, N_s, t, steps):
    """
    Performs digital backpropagation for chromatic dispersion compensation in time domain.

    Parameters:
        signal (tuple): Tuple of arrays representing the X and Y polarizations of the input signal.
        l_span (array): Array of fiber spans (lengths).
        gamma (float): Nonlinear coefficient of the fiber.
        alpha_db (float): Attenuation coefficient in dB.
        beta2 (array): Second-order dispersion parameter.
        dt (float): Time step between samples.
        B (float): Bandwidth of the signal.
        NF_db (float): Noise figure in dB.
        h (float): Planck's constant.
        f0 (float): Frequency of the signal.
        N_s (int): Number of samples in the signal.
        t (array): Time array.
        steps (int): Number of steps for numerical integration.

    Returns:
        tuple: The compensated X and Y signals, and the half and full linear operators used in the compensation.
    """
    # Extract X and Y components of the signal
    signal_recx, signal_recy = signal

    # Calculate the number of samples and create the angular frequency vector
    l_s = len(signal_recx)
    w = 2 * np.pi / (l_s * dt) * fftshift(np.arange(-l_s / 2, l_s / 2))
    
    # Adjust number of spans if fewer steps are specified (combines multiple spans into one)
    if steps < 1:
        n_span = int(len(l_span) / 2 * steps)
        l_span2 = np.zeros(n_span * 2)
        for i in range(n_span):
            l_span2[2 * i] = np.sum(l_span[np.array([2 * j for j in range(i * int(1 / steps), (i + 1) * int(1 / steps))])])
            l_span2[2 * i + 1] = np.sum(l_span[np.array([2 * j + 1 for j in range(i * int(1 / steps), (i + 1) * int(1 / steps))])])
        l_span = l_span2
        steps = 1

    # Loop through each span
    for j in range(n_span):
        # Calculate step sizes for dispersion compensation
        step_s1 = l_span[-2 * j - 2] / steps * 1e3
        step_s2 = l_span[-2 * j - 1] / steps * 1e3
        
        cf = np.ones(steps) / np.mean(np.ones(steps))  # Correction factor to ensure energy conservation

        # Fourier transform the signal
        signal_recfx = fft(signal_recx)
        signal_recfy = fft(signal_recy)

        # Apply half-step dispersion compensation
        dis = np.exp((1j / 2) * beta2[-2 * j - 1] * (w ** 2) * l_span[-2 * j - 1] * 1e3)
        signal_recfx_i = signal_recfx * dis
        signal_recfy_i = signal_recfy * dis
        signal_recx = ifft(signal_recfx_i)
        signal_recy = ifft(signal_recfy_i)

        # Linear operators for half and full steps using the DFT matrix
        W = dft(l_s, scale='sqrtn')
        Winv = 1 / (l_s * W)
        D = np.diagflat(np.exp(1j / 2 * beta2[-2 * j - 2] * (w ** 2) * step_s1 / 2))
        halfLin_op = np.dot(Winv, np.dot(D, W))

        # Create convolution kernel for DBP based on DFT-based linear operator
        kernel = halfLin_op[256 - 1]  # Assuming specific index based on system parameters
        
        # Perform the convolution and nonlinear phase shift iteratively
        for jj in range(steps):
            concX = np.concatenate([signal_recx[int(l_s / 2):], signal_recx, signal_recx[:int(l_s / 2) - 1],])
            concY = np.concatenate([signal_recy[int(l_s / 2):], signal_recy, signal_recy[:int(l_s / 2) - 1],])
            signal_recx = np.convolve(concX, kernel, mode='valid')
            signal_recy = np.convolve(concY, kernel, mode='valid')
            signal_recx, signal_recy = signal_recx * np.exp(-1j * cf[jj] * gamma * ((abs(signal_recx)) ** 2 + (2 / 3) * (abs(signal_recy)) ** 2) * step_s1), signal_recy * np.exp(-1j * cf[jj] * gamma * ((abs(signal_recy)) ** 2 + (2 / 3) * (abs(signal_recx)) ** 2) * step_s1)
            
    return signal_recx, signal_recy, halfLin_op, fullLin_op

def DBP_DM2(signal, l_span, gamma, alpha_db, beta2, dt, B, NF_db, h, f0, N_s, t, steps):
    """
    Performs digital backpropagation for chromatic dispersion compensation using frequency domain operations.
    This function is specifically used for dispersion-managed systems and is intended for comparison with
    a time-domain version of DBP.

    Parameters:
        signal (tuple): Tuple of arrays representing the X and Y polarizations of the input signal.
        l_span (array): Array of fiber spans (lengths) adjusted for dispersion management.
        gamma (float): Nonlinear coefficient of the fiber.
        alpha_db (float): Attenuation coefficient in dB.
        beta2 (array): Second-order dispersion parameter.
        dt (float): Time step between samples.
        B (float): Bandwidth of the signal.
        NF_db (float): Noise figure in dB.
        h (float): Planck's constant.
        f0 (float): Frequency of the signal.
        N_s (int): Number of samples in the signal.
        t (array): Time array.
        steps (int): Number of steps for numerical integration, adjusting the precision of the operation.

    Returns:
        tuple: The compensated X and Y signals after backpropagation.
    """
    # Extract X and Y components of the signal
    signal_recx, signal_recy = signal

    # Calculate the number of samples and create the angular frequency vector
    l_s = len(signal_recx)
    w = 2 * np.pi / (l_s * dt) * fftshift(np.arange(-l_s / 2, l_s / 2))
    
    # Adjust number of spans based on the provided steps (for partial spans)
    n_span = int(len(l_span) / 2)
    if steps < 1:
        n_span = int(len(l_span) / 2 * steps)
        l_span2 = np.zeros(n_span * 2)
        for i in range(n_span):
            # Sum the spans proportionally based on the steps
            l_span2[2 * i] = np.sum(l_span[np.array([2 * j for j in range(i * int(1 / steps), (i + 1) * int(1 / steps))])])
            l_span2[2 * i + 1] = np.sum(l_span[np.array([2 * j + 1 for j in range(i * int(1 / steps), (i + 1) * int(1 / steps))])])
        l_span = l_span2
        steps = 1

    # Perform backpropagation through each span
    for j in range(n_span):
        # Calculate step sizes for the current span
        step_s1 = l_span[-2 * j - 2] / steps * 1e3
        step_s2 = l_span[-2 * j - 1] / steps * 1e3

        # Adjust for fiber losses
        cf = np.exp((np.log(10) / 10) * alpha_db * np.linspace(1 / (2 * steps), 1 - 1 / (2 * steps), steps, endpoint=True) * l_span[-2 * j - 2])
        cf = np.ones(steps)  # Corrective factors for uniform gain across the span
        cf = cf / np.mean(cf)

        # Apply dispersion compensation via FFT
        signal_recfx = fft(signal_recx)
        signal_recfy = fft(signal_recy)

        # Dispersion effect
        dis = np.exp((1j / 2) * beta2[-2 * j - 1] * (w ** 2) * l_span[-2 * j - 1] * 1e3)
        signal_recfx_i = signal_recfx * dis
        signal_recfy_i = signal_recfy * dis

        # Inverse FFT to convert back to the time domain
        signal_recx = ifft(signal_recfx_i)
        signal_recy = ifft(signal_recfy_i)

        # Apply half-step dispersion and nonlinear phase shift iteratively for each step
        halfdis = np.exp((1j / 2) * beta2[-2 * j - 2] * (w ** 2) * step_s1 / 2)
        for jj in range(steps):
            signal_recfx = fft(signal_recx)
            signal_recfy = fft(signal_recy)
            signal_recfx_i = signal_recfx * halfdis
            signal_recfy_i = signal_recfy * halfdis
            signal_recx = ifft(signal_recfx_i)
            signal_recy = ifft(signal_recfy_i)
            signal_recx = signal_recx * np.exp(-1j * cf[jj] * gamma * ((abs(signal_recx)) ** 2 + (2 / 3) * (abs(signal_recy)) ** 2) * step_s1)
            signal_recy = signal_recy * np.exp(-1j * cf[jj] * gamma * ((abs(signal_recy)) ** 2 + (2 / 3) * (abs(signal_recx)) ** 2) * step_s1)

    return signal_recx, signal_recy

def sym_error(A,B,bins):
    A1 = np.digitize(A[0].flatten(),bins) 
    B1 = np.digitize(B[0].flatten(),bins)
    A2 = np.digitize(A[1].flatten(),bins)
    B2 = np.digitize(B[1].flatten(),bins)
    errors = np.sum(np.logical_or(A1 != B1, A2 != B2))
    return errors

def examples(A, trim, length, mode=1):
    """
    Forms smaller arrays from a long vector A for use as neural network training examples, with optional modes for output formatting.

    Parameters:
        A (np.ndarray): Input array, expected to be of shape (2, n) where '2' could represent complex components.
        trim (int): Number of samples to discard from each edge of the input array.
        length (int): The length of each output example.
        mode (int): Determines the output sampling rate:
                    - mode 1: Retains the original sampling rate (no downsampling).
                    - mode 2: Downscales the data by half, effectively reducing the sampling rate to one sample per symbol.

    Returns:
        tuple: A tuple containing:
               - Out (list): List of arrays formatted according to the specified mode.
               - indx (np.ndarray): Array of original indices used to create the examples, adjusted for trim.
    """
    # Trim edges of the input array
    n = A.shape[1]
    A = A[:, trim:n-trim]
    n = A.shape[1]

    # Calculate indices for the center of each example
    indx = np.arange(int(length / 2), n - int(length / 2), 2)
    N_examples = len(indx)

    # Generate matrix of indices for gathering data segments
    indxmat = np.repeat(indx.reshape(N_examples, 1), length, axis=1) + \
              np.repeat(np.arange(-int(length / 2), int(length / 2), dtype='int').reshape(1, length), N_examples, axis=0)

    # Split the array into segments using advanced indexing
    Split = np.take(A, indxmat, axis=1)

    # Format the output based on the mode specifying sampling rate adjustments
    if mode == 1:
        # Mode 1: Retain the same sampling rate as the input
        Out = [np.real(Split[0]),
               np.imag(Split[0]),
               np.real(Split[1]),
               np.imag(Split[1])]
    elif mode == 2:
        # Mode 2: Downsample by taking every second sample, reducing the rate to one sample per symbol
        Out = [np.real(Split[0, :, ::2]),
               np.imag(Split[0, :, ::2]),
               np.real(Split[1, :, ::2]),
               np.imag(Split[1, :, ::2])]

    # Return the formatted data and the adjusted indices
    return Out, indx + trim

def examples2(A, trim, length):
    """
    Extracts smaller segments from a larger array A after trimming edge samples, useful for creating training examples.

    Parameters:
        A (np.ndarray): Input 2D array where each row typically represents different polarizations of complex data.
        trim (int): Number of samples to discard from the start and end of the input array.
        length (int): The desired length of each output segment.

    Returns:
        tuple:
            Out (np.ndarray): An array where each row is a training example formed by concatenating real and imaginary parts of the segments.
            indx (np.ndarray): Array of indices representing the starting points of each segment in the original array after trimming.
    """
    # Trim the edges of the array to remove unwanted samples
    n = A.shape[1]
    A = A[:, trim:n-trim]
    n = A.shape[1]

    # Compute indices for the center of each segment
    indx = np.arange(int(length / 2), n - int(length / 2), 1)
    N_examples = len(indx)

    # Generate a matrix of indices to select segments
    indxmat = np.repeat(indx.reshape(N_examples, 1), length, axis=1) + \
              np.repeat(np.arange(-int(length / 2), int(length / 2), dtype='int').reshape(1, length), N_examples, axis=0)

    # Use advanced indexing to extract segments
    Split = A[:, indxmat]

    # Format the output by concatenating real and imaginary parts
    Out = np.concatenate([
        np.real(Split[0]),  # Real part of the first component
        np.imag(Split[0]),  # Imaginary part of the first component
        np.real(Split[1]),  # Real part of the second component
        np.imag(Split[1])   # Imaginary part of the second component
    ], axis=1)

    # Return the formatted data and adjusted indices
    return Out, indx + trim


def initR(shape, dtype=None):
    """
    Initializes the real part of a filter for zero kilometer chromatic dispersion.

    Parameters:
        shape (tuple): The shape of the filter array to be returned.
        dtype (data-type, optional): Desired data-type for the array.

    Returns:
        np.ndarray: An array where the center of the array along the first dimension is set to 1, with all other values as 0.
    """
    d = shape[0]
    weights = np.zeros(shape)
    # Set the central value along the first dimension to 1, creating an impulse at the center
    weights[int(d/2),:,:] = 1
    return weights

def initI(shape, dtype=None):
    """
    Initializes the imaginary part of a filter for zero kilometer chromatic dispersion to all zeros.

    Parameters:
        shape (tuple): The shape of the filter array to be returned.
        dtype (data-type, optional): Desired data-type for the array, default is None which implies np.float64.

    Returns:
        np.ndarray: An array filled with zeros.
    """
    # Return an array of zeros matching the input shape
    return np.zeros(shape)

def filter_taps(shape, distance=20e3, mode='r', dtype=None):
    """
    Initializes filter taps based on a specified distance for chromatic dispersion compensation
    in an optical fiber. The filter is constructed in the frequency domain and transformed
    to the time domain.

    Parameters:
        shape (tuple): The shape of the filter array to be returned, typically (depth, height, width).
        distance (float): The distance in meters over which chromatic dispersion is to be compensated. Default is 20,000 meters.
        mode (str): Specifies the mode of the filter ('r' for real or any other string for imaginary).
        dtype (data-type, optional): Desired data-type for the array, default is None which implies np.float64.

    Returns:
        np.ndarray: A filter array shaped according to `shape`, containing the central slice of the filter matrix.
    """
    VecLen = shape[0]  # Length of the vector for DFT
    W = dft(VecLen, scale='sqrtn')  # DFT matrix with sqrt normalization
    Winv = 1 / (VecLen * W)  # Inverse DFT matrix scaled by vector length
    dt = 1 / (32e9 * 2)  # Time interval, assuming a sample rate of 32 GHz with oversampling
    f = t2f(np.arange(0, VecLen * dt, dt))  # Frequency vector
    f = fftshift(f)  # Shift zero frequency to center
    beta2 = -2.166761921076912e-26  # Dispersion parameter (typical value for SMF)
    H = 1j * beta2 / 2 * (2 * np.pi * f) ** 2  # Frequency response of the dispersion
    D = np.diagflat(np.exp(distance * H))  # Diagonal matrix for dispersion transformation
    A = np.dot(Winv, np.dot(D, W))  # Apply the inverse DFT to get the filter in time domain
    indx = int((VecLen - 1) / 2)  # Index for the central slice
    if mode == 'r':
        return np.real(A[indx]).reshape(shape)  # Return the real part if mode is 'r'
    return np.imag(A[indx]).reshape(shape)  # Otherwise, return the imaginary part

def average_groups(data, group_size=4):
    """
    Averages groups of elements in an array. This function divides an array into contiguous subarrays,
    each of size specified by 'group_size', and computes the mean of each subarray.

    Parameters:
        data (np.array): The input array to be averaged.
        group_size (int): The number of elements in each group to be averaged together. Default is 4.

    Returns:
        np.array: An array of the averaged values. The length of this array is the length of 'data'
                  divided by 'group_size', rounded down to the nearest integer.

    Example:
        If `data` = [1, 2, 3, 4, 5, 6, 7, 8] and `group_size` = 4,
        the function will return [2.5, 6.5] because:
        - The average of [1, 2, 3, 4] is 2.5
        - The average of [5, 6, 7, 8] is 6.5
    """
    # Calculate the number of groups
    num_groups = int(len(data) / group_size)
    # Initialize the output array with zeros
    averaged_data = np.zeros(num_groups)
    # Compute the mean for each group and store it in the output array
    for i in range(num_groups):
        averaged_data[i] = np.mean(data[i*group_size:(i*group_size + group_size)])
    return averaged_data


def kernel_init(width, step_size, dt, beta2, shape, mode='full'):
    """
    Initializes a convolutional kernel for a neural network that simulates chromatic dispersion
    compensation in optical communications. This kernel is based on the linear propagation model
    in the frequency domain, converted to time domain via IDFT.

    Parameters:
        width (int): The size of the Fourier transform (and hence the convolution kernel).
        step_size (float): The propagation step size in meters.
        dt (float): The time interval between samples.
        beta2 (float): The second-order dispersion parameter.
        shape (tuple): The shape of the output kernel tensor.
        mode (str): Specifies the operation mode; 'full' for a full-step dispersion compensation
                    or 'half' for a half-step. Default is 'full'.

    Returns:
        np.array: A complex-valued array shaped according to `shape`, representing the initialized
                  convolutional kernel for the specified mode.

    Explanation:
        - The function uses the discrete Fourier transform (DFT) matrices to construct linear operators
          for the dispersion effect modeled in the frequency domain.
        - Depending on the 'mode', it calculates either a half-step or full-step dispersion compensation.
        - This initial kernel can be used in model-based neural networks for fiber optic communication systems
          to simulate the inverse effects of chromatic dispersion over a specified step size.
    """
    # Calculate the Fourier and inverse Fourier transform matrices
    W = dft(width, scale='sqrtn')
    Winv = 1 / (width * W)

    # Frequency grid
    w = 2 * np.pi / (width * dt) * np.fft.fftshift(np.arange(-width / 2, width / 2))

    # Dispersion operator for half-step
    D_half = np.diagflat(np.exp(1j / 2 * beta2 * (w ** 2) * step_size / 2))
    halfLin_op = np.dot(Winv, np.dot(D_half, W))

    # Dispersion operator for full-step if required
    if mode == 'full':
        D_full = np.diagflat(np.exp(1j / 2 * beta2 * (w ** 2) * step_size))
        fullLin_op = np.dot(Winv, np.dot(D_full, W))
        return fullLin_op[int(width / 2)].reshape(shape)
    else:
        return halfLin_op[int(width / 2)].reshape(shape)

    
def cpe_rotation(q0_e,s,sr):
    q0_DBP = 0 + q0_e # recived signal.
    for l in range(2):
        for _ in range(8):
            a = np.concatenate([q0_DBP[l,0,::sr],q0_DBP[l,1,::sr],],axis=0)
            b = np.concatenate([s[l,0],s[l,1],],axis=0)
            Angle = np.mod(np.angle(a)-np.angle(b),2*np.pi)
            Angle2 = np.array([i for i in Angle if i<np.pi]+ [i-2*np.pi for i in Angle if i>np.pi])
            avgAngle = np.sum(Angle2 * abs(a))/np.sum(abs(a))
            q0_DBP[l,0],q0_DBP[l,1] = q0_DBP[l,0] * np.exp(-1j * avgAngle), q0_DBP[l,1] * np.exp(-1j * avgAngle)
    return q0_DBP

def BER_QAM16(expected, estimated):
    """
    Calculate the bit error rate (BER) for QAM16.
    
    Parameters:
        expected (np.array): The expected complex symbol array.
        estimated (np.array): The estimated complex symbol array received from the channel.

    Returns:
        float: The quality factor (Q-factor) in dB corresponding to the calculated BER.
    """
    # Define QAM16 constellation points normalized to unit power
    const = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    # Calculate thresholds for decision making
    thresholds = np.diff(const) / 2
    mid_points = const[:-1] + thresholds

    # Digitize and map the estimated symbols back to the closest constellation points
    real_decisions = np.digitize(np.real(estimated.flatten()), mid_points) * 2 * thresholds + const[0]
    imag_decisions = np.digitize(np.imag(estimated.flatten()), mid_points) * 2 * thresholds + const[0]
    s_dec = real_decisions + 1j * imag_decisions

    # Calculate symbol errors
    symbol_errors = np.sum(np.abs(s_dec - expected.flatten()) > 1e-8)

    # Convert symbols to bits and count bit errors
    b_dec = s2b(np.round(s_dec.flatten() * np.sqrt(10)))
    b_exc = s2b(np.round(expected.flatten() * np.sqrt(10)))
    bits_errors = np.sum(b_dec != b_exc)
    BER = bits_errors / len(b_exc)

    # Convert BER to Q-factor in dB
    Qfac_dB = Qfac(BER)
    return Qfac_dB

def BER_QAM64(expected, estimated):
    """
    Calculate the bit error rate (BER) for QAM64.
    
    Parameters:
        expected (np.array): The expected complex symbol array.
        estimated (np.array): The estimated complex symbol array received from the channel.

    Returns:
        float: The quality factor (Q-factor) in dB corresponding to the calculated BER.
    """
    # Define QAM64 constellation points normalized to unit power
    const = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)
    # Calculate thresholds for decision making
    thresholds = np.diff(const) / 2
    mid_points = const[:-1] + thresholds

    # Digitize and map the estimated symbols back to the closest constellation points
    real_decisions = np.digitize(np.real(estimated.flatten()), mid_points) * 2 * thresholds + const[0]
    imag_decisions = np.digitize(np.imag(estimated.flatten()), mid_points) * 2 * thresholds + const[0]
    s_dec = real_decisions + 1j * imag_decisions

    # Calculate symbol errors
    symbol_errors = np.sum(np.abs(s_dec - expected.flatten()) > 1e-8)

    # Convert symbols to bits and count bit errors
    b_dec = s2bQAM64(np.round(s_dec.flatten() * np.sqrt(42)))
    b_exc = s2bQAM64(np.round(expected.flatten() * np.sqrt(42)))
    bits_errors = np.sum(b_dec != b_exc)
    BER = bits_errors / len(b_exc)

    # Convert BER to Q-factor in dB
    Qfac_dB = Qfac(BER)
    return Qfac_dB


def examples_gen(A,length,step=1):
    n = np.shape(A)[1]
    indx = np.arange(int(length/2),n-int(length/2)+1,step)
    N_examples = len(indx)
    indxmat = np.repeat(indx.reshape(N_examples,1),length,axis = 1) + np.repeat(np.arange(int(-length/2),int(length/2),dtype='int').reshape(1,length),N_examples,axis=0)
    Split = np.expand_dims(A[:,indxmat],3)
    Out = [np.real(Split[0]),
           np.imag(Split[0])]
    return Out,indx

def examples_genXY(A,length,step=1):
    n = np.shape(A)[1]
    indx = np.arange(int(length/2),n-int(length/2)+1,step)
    N_examples = len(indx)
    indxmat = np.repeat(indx.reshape(N_examples,1),length,axis = 1) + np.repeat(np.arange(int(-length/2),int(length/2),dtype='int').reshape(1,length),N_examples,axis=0)
    Split = np.expand_dims(A[:,indxmat],3)
    Out = [np.real(Split[0]),
           np.imag(Split[0]),
           np.real(Split[1]),
           np.imag(Split[1]),
           ]
    return Out,indx

class Phase_function(tf.keras.layers.Layer):
    def __init__(self,initial):
        super(Phase_function, self).__init__()
        self.scale = initial
    def call(self, ReX,ImX,ReY,ImY):
        termX = ImX**2+ReX**2
        termY = ImY**2+ReY**2       
        Ang = self.scale*(termX + termY)    
        OutXr = tf.cos(Ang) * ReX - ImX * tf.sin(Ang)
        OutXi = tf.cos(Ang) * ImX + ReX * tf.sin(Ang)
        OutYr = tf.cos(Ang) * ReY - ImY * tf.sin(Ang)
        OutYi = tf.cos(Ang) * ImY + ReY * tf.sin(Ang)
        return OutXr,OutXi,OutYr,OutYi