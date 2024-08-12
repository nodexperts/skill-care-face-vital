import pywt
import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks
from .metrics import _reform_data_from_dict

# Adaptive Bandpass Filter
def bandpass_filter(signal, lowcut=0.5, highcut=4.0, fs=30, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Wavelet Denoising
def wavelet_denoise(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-level])) / 0.6745
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal


# Adaptive Peak Detection
def adaptive_peak_detection(signal, distance=30, prominence=0.5):
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    return peaks

# IBI Calculation
def calculate_ibi(peaks, fs):
    ibi = np.diff(peaks) / fs
    return ibi


# Time-Domain HRV Metrics
def time_domain_hrv(ibi):
    sdnn = np.std(ibi)  # Standard Deviation of NN intervals
    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))  # Root Mean Square of Successive Differences
    nn50 = np.sum(np.abs(np.diff(ibi)) > 0.05)  # Number of successive differences > 50ms
    pnn50 = (nn50 / len(ibi)) * 100  # Proportion of NN50 to total IBI
    return sdnn, rmssd, pnn50

# Frequency-Domain HRV Metrics
def frequency_domain_hrv(ibi, fs=30):
    f, pxx = welch(ibi, fs=fs, nperseg=len(ibi), scaling='density')
    lf_band = np.logical_and(f >= 0.04, f <= 0.15)
    hf_band = np.logical_and(f >= 0.15, f <= 0.4)
    lf = np.trapz(pxx[lf_band], f[lf_band])
    hf = np.trapz(pxx[hf_band], f[hf_band])
    lf_hf_ratio = lf / hf if hf != 0 else np.inf
    return lf, hf, lf_hf_ratio


def process_signal_for_hrv(predictions, fs=30):
    rppg_signal = _reform_data_from_dict(predictions)
    filtered_signal = bandpass_filter(rppg_signal, fs=fs)
    denoised_signal = wavelet_denoise(filtered_signal)
    peaks = adaptive_peak_detection(denoised_signal)
    ibi = calculate_ibi(peaks, fs)
    sdnn, rmssd, pnn50 = time_domain_hrv(ibi)
    lf, hf, lf_hf_ratio = frequency_domain_hrv(ibi, fs)
    return {
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'pNN50': pnn50,
        'LF': lf,
        'HF': hf,
        'LF/HF Ratio': lf_hf_ratio
    }