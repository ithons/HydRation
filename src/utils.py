import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

def analyze_ppg_cycle(signal, fs, show_plot=False, window_length_sec=0.5):
    """
    Analyze PPG signal using area-based calculations for hydration metrics.
    
    Args:
        signal: PPG signal array
        fs: Sampling frequency in Hz
        show_plot: Whether to show visualization plots
        window_length_sec: Window length for baseline removal in seconds
        
    Returns:
        Dictionary containing:
        - num_peaks: Number of peaks detected
        - tpa_vpa_ratio: Ratio of Total Pulse Area to Valley-to-Peak Area
        - avg_tpa: Average Total Pulse Area
        - avg_vpa: Average Valley-to-Peak Area
    """
    # Create smoothed baseline for normalization
    window_length = int(fs * window_length_sec)
    if window_length % 2 == 0:  # Make window length odd
        window_length += 1
    baseline = sp.savgol_filter(signal, window_length, 2)
    
    # Normalize signal by subtracting baseline
    normalized_signal = signal - baseline
    normalized_signal = sp.savgol_filter(normalized_signal, 5, 2)
    
    # Find peaks and valleys using first derivative
    d1 = sp.savgol_filter(normalized_signal, 9, 2, deriv=1)
    zero_crossings = np.where(np.diff(np.signbit(d1)))[0]
    
    peaks = []
    valleys = []
    min_distance = int(fs * 60/360)  # Minimum distance between peaks
    
    last_peak = -min_distance
    last_valley = -min_distance
    
    for i in range(len(zero_crossings)-1):
        idx = zero_crossings[i]
        if idx + 1 >= len(d1):
            continue
            
        if d1[idx] >= 0 and d1[idx+1] < 0:  # Peak: positive to negative
            if idx - last_peak >= min_distance:
                peaks.append(idx)
                last_peak = idx
        elif d1[idx] < 0 and d1[idx+1] >= 0:  # Valley: negative to positive
            if idx - last_valley >= min_distance:
                valleys.append(idx)
                last_valley = idx
    
    peaks = np.array(peaks)
    valleys = np.array(valleys)
    
    if len(peaks) < 2 or len(valleys) < 2:
        return None
    
    # Create envelopes
    time_points = np.arange(len(normalized_signal))
    upper_envelope = np.interp(time_points, peaks, normalized_signal[peaks])
    lower_envelope = np.interp(time_points, valleys, normalized_signal[valleys])
    
    # Calculate areas
    tpa_list = []  # Total Pulse Area
    vpa_list = []  # Valley-to-Peak Area
    
    if show_plot:
        plt.figure(figsize=(15, 10))
        time = np.arange(len(signal)) / fs
        
        plt.subplot(2, 1, 1)
        plt.plot(time, normalized_signal, label='Normalized PPG Signal')
        plt.plot(peaks/fs, normalized_signal[peaks], 'ro', label='Peaks')
        plt.plot(valleys/fs, normalized_signal[valleys], 'go', label='Valleys')
        plt.plot(time, upper_envelope, 'b--', alpha=0.5, label='Upper Envelope')
        plt.plot(time, lower_envelope, 'r--', alpha=0.5, label='Lower Envelope')
    
    for i in range(len(peaks)-1):
        peak = peaks[i]
        next_peak = peaks[i+1]
        
        # Get segment between peaks
        t_segment = np.arange(peak, next_peak+1)
        signal_segment = normalized_signal[peak:next_peak+1]
        upper_segment = upper_envelope[peak:next_peak+1]
        lower_segment = lower_envelope[peak:next_peak+1]
        
        # Calculate areas
        vpa = abs(np.trapezoid(upper_segment - signal_segment, dx=1/fs))
        tpa = abs(np.trapezoid(signal_segment - lower_segment, dx=1/fs))
        
        if vpa > 0 and tpa > 0:
            tpa_list.append(tpa)
            vpa_list.append(vpa)
            
            if show_plot:
                plt.fill_between(time[peak:next_peak+1], 
                               upper_segment,
                               signal_segment,
                               alpha=0.3, color='blue', 
                               label='VPA' if i==0 else "")
                plt.fill_between(time[peak:next_peak+1],
                               signal_segment,
                               lower_segment,
                               alpha=0.3, color='red', 
                               label='TPA' if i==0 else "")
    
    tpa_vpa_ratio = np.mean(np.array(tpa_list) / np.array(vpa_list)) if len(tpa_list) > 0 else 0
        
    if show_plot:
        plt.title(f'PPG Analysis - TPA/VPA Ratio: {tpa_vpa_ratio:.3f}')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Signal')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(time, d1, label='First Derivative')
        plt.title('Signal Derivatives')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Derivative')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'num_peaks': len(peaks),
        'tpa_vpa_ratio': tpa_vpa_ratio,
        'avg_tpa': np.mean(tpa_list) if len(tpa_list) > 0 else 0,
        'avg_vpa': np.mean(vpa_list) if len(vpa_list) > 0 else 0
    }

def classify_hydration(tpa_vpa_ratio):
    """
    Classify hydration level based on TPA/VPA ratio.
    
    Args:
        tpa_vpa_ratio: TPA/VPA ratio from PPG analysis
        
    Returns:
        String indicating hydration classification:
        - 'Severe Dehydration': ratio < 0.559
        - 'Mild Dehydration': 0.559 <= ratio < 0.815
        - 'Normal': 0.815 <= ratio <= 1.326
        - 'Overhydration': ratio > 1.326
    """
    if tpa_vpa_ratio < 0.559:
        return 'Severe Dehydration'
    elif tpa_vpa_ratio < 0.815:
        return 'Mild Dehydration'
    elif tpa_vpa_ratio <= 1.326:
        return 'Normal'
    else:
        return 'Overhydration'
