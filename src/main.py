import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp
import pandas as pd
import os
from glob import glob

def fiducial_points(x, pks, fs, vis):
    """
    Description: Pulse detection and correction from pulsatile signals
    """    
    # First, second and third derivatives
    d1x = sp.savgol_filter(x, 9, 5, deriv=1) 
    d2x = sp.savgol_filter(x, 9, 5, deriv=2) 
    d3x = sp.savgol_filter(x, 9, 5, deriv=3) 
    
    # Find diastolic valleys between consecutive peaks
    dia = []
    dic = []
    
    for i in range(len(pks)-1):
        # Define search window between current and next peak
        start_idx = pks[i]
        end_idx = pks[i+1]
        
        # Search for valley in the window
        window = x[start_idx:end_idx]
        if len(window) < 3:  # Skip if window too small
            continue
            
        # Find valley as the minimum point
        valley_idx = start_idx + np.argmin(window)
        
        # Validate valley using first derivative
        if valley_idx > 0 and valley_idx < len(d1x)-1:
            if d1x[valley_idx-1] < 0 and d1x[valley_idx+1] > 0:  # Zero crossing
                dia.append(valley_idx)
                
                # Find dicrotic notch between peak and valley
                notch_window = d2x[start_idx:valley_idx]
                if len(notch_window) > 3:
                    notch_candidates, _ = sp.find_peaks(-notch_window)
                    if len(notch_candidates) > 0:
                        # Take the most prominent notch
                        notch_idx = start_idx + notch_candidates[np.argmax(notch_window[notch_candidates])]
                        # Validate notch position
                        if start_idx < notch_idx < valley_idx:
                            dic.append(notch_idx)
    
    dia = np.array(dia, dtype=int)
    dic = np.array(dic, dtype=int)
    
    # Creation of dictionary with relevant points
    fidp = {
        'pks': pks.astype(int),
        'dia': dia,
        'dic': dic
    }
    
    # Visualize if requested
    if vis:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Create time axis in seconds
        time = np.arange(len(x)) / fs
        
        # Plot signal with fiducial points
        ax1.plot(time, x, color='black', label='PPG Signal')
        ax1.scatter(pks/fs, x[pks], color='red', label='Systolic Peaks')
        ax1.scatter(dia/fs, x[dia], color='blue', label='Diastolic Valleys')
        ax1.scatter(dic/fs, x[dic], color='green', label='Dicrotic Notches')
        ax1.legend()
        ax1.set_title('PPG Signal with Fiducial Points')
        ax1.set_ylabel('Amplitude')
        
        # Plot derivatives
        ax2.plot(time, d1x, label='First Derivative')
        ax2.plot(time, d2x, label='Second Derivative')
        ax2.legend()
        ax2.set_title('Signal Derivatives')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.show()
    
    return fidp

def calculate_tpa_vpa_ratio(signal, fidp):
    """
    Calculate TPA/VPA ratio using detected fiducial points
    """
    tpa_list = []
    vpa_list = []
    
    peaks = fidp['pks']
    valleys = fidp['dia']
    notches = fidp['dic']
    
    # Print the number of points detected
    print(f"Detected: {len(peaks)} peaks, {len(valleys)} valleys, {len(notches)} notches")
    
    # Get the minimum length to avoid index errors
    min_length = min(len(peaks)-1, len(valleys), len(notches))
    
    for i in range(min_length):
        try:
            # Get points for current cycle
            peak = peaks[i]
            valley = valleys[i]
            notch = notches[i]
            
            # Calculate amplitudes
            tpa = signal[peak] - signal[notch]
            vpa = signal[peak] - signal[valley]
            
            # Only validate that VPA is not zero to avoid division by zero
            if vpa != 0:
                tpa_list.append(tpa)
                vpa_list.append(vpa)
                
        except IndexError:
            continue
    
    # Calculate mean ratio if we have valid measurements
    if len(tpa_list) >= 1:  # Require at least 1 valid cycle
        ratios = np.array(tpa_list) / np.array(vpa_list)
        return np.mean(ratios)
    else:
        print("Warning: Not enough valid pulse waves detected")
        return 0.0

def analyze_ppg(signal, fs):
    """
    Analyze PPG signal to calculate TPA/VPA ratio
    """
    # Handle NaN values by interpolation
    if np.any(np.isnan(signal)):
        nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
        signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
    
    # Preprocess signal with bandpass filter
    nyq = fs/2
    low = 0.5/nyq
    high = 8.0/nyq
    b, a = sp.butter(2, [low, high], btype='band')
    signal = sp.filtfilt(b, a, signal)
    
    # Remove baseline wander
    window = int(2 * fs)  # 2 second window
    baseline = sp.savgol_filter(signal, window_length=window, polyorder=2)
    signal = signal - baseline
    
    # Normalize signal
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    # Find initial peaks with physiological constraints
    min_distance = int(fs * 60/180)  # Maximum 180 BPM
    max_distance = int(fs * 60/40)   # Minimum 40 BPM
    
    # Calculate first derivative
    d1 = sp.savgol_filter(signal, 9, 2, deriv=1)
    
    # Find zero crossings in first derivative (potential peaks)
    zero_crossings = np.where(np.diff(np.signbit(d1)))[0]
    
    # Filter zero crossings to only get negative-to-positive (peaks)
    peak_candidates = zero_crossings[d1[zero_crossings] >= 0]
    
    # Calculate prominence for each candidate
    prominences = []
    for peak in peak_candidates:
        # Look for valleys on both sides
        left_idx = max(0, peak - min_distance)
        right_idx = min(len(signal), peak + min_distance)
        
        left_min = np.min(signal[left_idx:peak])
        right_min = np.min(signal[peak:right_idx])
        
        # Calculate prominence as height above higher of the two valleys
        prominence = signal[peak] - max(left_min, right_min)
        prominences.append(prominence)
    
    prominences = np.array(prominences)
    
    # Filter peaks by prominence and minimum distance
    valid_peaks = []
    last_peak = -min_distance  # Initialize with impossible peak position
    
    for idx, (peak, prom) in enumerate(zip(peak_candidates, prominences)):
        if prom > 0.05 and peak - last_peak >= min_distance:
            valid_peaks.append(peak)
            last_peak = peak
    
    valid_peaks = np.array(valid_peaks)
    
    if len(valid_peaks) > 2:  # Need at least 3 peaks for analysis
        # Get fiducial points
        fidp = fiducial_points(signal, valid_peaks, fs, vis=True)
        
        # Calculate ratio
        ratio = calculate_tpa_vpa_ratio(signal, fidp)
        return ratio
    
    print("Warning: Could not detect valid peaks in the signal")
    return 0.0

def normalize_with_acceleration(signal, accel_x, accel_y, accel_z, fs):
    """
    Normalize PPG signal using acceleration data through adaptive filtering
    """
    # Convert all inputs to float arrays
    signal = np.array(signal, dtype=float)
    accel_x = np.array(accel_x, dtype=float)
    accel_y = np.array(accel_y, dtype=float)
    accel_z = np.array(accel_z, dtype=float)
    
    # Calculate acceleration magnitude
    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    
    # Normalize acceleration components
    accel_x = accel_x / np.std(accel_x)
    accel_y = accel_y / np.std(accel_y)
    accel_z = accel_z / np.std(accel_z)
    
    # Create acceleration feature matrix
    X = np.vstack([accel_x, accel_y, accel_z, np.ones_like(accel_x)]).T
    
    # Estimate motion artifacts using least squares regression
    beta = np.linalg.lstsq(X, signal, rcond=None)[0]
    motion_estimate = X @ beta
    
    # Subtract motion artifacts
    clean_signal = signal - motion_estimate
    
    # Apply bandpass filter to remove any remaining noise
    nyq = fs/2
    low = 0.5/nyq
    high = 8.0/nyq
    b, a = sp.butter(2, [low, high], btype='band')
    clean_signal = sp.filtfilt(b, a, clean_signal)
    
    # Normalize the clean signal
    clean_signal = (clean_signal - np.mean(clean_signal)) / np.std(clean_signal)
    
    return clean_signal

def analyze_ppg_with_motion(signal, accel_x, accel_y, accel_z, fs):
    """
    Analyze PPG signal with motion artifact detection and normalization
    """
    # First, normalize the signal using acceleration data
    clean_signal = normalize_with_acceleration(signal, accel_x, accel_y, accel_z, fs)
    
    # Plot original vs cleaned signal
    plt.figure(figsize=(12, 8))
    
    # Create time axis in seconds
    time = np.arange(len(signal)) / fs
    
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label='Original')
    plt.title('Original PPG Signal')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time, clean_signal, label='Normalized')
    plt.title('Normalized PPG Signal')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(time, accel_x, label='X')
    plt.plot(time, accel_y, label='Y')
    plt.plot(time, accel_z, label='Z')
    plt.title('Acceleration Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (g)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Analyze the normalized signal
    return analyze_ppg(clean_signal, fs)

def detect_peaks_fixed(signal, fs):
    """Original method with fixed thresholds"""
    min_distance = int(fs * 60/180)  # Maximum 180 BPM
    peaks, properties = sp.find_peaks(signal,
                                    distance=min_distance,
                                    height=0.2,
                                    prominence=0.05)
    return peaks[properties['prominences'] > 0.05]

def detect_peaks_derivative(signal, fs):
    """Derivative-based peak detection"""
    # Calculate derivatives
    d1 = sp.savgol_filter(signal, 9, 2, deriv=1)
    d2 = sp.savgol_filter(signal, 9, 2, deriv=2)
    
    # First find all potential peaks (local maxima)
    peaks, _ = sp.find_peaks(signal, distance=int(fs * 60/180))  # At least 60/180 seconds apart
    
    valid_peaks = []
    valleys = []
    notches = []
    
    # For each peak, find the corresponding valley and notch
    for i in range(len(peaks)-1):
        peak = peaks[i]
        next_peak = peaks[i+1]
        
        # Find valley (minimum between peaks)
        valley_region = signal[peak:next_peak]
        if len(valley_region) > 3:
            valley = peak + np.argmin(valley_region)
            
            # Look for dicrotic notch between peak and valley
            notch_region = signal[peak:valley]
            if len(notch_region) > 10:  # Need enough points to find notch
                # Dicrotic notch should be a local minimum in first derivative
                d1_segment = d1[peak:valley]
                notch_candidates, _ = sp.find_peaks(-d1_segment)
                
                if len(notch_candidates) > 0:
                    # Take the most prominent notch that's lower than the peak
                    valid_notches = []
                    for nc in notch_candidates:
                        notch_idx = peak + nc
                        # Check if notch is lower than peak and higher than valley
                        if (signal[notch_idx] < signal[peak] and 
                            signal[notch_idx] > signal[valley]):
                            valid_notches.append((notch_idx, signal[peak] - signal[notch_idx]))
                    
                    if valid_notches:
                        # Take the notch with largest drop from peak
                        notch = max(valid_notches, key=lambda x: x[1])[0]
                        valid_peaks.append(peak)
                        valleys.append(valley)
                        notches.append(notch)
    
    return np.array(valid_peaks), np.array(valleys), np.array(notches)

def evaluate_detection(signal, peaks, valleys, notches, fs):
    """Evaluate peak detection quality with proper TPA/VPA calculation"""
    # Calculate expected number of peaks based on typical heart rate range
    duration = len(signal) / fs
    min_expected = int(duration * 40/60)  # 40 BPM
    max_expected = int(duration * 180/60)  # 180 BPM
    
    # Calculate TPA and VPA for each cycle
    tpa_list = []  # Total Pulse Area
    vpa_list = []  # Valley-to-Peak Area
    
    min_length = min(len(peaks), len(valleys), len(notches))
    for i in range(min_length):
        peak = peaks[i]
        valley = valleys[i]
        notch = notches[i]
        
        if signal[peak] > signal[notch] > signal[valley]:
            # Calculate amplitudes
            tpa = signal[peak] - signal[notch]      # Total Pulse Amplitude
            vpa = signal[peak] - signal[valley]      # Valley-to-Peak Amplitude
            
            if vpa > 0:  # Ensure valid amplitude
                tpa_list.append(tpa)
                vpa_list.append(vpa)
    
    # Calculate TPA/VPA ratio
    if len(tpa_list) > 0:
        tpa_vpa_ratio = np.mean(np.array(tpa_list) / np.array(vpa_list))
    else:
        tpa_vpa_ratio = 0
        
    return {
        'num_peaks': len(peaks),
        'tpa_vpa_ratio': tpa_vpa_ratio,
        'avg_tpa': np.mean(tpa_list) if len(tpa_list) > 0 else 0,
        'avg_vpa': np.mean(vpa_list) if len(vpa_list) > 0 else 0,
        'within_expected': min_expected <= len(peaks) <= max_expected
    }

def analyze_ppg_cycle(signal, fs, show_plot=False):
    """Analyze PPG signal using area-based calculations"""
    # Find main peaks with lower prominence to catch more peaks
    peaks, _ = sp.find_peaks(signal, distance=int(fs * 60/180), prominence=0.05)
    
    # Find main valleys
    valleys, _ = sp.find_peaks(-signal, distance=int(fs * 60/180), prominence=0.05)
    
    # Ensure we have alternating peaks and valleys
    valid_peaks = []
    valid_valleys = []
    
    # Start with the first valley
    if len(peaks) > 0 and len(valleys) > 0:
        # Find first valley before first peak
        first_peak = peaks[0]
        first_valley_idx = 0
        while first_valley_idx < len(valleys) and valleys[first_valley_idx] < first_peak:
            first_valley_idx += 1
        
        if first_valley_idx > 0:  # We found a valley before the first peak
            first_valley_idx -= 1
            current_valley = valleys[first_valley_idx]
            valley_idx = first_valley_idx + 1
            
            for peak_idx in range(len(peaks)-1):
                current_peak = peaks[peak_idx]
                next_peak = peaks[peak_idx + 1]
                
                # Find next valley between current peak and next peak
                while valley_idx < len(valleys) and valleys[valley_idx] <= current_peak:
                    valley_idx += 1
                    
                if valley_idx >= len(valleys):
                    break
                    
                next_valley = valleys[valley_idx]
                
                # If we have a valid valley-peak-valley sequence
                if current_valley < current_peak < next_valley < next_peak:
                    valid_peaks.append(current_peak)
                    valid_valleys.append(current_valley)
                    current_valley = next_valley
            
            # Add the last valley if we have one
            if valley_idx < len(valleys):
                valid_valleys.append(valleys[valley_idx])
    
    if len(valid_peaks) < 2 or len(valid_valleys) < 2:
        return None
        
    valid_peaks = np.array(valid_peaks)
    valid_valleys = np.array(valid_valleys)
    
    # Create continuous lower envelope across all valleys
    # Add extra valleys at the start and end to prevent edge effects
    from scipy.interpolate import interp1d
    
    # Add extra valleys at edges if needed
    all_valleys = list(valid_valleys)
    if valid_valleys[0] > valid_peaks[0]:
        # Add valley before first peak using linear extrapolation
        valley_before = valid_valleys[0] - (valid_valleys[1] - valid_valleys[0])
        valley_height = signal[valid_valleys[0]] - (signal[valid_valleys[1]] - signal[valid_valleys[0]])
        all_valleys.insert(0, valley_before)
        
    if valid_valleys[-1] < valid_peaks[-1]:
        # Add valley after last peak using linear extrapolation
        valley_after = valid_valleys[-1] + (valid_valleys[-1] - valid_valleys[-2])
        valley_height = signal[valid_valleys[-1]] - (signal[valid_valleys[-1]] - signal[valid_valleys[-2]])
        all_valleys.append(valley_after)
    
    all_valleys = np.array(all_valleys)
    valley_interpolator = interp1d(all_valleys, signal[all_valleys], kind='cubic', 
                                 fill_value='extrapolate')
    
    # Calculate areas for each cycle
    tpa_list = []  # Total Pulse Area (under curve)
    vpa_list = []  # Valley-to-Peak Area (over curve)
    
    if show_plot:
        plt.figure(figsize=(15, 10))
        
        # Plot signal and detected points
        plt.subplot(2, 1, 1)
        time = np.arange(len(signal)) / fs
        plt.plot(time, signal, label='PPG Signal')
        plt.plot(valid_peaks/fs, signal[valid_peaks], 'ro', label='Peaks')
        plt.plot(valid_valleys/fs, signal[valid_valleys], 'go', label='Valleys')
    
    # Calculate areas and shade them
    for i in range(len(valid_peaks)-1):
        peak = valid_peaks[i]
        next_peak = valid_peaks[i+1]
        
        # Get all points between peaks
        t_segment = np.arange(peak, next_peak+1)
        signal_segment = signal[peak:next_peak+1]
        
        # Create upper envelope by connecting peaks
        upper_envelope = np.linspace(signal[peak], signal[next_peak], len(t_segment))
        
        # Get lower envelope values for this segment using the continuous interpolator
        lower_envelope = valley_interpolator(t_segment)
        
        # Calculate VPA (area between upper envelope and curve)
        vpa = abs(np.trapezoid(upper_envelope - signal_segment, dx=1/fs))
        
        # Calculate TPA (area between curve and lower envelope)
        tpa = abs(np.trapezoid(signal_segment - lower_envelope, dx=1/fs))
        
        if vpa > 0 and tpa > 0:  # Ensure valid areas
            tpa_list.append(tpa)
            vpa_list.append(vpa)
            
            if show_plot:
                # Shade VPA - area between upper envelope and curve
                plt.fill_between(time[peak:next_peak+1], 
                               upper_envelope,
                               signal_segment,
                               alpha=0.3, color='blue', 
                               label='VPA' if i==0 else "")
                
                # Shade TPA - area between curve and lower envelope
                plt.fill_between(time[peak:next_peak+1],
                               signal_segment,
                               lower_envelope,
                               alpha=0.3, color='red', 
                               label='TPA' if i==0 else "")
                
                # Plot envelopes
                plt.plot(time[peak:next_peak+1], upper_envelope, 'b--', alpha=0.5, label='Upper Envelope' if i==0 else "")
                plt.plot(time[peak:next_peak+1], lower_envelope, 'r--', alpha=0.5, label='Lower Envelope' if i==0 else "")
    
    # Calculate ratio
    if len(tpa_list) > 0:
        tpa_vpa_ratio = np.mean(np.array(tpa_list) / np.array(vpa_list))
    else:
        tpa_vpa_ratio = 0
        
    if show_plot:
        plt.title(f'PPG Analysis - TPA/VPA Ratio: {tpa_vpa_ratio:.3f}')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.grid(True)
        
        # Plot derivatives for reference
        plt.subplot(2, 1, 2)
        d1 = sp.savgol_filter(signal, 9, 2, deriv=1)
        d2 = sp.savgol_filter(signal, 9, 2, deriv=2)
        plt.plot(time, d1, label='First Derivative')
        plt.plot(time, d2, label='Second Derivative')
        plt.title('Signal Derivatives')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Derivative')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'num_peaks': len(valid_peaks),
        'tpa_vpa_ratio': tpa_vpa_ratio,
        'avg_tpa': np.mean(tpa_list) if len(tpa_list) > 0 else 0,
        'avg_vpa': np.mean(vpa_list) if len(vpa_list) > 0 else 0
    }

def analyze_file(filepath, show_plot=False):
    """Analyze a single file"""
    try:
        # Read data
        df = pd.read_csv(filepath)
        green = df.iloc[:, 0].values
        fs = 100
        
        # Preprocess signal
        green = (green - np.min(green)) / (np.max(green) - np.min(green))
        
        # Analyze signal
        results = analyze_ppg_cycle(green, fs, show_plot)
        return results
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def main():
    # Get list of CSV files
    data_dir = 'data/regular'
    csv_files = glob(os.path.join(data_dir, '*.csv'))
    
    # Analyze all files
    file_results = []  # List of (filepath, results) tuples
    all_ratios = []
    
    for filepath in csv_files:
        results = analyze_file(filepath, show_plot=False)  # Don't show plots yet
        
        if results and results['tpa_vpa_ratio'] > 0:  # Only include valid ratios
            file_results.append((filepath, results))
            all_ratios.append(results['tpa_vpa_ratio'])
    
    # Sort by TPA/VPA ratio to find outliers
    file_results.sort(key=lambda x: x[1]['tpa_vpa_ratio'], reverse=True)
    
    # Show the top 5 outliers
    print("\nTop 5 Outliers:")
    for filepath, results in file_results[:5]:
        print(f"\nAnalyzing outlier: {os.path.basename(filepath)}")
        print(f"TPA/VPA Ratio: {results['tpa_vpa_ratio']:.3f}")
        # Re-analyze with plots
        analyze_file(filepath, show_plot=True)
    
    # Print overall statistics
    print("\nOverall Statistics:")
    ratios = np.array(all_ratios)
    print(f"Number of valid samples: {len(ratios)}")
    print(f"Average TPA/VPA Ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    print(f"Range: {np.min(ratios):.3f} to {np.max(ratios):.3f}")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(ratios, bins=30, edgecolor='black')
    plt.title('Distribution of TPA/VPA Ratios in Regular PPG Signals')
    plt.xlabel('TPA/VPA Ratio')
    plt.ylabel('Frequency')
    
    # Add mean and std lines
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    plt.axvline(mean_ratio, color='r', linestyle='--', label=f'Mean: {mean_ratio:.3f}')
    plt.axvline(mean_ratio + std_ratio, color='g', linestyle=':', label=f'Mean ± SD: {mean_ratio:.3f} ± {std_ratio:.3f}')
    plt.axvline(mean_ratio - std_ratio, color='g', linestyle=':')
    
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()