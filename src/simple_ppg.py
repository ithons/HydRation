import os
import cv2
import glob
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from main import analyze_ppg_cycle

def butter_bandpass(low, high, fs, order=2):
    """Create Butterworth bandpass filter"""
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a

def analyze_video_chrom(args):
    """
    Extract PPG signal using CHROM method with heart rate mode.
    Trims first and last 3 seconds of video for stability.
    
    Args:
        args: Tuple of (video_path, mode)
        
    Returns:
        Dictionary containing:
        - bvp: Blood volume pulse signal
        - hr: Mean heart rate
        - hr_std: Heart rate standard deviation
        - timing: Time points
        - peaks: Peak locations
        - hydration_results: TPA/VPA analysis results
    """
    video_path, _ = args  # mode is always heart_rate
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Downsample to ~30 fps
        skip_frames = max(1, int(fps / 30))
        target_fps = fps / skip_frames
        
        # Skip first and last 3 seconds
        start_frame = int(3 * fps)
        end_frame = total_frames - int(3 * fps)
        
        if end_frame - start_frame < fps * 3:
            print(f"Warning: Video too short: {os.path.basename(video_path)}")
            return None
            
        signals = []
        times = []
        t = 0
        frame_count = 0
        
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if start_frame <= frame_count <= end_frame and frame_count % skip_frames == 0:
                # Select center region
                h, w = frame.shape[:2]
                roi = frame[h//4:3*h//4, w//4:3*w//4]
                
                # Extract color channels (BGR format)
                b = np.mean(roi[:, :, 0])
                g = np.mean(roi[:, :, 1])
                r = np.mean(roi[:, :, 2])
                
                # Heart rate mode CHROM
                x = 3 * g - 2 * r
                y = 1.5 * g + r - 1.5 * b
                
                signals.append([x, y])
                times.append(t)
                t += 1.0/target_fps
                
            frame_count += 1
            
        cap.release()
        
        if len(signals) < 10:
            return None
            
        # Process CHROM signals
        signals = np.array(signals)
        X = (signals[:, 0] - np.mean(signals[:, 0])) / np.std(signals[:, 0])
        Y = (signals[:, 1] - np.mean(signals[:, 1])) / np.std(signals[:, 1])
        
        # Combine signals
        alpha = np.std(X) / np.std(Y)
        ppg = X - alpha * Y
        
        # Bandpass filter (0.7-4.0 Hz)
        b, a = butter_bandpass(0.7, 4.0, target_fps)
        filtered_hr = filtfilt(b, a, ppg)
        
        # Derivative-based peak detection
        d1 = savgol_filter(filtered_hr, 9, 2, deriv=1)
        zero_crossings = np.where(np.diff(np.signbit(d1)))[0]
        
        peaks = []
        min_distance = int(target_fps * 60/120)  # Minimum distance for 120 BPM
        last_peak = -min_distance
        
        for i in range(len(zero_crossings)-1):
            idx = zero_crossings[i]
            if idx + 1 >= len(d1):
                continue
                
            if d1[idx] >= 0 and d1[idx+1] < 0:  # Peak: positive to negative
                if idx - last_peak >= min_distance:
                    peaks.append(idx)
                    last_peak = idx
        
        peaks = np.array(peaks)
        
        # Calculate heart rate
        if len(peaks) > 1:
            intervals = np.diff(peaks) / target_fps
            hr_inst = 60 / intervals
            hr_inst = hr_inst[(hr_inst >= 40) & (hr_inst <= 120)]  # Physiological range
            
            if len(hr_inst) > 0:
                hr_mean = np.mean(hr_inst)
                hr_std = np.std(hr_inst)
            else:
                hr_mean = hr_std = np.nan
        else:
            hr_mean = hr_std = np.nan
        
        # Calculate hydration metrics
        hydration_results = analyze_ppg_cycle(filtered_hr, target_fps)
        
        return {
            'bvp': filtered_hr,
            'hydration_signal': filtered_hr,
            'hr': hr_mean,
            'hr_std': hr_std,
            'timing': np.array(times),
            'peaks': peaks,
            'filename': os.path.basename(video_path),
            'hydration_results': hydration_results,
            'sampling_rate': target_fps
        }
            
    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
        return None

def analyze_videos():
    """Analyze all videos in the data/video directory using parallel processing"""
    video_files = []
    for ext in ['*.mov', '*.MOV', '*.mp4', '*.MP4']:
        video_files.extend(glob.glob(os.path.join('data/video', ext)))
    
    if not video_files:
        print("No video files found in data/video directory")
        return
        
    print(f"Found {len(video_files)} videos to analyze")
    
    # Process videos in parallel using heart rate mode
    args = [(video, 'heart_rate') for video in video_files]
    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = pool.map(analyze_video_chrom, args)
    
    # Plot results for each video
    for result in results:
        if result is None:
            continue
            
        print(f"\nResults for {result['filename']}:")
        hr = result['hr']
        hr_std = result['hr_std']
        hydration = result['hydration_results']
        
        print(f"Heart Rate: {hr:.1f} ± {hr_std:.1f} BPM")
        print(f"Hydration Metrics:")
        print(f"  TPA/VPA Ratio: {hydration['tpa_vpa_ratio']:.2f}")
        print(f"  Average TPA: {hydration['avg_tpa']:.2f}")
        print(f"  Average VPA: {hydration['avg_vpa']:.2f}")
        print(f"  Number of Peaks: {hydration['num_peaks']}")
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'PPG Analysis for {result["filename"]}', fontsize=16, y=0.95)
        
        # Heart rate plot
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(result['timing'], result['bvp'], 'b-', label='PPG Signal')
        ax1.plot(result['timing'][result['peaks']], 
                result['bvp'][result['peaks']], 'ro',
                label='Peaks')
        ax1.set_title(f'Heart Rate: {hr:.1f} ± {hr_std:.1f} BPM')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Hydration plot
        analyze_ppg_cycle(result['hydration_signal'], 
                         result['sampling_rate'],
                         show_plot=True,
                         window_length_sec=0.5)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    analyze_videos()
