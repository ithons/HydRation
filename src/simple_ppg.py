import os
import cv2
import glob
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib import get_backend
from main import calculate_tpa_vpa_ratio, analyze_ppg_cycle

def butter_bandpass(low, high, fs, order=2):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def analyze_video_chrom(args):
    """
    Extract PPG signal using CHROM method with different channel combinations.
    Trims first and last 3 seconds of video.
    """
    video_path, mode = args
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip to achieve roughly 30 fps
        skip_frames = max(1, int(fps / 30))
        target_fps = fps / skip_frames
        
        # Calculate frames to skip at start and end (3 seconds)
        start_frame = int(3 * fps)
        end_frame = total_frames - int(3 * fps)
        
        if end_frame - start_frame < fps * 3:  # Require at least 3 seconds
            print(f"Warning: Video too short: {os.path.basename(video_path)}")
            return None
            
        signals = []
        times = []
        t = 0
        frame_count = 0
        
        # Store raw red channel for hydration calculation
        red_signal = []
        
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
                
                # Store raw red signal
                red_signal.append(r)
                
                # Different combinations for X and Y signals
                if mode == 'original':
                    x = 3 * r - 2 * g
                    y = 1.5 * r + g - 1.5 * b
                elif mode == 'heart_rate':
                    x = 3 * g - 2 * r
                    y = 1.5 * g + r - 1.5 * b
                elif mode == 'blood_volume':
                    x = 3 * r - g
                    y = 1.5 * r - 1.5 * b
                
                signals.append([x, y])
                times.append(t)
                t += 1.0/target_fps
                
            frame_count += 1
            
        cap.release()
        
        if len(signals) < 10:  # Need minimum number of samples
            return None
            
        signals = np.array(signals)
        X = signals[:, 0]
        Y = signals[:, 1]
        
        # Normalize
        X = (X - np.mean(X)) / np.std(X)
        Y = (Y - np.mean(Y)) / np.std(Y)
        
        # Alpha = std(X)/std(Y) to equalize noise
        alpha = np.std(X) / np.std(Y)
        ppg = X - alpha * Y
        
        # Bandpass filter for heart rate
        b, a = butter_bandpass(0.7, 4.0, target_fps)
        filtered_hr = filtfilt(b, a, ppg)
        
        # Calculate heart rate
        peaks_hr, _ = find_peaks(filtered_hr, distance=int(target_fps/3))
        
        if len(peaks_hr) > 1:
            intervals = np.diff(peaks_hr) / target_fps
            hr_inst = 60 / intervals
            hr_inst = hr_inst[(hr_inst >= 40) & (hr_inst <= 200)]
            
            if len(hr_inst) > 0:
                hr_mean = np.mean(hr_inst)
                hr_std = np.std(hr_inst)
            else:
                hr_mean = hr_std = np.nan
        else:
            hr_mean = hr_std = np.nan
        
        # Calculate hydration using TPA/VPA on the same filtered signal
        hydration_results = analyze_ppg_cycle(filtered_hr, target_fps)
        
        return {
            'bvp': filtered_hr,
            'hydration_signal': filtered_hr,  # Use same signal for both
            'hr': hr_mean,
            'hr_std': hr_std,
            'timing': np.array(times),
            'peaks': peaks_hr,
            'mode': mode,
            'filename': os.path.basename(video_path),
            'hydration_results': hydration_results,
            'sampling_rate': target_fps
        }
            
    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)} with CHROM ({mode}): {str(e)}")
        return None

def analyze_videos():
    """Analyze all videos in the data/video directory using parallel processing"""
    # Get all video files
    video_dir = 'data/video'
    video_patterns = ['*.mov', '*.MOV', '*.mp4', '*.MP4']
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(video_dir, pattern)))
    
    if not video_files:
        print("No video files found in data/video directory")
        return
        
    # Find the specific file
    target_file = '1739692066788000.mov'
    video_files = [f for f in video_files if os.path.basename(f) == target_file]
    
    if not video_files:
        print(f"Could not find {target_file} in data/video directory")
        return
        
    print(f"Analyzing file: {video_files[0]}")
    
    # Prepare arguments for parallel processing
    modes = ['blood_volume']  # Only use blood volume mode for hydration
    args = [(video, mode) for video in video_files for mode in modes]
    
    # Process videos in parallel
    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = pool.map(analyze_video_chrom, args)
    
    # Group results by video
    video_results = {}
    for result in results:
        if result is not None:
            filename = result['filename']
            if filename not in video_results:
                video_results[filename] = {}
            video_results[filename][result['mode']] = result
    
    # Plot results
    for filename, modes in video_results.items():
        if len(modes) == 0:
            continue
            
        print(f"\nResults for {filename}:")
        
        for mode, results in modes.items():
            hr = results['hr']
            hr_std = results['hr_std']
            hydration = results['hydration_results']
            
            print(f"Heart Rate: {hr:.1f} ± {hr_std:.1f} BPM")
            print(f"Hydration Metrics:")
            print(f"  TPA/VPA Ratio: {hydration['tpa_vpa_ratio']:.2f}")
            print(f"  Average TPA: {hydration['avg_tpa']:.2f}")
            print(f"  Average VPA: {hydration['avg_vpa']:.2f}")
            print(f"  Number of Peaks: {hydration['num_peaks']}")
            
            # Create figure with 2 subplots
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f'PPG Analysis for {filename}', fontsize=16, y=0.95)
            
            # Heart rate plot
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(results['timing'], results['bvp'], 'b-', label='PPG Signal')
            ax1.plot(results['timing'][results['peaks']], 
                    results['bvp'][results['peaks']], 'ro',
                    label='Peaks')
            ax1.set_title(f'Heart Rate: {hr:.1f} ± {hr_std:.1f} BPM')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Hydration plot
            analyze_ppg_cycle(results['hydration_signal'], 
                            results['sampling_rate'],
                            show_plot=True,
                            window_length_sec=0.5)  # Back to original 0.5-second window
            
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    analyze_videos()
