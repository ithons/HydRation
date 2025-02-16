import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from src.simple_ppg import analyze_video_chrom
import pandas as pd
import os
from glob import glob
from src.utils import analyze_ppg_cycle, classify_hydration

def analyze_file(filepath, show_plot=False):
    """Analyze a single file"""
    try:
        # Read data
        df = pd.read_csv(filepath)
        green = df.iloc[:, 0].values
        fs = 100
        
        # Analyze signal
        results = analyze_ppg_cycle(green, fs, show_plot)
        return results
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def process_video(video_path, show_plot=False):
    """
    Process a single video file and return heart rate and hydration metrics.
    
    Args:
        video_path: Path to video file
        show_plot: Whether to show visualization plots
        
    Returns:
        Tuple of (heart_rate, hydration_score, hydration_class) where:
        - heart_rate: Average heart rate in BPM
        - hydration_score: TPA/VPA ratio
        - hydration_class: Classification of hydration level
        Returns (None, None, None) if processing fails
    """
    try:
        # Process video using CHROM method
        result = analyze_video_chrom((video_path, 'heart_rate'))
        
        if result is None:
            print(f"Error: Could not process video {video_path}")
            return None, None, None
            
        heart_rate = result['hr']
        hydration_score = result['hydration_results']['tpa_vpa_ratio']
        hydration_class = classify_hydration(hydration_score)
        
        if show_plot:
            # Create visualization
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f'PPG Analysis Results', fontsize=16, y=0.95)
            
            # Heart rate plot
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(result['timing'], result['bvp'], 'b-', label='PPG Signal')
            ax1.plot(result['timing'][result['peaks']], 
                    result['bvp'][result['peaks']], 'ro',
                    label='Peaks')
            ax1.set_title(f'Heart Rate: {heart_rate:.1f} BPM')
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
        
        return heart_rate, hydration_score, hydration_class
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None, None, None

'''
def main():
    # Get list of CSV files
    data_dir = 'data/regular'
    csv_files = glob(os.path.join(data_dir, '*.csv'))
    
    # Analyze all files
    file_results = []  # List of (filepath, results) tuples
    all_ratios = []
    all_files = []
    
    for filepath in csv_files:
        results = analyze_file(filepath, show_plot=False)  # Don't show plots yet
        
        if results and results['tpa_vpa_ratio'] > 0:  # Only include valid ratios
            file_results.append((filepath, results))
            all_ratios.append(results['tpa_vpa_ratio'])
            all_files.append(filepath)
    
    # Show 5 random samples
    print("\nAnalyzing 5 Random Samples:")
    sample_indices = np.random.choice(len(file_results), 5, replace=False)
    
    for idx in sample_indices:
        filepath, results = file_results[idx]
        print(f"\nAnalyzing sample: {os.path.basename(filepath)}")
        print(f"TPA/VPA Ratio: {results['tpa_vpa_ratio']:.3f}")
        
        # Load and analyze the signal
        signal = pd.read_csv(filepath).iloc[:, 0].values
        analyze_ppg_cycle(signal, fs=100, show_plot=True)
    
    # Calculate statistics on full dataset
    ratios = np.array(all_ratios)
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    
    # Calculate hydration thresholds
    normal_upper = mean_ratio + std_ratio
    normal_lower = mean_ratio - std_ratio
    dehydration_severe = mean_ratio - 2*std_ratio
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    # Calculate number of bins using Freedman-Diaconis rule
    q75, q25 = np.percentile(ratios, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(ratios) ** (1/3))
    num_bins = int((np.max(ratios) - np.min(ratios)) / bin_width)
    
    plt.hist(ratios, bins=num_bins, edgecolor='black')
    plt.title('Distribution of TPA/VPA Ratios')
    plt.xlabel('TPA/VPA Ratio')
    plt.ylabel('Frequency')
    
    # Add mean and threshold lines
    plt.axvline(mean_ratio, color='r', linestyle='--', 
                label=f'Mean: {mean_ratio:.3f}')
    plt.axvline(normal_lower, color='y', linestyle=':', 
                label=f'Mild Dehydration: <{normal_lower:.3f}')
    plt.axvline(dehydration_severe, color='r', linestyle=':', 
                label=f'Severe Dehydration: <{dehydration_severe:.3f}')
    plt.axvline(normal_upper, color='y', linestyle=':', 
                label=f'Overhydration: >{normal_upper:.3f}')
    
    plt.legend()
    plt.grid(True)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(ratios)}")
    print(f"Average TPA/VPA Ratio: {mean_ratio:.3f} ± {std_ratio:.3f}")
    print(f"Range: {np.min(ratios):.3f} to {np.max(ratios):.3f}")
    
    print("\nProposed Hydration Thresholds:")
    print(f"Normal Range: {normal_lower:.3f} - {normal_upper:.3f}")
    print(f"Mild Dehydration: <{normal_lower:.3f}")
    print(f"Severe Dehydration: <{dehydration_severe:.3f}")
    print(f"Overhydration: >{normal_upper:.3f}")
    
    plt.show()'''

if __name__ == "__main__":
    # Example usage
    video_path = "data/video/65bpmWatch.mov"
    hr, score, classification = process_video(video_path, show_plot=True)
    
    if hr is not None:
        print(f"\nResults:")
        print(f"Heart Rate: {hr:.1f} BPM")
        print(f"Hydration Score: {score:.3f}")
        print(f"Classification: {classification}")
    # main()