import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from simple_ppg import analyze_video_chrom
import pandas as pd
import os
from glob import glob
from utils import analyze_ppg_cycle, classify_hydration
import seaborn as sns
from matplotlib.patches import Rectangle
from datetime import datetime

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

def create_aesthetic_trace(signal, time, peaks, sampling_rate, save_path=None):
    """
    Create an aesthetic visualization of the PPG signal trace.
    
    Args:
        signal: PPG signal array
        time: Time points array
        peaks: Array of peak indices
        sampling_rate: Sampling rate in Hz
        save_path: Optional path to save the plot
        
    Returns:
        Path to saved plot if save_path provided, None otherwise
    """
    # Set style
    plt.style.use('default')
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    sns.set_palette("husl")
    
    # Create figure with transparent background
    fig = plt.figure(figsize=(12, 4), dpi=100)
    ax = plt.gca()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.1)
    
    # Plot the signal
    line_color = '#3498db'  # Bright blue
    
    # Smooth the signal for aesthetics
    window = int(sampling_rate * 0.1)  # 100ms window
    if window % 2 == 0:
        window += 1
    smoothed = sp.savgol_filter(signal, window, 2)
    
    # Plot main signal with fill to bottom
    plt.plot(time, smoothed, color=line_color, linewidth=2, alpha=0.8)
    
    # Set y limits to include some padding
    ymax = max(smoothed) + 0.1 * (max(smoothed) - min(smoothed))
    ymin = min(smoothed) - 0.1 * (max(smoothed) - min(smoothed))
    plt.ylim(ymin, ymax)
    
    # Fill from signal to bottom of plot
    plt.fill_between(time, smoothed, ymin, color=line_color, alpha=0.3)
    
    # Clean up the plot
    plt.grid(True, alpha=0.15)
    plt.xlabel('Time (s)', color='white', fontsize=14)
    plt.ylabel('', color='white', fontsize=14)
    
    # Remove y-axis label and ticks
    ax.yaxis.set_ticks([])
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_alpha(0.3)
    ax.spines['bottom'].set_color('white')
    
    # Set tick colors and sizes to white
    ax.tick_params(colors='white', labelsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, 
                   bbox_inches='tight',
                   pad_inches=0.1,
                   transparent=True,
                   dpi=300)
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()
        return None

def process_video(video_path, show_plot=False, save_trace=None):
    """
    Process a single video file and return heart rate and hydration metrics.
    
    Args:
        video_path: Path to video file
        show_plot: Whether to show visualization plots
        save_trace: Optional path to save the aesthetic trace plot
        
    Returns:
        Tuple of (heart_rate, hydration_score, hydration_class, water_recommendation) where:
        - heart_rate: Average heart rate in BPM
        - hydration_score: TPA/VPA ratio
        - hydration_class: Classification of hydration level
        - water_recommendation: Recommended water intake in oz
        Returns (None, None, None, None) if processing fails
    """
    try:
        # Process video using CHROM method
        result = analyze_video_chrom((video_path, 'heart_rate'))
        
        if result is None:
            print(f"Error: Could not process video {video_path}")
            return None, None, None, None
            
        heart_rate = round(result['hr'])  # Round to nearest integer
        hydration_score = round(result['hydration_results']['tpa_vpa_ratio'], 3)  # Round to 3 decimal places
        hydration_class = classify_hydration(hydration_score)
        # Calculate water recommendation
        if hydration_class == "Mild Dehydration":
            water_recommendation = round((hydration_score/4.284) * 33.8, 1)
        else:
            water_recommendation = 0.0
        
        # Create aesthetic trace if requested
        if save_trace:
            create_aesthetic_trace(
                result['bvp'],
                result['timing'],
                result['peaks'],
                result['sampling_rate'],
                save_trace
            )
        
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
            ax1.set_title(f'Heart Rate: {heart_rate} BPM')
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
        
        return heart_rate, hydration_score, hydration_class, water_recommendation
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    # Create traces directory if it doesn't exist
    traces_dir = "data/traces"
    os.makedirs(traces_dir, exist_ok=True)
    
    # Generate timestamp for trace file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = os.path.join(traces_dir, f"trace_{timestamp}.png")
    
    # Example usage
    video_path = "data/video/65bpmWatch.mov"
    hr, score, classification, water = process_video(
        video_path, 
        show_plot=False,
        save_trace=trace_path
    )
    
    if hr is not None:
        print(f"\nResults:")
        print(f"Heart Rate: {hr} BPM")
        print(f"Hydration Score: {score:.3f}")
        print(f"Classification: {classification}")
        print(f"Recommended Water Intake: {water:.1f} oz")
        print(f"Trace saved as: {trace_path}")