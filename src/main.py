"""
PPG Signal Analysis Module using pyPPG toolbox
For more information see: https://pyppg.readthedocs.io/en/latest/index.html
"""

import numpy as np
from scipy.integrate import trapezoid
from pyPPG import PPG
from pyPPG.preproc import preprocess_signal  # Updated import
from pyPPG.biomarkers import get_fiducial_points  # Updated function name
from typing import List, Tuple, Optional
import logging
from dotmap import DotMap

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPGAnalyzer:
    """Class for analyzing PPG signals and calculating relevant metrics."""
    
    def __init__(self, signal: np.ndarray, sampling_rate: float):
        """
        Initialize PPG analyzer with signal data.
        
        Args:
            signal (np.ndarray): Raw PPG signal data
            sampling_rate (float): Sampling rate in Hz (should be >= 75Hz as per docs)
        """
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.ppg_obj = None
        self.fiducials = None
    
    def preprocess_signal(self) -> None:
        """
        Preprocess the PPG signal using pyPPG's preprocessing tools.
        Includes filtering and derivatives computation.
        """
        try:
            # Create PPG object with signal data using DotMap as per documentation
            signal_data = DotMap({
                'signal': self.signal,  # Updated key from 'v' to 'signal'
                'sampling_rate': self.sampling_rate  # Updated key from 'fs'
            })
            self.ppg_obj = PPG(signal_data)
            
            # Preprocess signal using the documented function
            self.ppg_obj = preprocess_signal(
                signal=self.ppg_obj.signal,
                sampling_rate=self.sampling_rate,
                filter_type='butterworth',  # As specified in docs
                cutoff_freq=[0.5, 8.0]  # Standard cutoff frequencies for PPG
            )
            
            # Get fiducial points using the documented function
            self.fiducials = get_fiducial_points(
                signal=self.ppg_obj,
                sampling_rate=self.sampling_rate
            )
            
        except Exception as e:
            logger.error(f"Error during signal preprocessing: {str(e)}")
            raise
    
    def _get_heartbeat_indices(self) -> List[Tuple[int, int, int]]:
        """
        Extract indices for each heartbeat using fiducial points.
        """
        # Updated keys based on documentation
        onsets = sorted(self.fiducials['onsets'])
        peaks = sorted(self.fiducials['systolic_peaks'])
        offsets = sorted(self.fiducials['offsets'])
        
        heartbeats = []
        for i in range(len(onsets)):
            if i < len(peaks) and i < len(offsets):
                if onsets[i] < peaks[i] < offsets[i]:
                    heartbeats.append((onsets[i], peaks[i], offsets[i]))
                
        return heartbeats
    
    def _calculate_area(self, signal: np.ndarray, baseline: np.ndarray) -> float:
        """
        Calculate area between signal and baseline.
        
        Args:
            signal (np.ndarray): Signal values
            baseline (np.ndarray): Baseline values
            
        Returns:
            float: Calculated area
        """
        return trapezoid(np.maximum(signal - baseline, 0))
    
    def calculate_tpa_vpa_ratio(self) -> float:
        """
        Calculate the TPA/VPA ratio for the PPG signal.
        
        Returns:
            float: TPA/VPA ratio
        """
        if self.ppg_obj is None:
            self.preprocess_signal()
            
        tpa_total = 0.0
        vpa_total = 0.0
        
        try:
            heartbeats = self._get_heartbeat_indices()
            
            for start_idx, peak_idx, end_idx in heartbeats:
                # Construct systolic and diastolic lines
                systolic_up = np.linspace(
                    self.ppg_obj.ppg[start_idx],
                    self.ppg_obj.ppg[peak_idx],
                    num=peak_idx - start_idx + 1
                )
                systolic_down = np.linspace(
                    self.ppg_obj.ppg[peak_idx],
                    self.ppg_obj.ppg[end_idx],
                    num=end_idx - peak_idx
                )
                systolic_line = np.append(systolic_up, systolic_down)
                
                diastolic_line = np.linspace(
                    self.ppg_obj.ppg[start_idx],
                    self.ppg_obj.ppg[end_idx],
                    num=end_idx - start_idx + 1
                )
                
                # Get signal for this heartbeat
                signal_range = self.ppg_obj.ppg[start_idx:end_idx+1]
                
                # Calculate areas
                tpa = self._calculate_area(signal_range, systolic_line[:-1])
                vpa = self._calculate_area(diastolic_line[:-1], signal_range)
                
                tpa_total += tpa
                vpa_total += vpa
            
            ratio = tpa_total / vpa_total if vpa_total != 0 else 0.0
            logger.info(f"Calculated TPA/VPA ratio: {ratio:.4f}")
            return ratio
            
        except Exception as e:
            logger.error(f"Error calculating TPA/VPA ratio: {str(e)}")
            raise

def main():
    """Example usage of PPGAnalyzer."""
    # Generate example PPG data
    sampling_rate = 100  # Hz
    time = np.linspace(0, 10, 1000)
    ppg_signal = np.sin(2 * np.pi * 1 * time) + 0.5 * np.sin(2 * np.pi * 2 * time)
    
    try:
        # Create analyzer and calculate ratio
        analyzer = PPGAnalyzer(ppg_signal, sampling_rate)
        ratio = analyzer.calculate_tpa_vpa_ratio()
        print(f"TPA/VPA Ratio: {ratio:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()