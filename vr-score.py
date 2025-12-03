import cv2
import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

class VideoForensics:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def _get_noise_residual(self, frame):
        """
        Extracts high-frequency noise by subtracting a Gaussian blurred version 
        from the original. (Proxy for sensor noise analysis)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, denoised)
        return noise

    def analyze_spatial_realism(self, num_frames=5):
        """
        Metric 1: Spatial Realism via FFT.
        AI videos often lack high-frequency detail or have artifact spikes.
        We measure the ratio of High-Freq to Low-Freq energy.
        """
        scores = []
        # Sample frames evenly
        indices = np.linspace(0, self.frame_count-1, num_frames, dtype=int)
        
        for idx in indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # FFT Transformation
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            
            # Calculate ratio of High Freq energy vs Low Freq energy
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Mask for low freq (center circle)
            radius = min(h, w) // 8
            y, x = np.ogrid[:h, :w]
            mask_area = (x - center_w)**2 + (y - center_h)**2 <= radius**2
            
            low_freq_energy = np.mean(magnitude_spectrum[mask_area])
            high_freq_energy = np.mean(magnitude_spectrum[~mask_area])
            
            # A natural image usually has a balanced decay. 
            # Drastic drop-off = AI smoothness. Spikes = AI artifacts.
            # We return a ratio; extremely low values (< 0.5) often indicate AI smoothing.
            scores.append(high_freq_energy / (low_freq_energy + 1e-5))
            
        return np.mean(scores) if scores else 0

    def analyze_temporal_soundness(self, sample_size=20):
        """
        Metric 2: Temporal Soundness via Optical Flow Jitter.
        Measures if background texture moves 'smoothly' or jitters.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = self.cap.read()
        if not ret: return 0
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        jitter_scores = []
        
        # Analyze a burst of frames
        for _ in range(sample_size):
            ret, frame = self.cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Dense optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Magnitude of flow
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Logic: In real video, "still" backgrounds have near-0 motion.
            # In AI video, "still" backgrounds often shimmer (temporal noise).
            # We look at the variance of the lowest 50% of motion pixels.
            static_mask = mag < np.percentile(mag, 50)
            if np.sum(static_mask) > 0:
                # Variance of 'static' pixels. High variance = AI "shimmer"
                background_jitter = np.var(mag[static_mask])
                jitter_scores.append(background_jitter)
            
            prev_gray = gray
            
        return np.mean(jitter_scores) if jitter_scores else 0

    def analyze_fingerprint_realism(self, num_frames=5):
        """
        Metric 3: Fingerprint Realism via Noise Kurtosis.
        Real sensors have heavy-tailed (high kurtosis) noise due to demosaicing.
        AI generators often output Gaussian (Kurtosis ~0) or uniform noise.
        """
        kurtosis_scores = []
        indices = np.linspace(0, self.frame_count-1, num_frames, dtype=int)

        for idx in indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret: break
            
            noise = self._get_noise_residual(frame)
            k_score = kurtosis(noise.flatten())
            kurtosis_scores.append(k_score)
            
        return np.mean(kurtosis_scores) if kurtosis_scores else 0

    def get_difficulty_metrics(self):
        return {
            "spatial_score": self.analyze_spatial_realism(),
            "temporal_jitter": self.analyze_temporal_soundness(),
            "noise_kurtosis": self.analyze_fingerprint_realism()
        }


if __name__ == "__main__":
    import os
    file_name = "test_2.mp4"
    video_path = os.path.join("data", file_name)  # Replace with your video path
    print("Video Forensics Analysis")
    analyzer = VideoForensics(video_path)
    metrics = analyzer.get_difficulty_metrics()
    print(f"Spatial Realism Score (Higher=Real): {metrics['spatial_score']:.2f}")
    print(f"Temporal Jitter Score (Lower=Real): {metrics['temporal_jitter']:.4f}")
    print(f"Noise Kurtosis Score (Higher=Real): {metrics['noise_kurtosis']:.2f}")