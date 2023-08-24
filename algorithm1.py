import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft

def compute_features(raw_data_sequence):
    init = 1
    w_t = 0
    features = []

    while len(raw_data_sequence) > 60:  # Assuming raw data sequence is greater than 1 minute
        if init:
            prev_lag = 0
            post_lag = 1
            init = 0
        
        for i in range(w_t - prev_lag, w_t + post_lag):
            w_t_values = raw_data_sequence[i]
            mean = np.mean(w_t_values)
            asymmetry = skew(w_t_values)
            peakedness = kurtosis(w_t_values)
            w_min = np.min(w_t_values)
            w_max = np.max(w_t_values)
            sample_variances = np.cov(w_t_values)
            eigenvalues = np.linalg.eigvals(sample_variances)
            logarithm = np.log(np.triu(sample_variances))
            frequency_magnitude_components = np.abs(fft(w_t_values))
            fft_most_energetic = np.argsort(frequency_magnitude_components)[-10:]
            
            features.append({
                'mean': mean,
                'asymmetry': asymmetry,
                'peakedness': peakedness,
                'w_min': w_min,
                'w_max': w_max,
                'sample_variances': sample_variances,
                'eigenvalues': eigenvalues,
                'logarithm': logarithm,
                'frequency_magnitude_components': frequency_magnitude_components,
                'fft_most_energetic': fft_most_energetic
            })
        
        w_t += 1
        prev_lag = 0.5
        post_lag = 1.5

    return features

# Example usage
raw_data_sequence = np.random.rand(500)  # Replace with your actual EEG raw data sequence
extracted_features = compute_features(raw_data_sequence)
for feature in extracted_features:
    print(feature)
