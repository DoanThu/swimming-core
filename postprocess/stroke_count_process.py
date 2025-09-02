import numpy as np 
from scipy.signal import windows
from scipy.signal import detrend


def get_stroke_count_cap(x, sampling_rate=30):
    x = x - np.mean(x)              # zero-mean
    x = detrend(x)                  # remove trend
    x *= windows.hann(len(x))       # apply window
    T = len(x)

    # Compute FFT
    fft_vals = np.fft.fft(x)
    fft_freqs = np.fft.fftfreq(T, d=1/sampling_rate)  # d=1/fs is the sample spacing

    mask = (fft_freqs >= 0.2) & (fft_freqs <= 1.2)
    freqs_in_range = fft_freqs[mask]
    magnitude_in_range = np.abs(fft_vals[mask])

    # Get index of the max amplitude
    dominant_index = np.argmax(magnitude_in_range)
    dominant_freq = freqs_in_range[dominant_index]
    
    return dominant_freq*len(x)/sampling_rate

def flip_skeletons(skeletons):
    skeletons = np.array(skeletons)
    # This works with horizontal videos
    for i in range(len(skeletons)):
        skeleton = skeletons[i]
        if skeleton[9,1] < skeleton[10,1]:
            skeletons[i][9], skeletons[i][10] = skeletons[i][10], skeletons[i][9]
    return skeletons

def get_nose_wrist_distance(skeletons):
    # This works with horitontal videos
    skeletons = np.array(skeletons)
    return skeletons[:,0,0]-skeletons[:,9,0]