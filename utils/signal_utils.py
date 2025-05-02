import numpy as np

def moving_average(x, window_size=5):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')