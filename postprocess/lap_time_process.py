import numpy as np 

class LapTime:
    def __init__(self):
        pass 

    def remove_short_peaks(self, signal, max_len=2):
        """
        Replace short runs of 1s (<= max_len) with 0s.
        Input: binary 1D list or array
        Returns: cleaned signal (numpy array)
        """
        signal = np.array(signal)
        cleaned = signal.copy()

        i = 0
        while i < len(signal):
            if signal[i] == 1:
                start = i
                while i < len(signal) and signal[i] == 1:
                    i += 1
                length = i - start
                if length <= max_len:
                    cleaned[start:i] = 0
            else:
                i += 1

        return cleaned
    
    def find_zero_segments(self, signal, fps, min_duration_sec=5.0):
        """
        Find start and end times (in seconds) for continuous 0 segments,
        and only keep those longer than min_duration_sec.

        Args:
            signal (list[int]): binary list of 0s and 1s
            fps (int): frames per second
            min_duration_sec (float): duration threshold in seconds

        Returns:
            List of tuples: (start_time_sec, end_time_sec)
        """
        zero_segments = []
        in_zero = False
        min_frames = int(min_duration_sec * fps)

        for i, val in enumerate(signal):
            if val == 0 and not in_zero:
                start_frame = i
                in_zero = True
            elif val == 1 and in_zero:
                end_frame = i - 1
                if end_frame - start_frame + 1 >= min_frames:
                    zero_segments.append((start_frame / fps, end_frame / fps))
                in_zero = False

        # If the signal ends with 0
        if in_zero:
            end_frame = len(signal) - 1
            if end_frame - start_frame + 1 >= min_frames:
                zero_segments.append((start_frame / fps, end_frame / fps))

        return zero_segments