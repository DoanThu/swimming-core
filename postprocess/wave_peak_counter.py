import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # Or 'TkAgg', 'Qt5Agg', etc.
import matplotlib.pyplot as plt
import os 
from scipy.signal import find_peaks

def classify_window_sine_vs_noise(x, fs, min_freq=0.1,
                                  peak_ratio_thr=8.0, flatness_thr=0.5, ac_peak_thr=0.2):
    x = np.asarray(x, float)
    n = len(x)
    if n < 16:
        return {"label": "noise", "dominant_frequency_hz": 0.0,
                "peak_ratio": 0.0, "spectral_flatness": 1.0, "autocorr_peak": 0.0}
    x = x - np.mean(x)
    w = np.hanning(n)
    xw = x * w
    freqs = np.fft.rfftfreq(n, d=1/fs)
    spec = np.fft.rfft(xw)
    power = (np.abs(spec)**2) / (np.sum(w**2) + 1e-12)
    mask = (freqs >= min_freq)
    mask[0] = False
    if not np.any(mask):
        return {"label": "noise", "dominant_frequency_hz": 0.0,
                "peak_ratio": 0.0, "spectral_flatness": 1.0, "autocorr_peak": 0.0}
    f_sel, p_sel = freqs[mask], power[mask] + 1e-18
    k = int(np.argmax(p_sel))
    f_dom = float(f_sel[k])
    p_peak = float(p_sel[k])
    bg_median = float(np.median(np.delete(p_sel, k))) if len(p_sel) > 1 else p_peak
    peak_ratio = p_peak / (bg_median + 1e-18)
    spectral_flatness = float(np.exp(np.mean(np.log(p_sel))) / (np.mean(p_sel) + 1e-18))
    ac = np.correlate(x, x, mode="full")[n-1:]
    ac = ac / (ac[0] + 1e-12)
    if len(ac) > 2:
        lag_idx = int(np.argmax(ac[1:])) + 1
        ac_peak = float(ac[lag_idx])
    else:
        ac_peak = 0.0
    is_wave = (peak_ratio > peak_ratio_thr) and (spectral_flatness < flatness_thr) and (ac_peak > ac_peak_thr)
    return {"label": "wave" if is_wave else "noise",
            "dominant_frequency_hz": f_dom,
            "peak_ratio": peak_ratio,
            "spectral_flatness": spectral_flatness,
            "autocorr_peak": ac_peak}

def label_windows(x, fs, window_sec=1.0, step_sec=0.5, **clf_kwargs):
    n = len(x)
    W = max(16, int(window_sec * fs))
    S = max(1, int(step_sec * fs))
    labels, fdoms, spans, metrics = [], [], [], []
    for start in range(0, max(1, n - W + 1), S):
        end = start + W
        res = classify_window_sine_vs_noise(x[start:end], fs, **clf_kwargs)
        labels.append(res["label"])
        fdoms.append(res.get("dominant_frequency_hz", 0.0))
        spans.append((start, min(end, n)))
        metrics.append(res)
    return labels, fdoms, spans, metrics

def build_wave_mask(n, spans, labels):
    votes = np.zeros(n, dtype=np.int16)
    cover = np.zeros(n, dtype=np.int16)
    for (start, end), lab in zip(spans, labels):
        cover[start:end] += 1
        if lab == "wave":
            votes[start:end] += 1
    mask = np.zeros(n, dtype=bool)
    sel = cover > 0
    mask[sel] = votes[sel] * 2 >= cover[sel]
    return mask

def detect_peaks_in_wave_regions(x, fs, wave_mask, min_prominence=0.2, fdoms=None, spans=None):
    x = np.asarray(x, float)
    n = len(x)
    min_distance = 10
    height = max(x)*0.9
    if fdoms is not None and spans is not None:
        per_samples = []
        for f, (s, e) in zip(fdoms, spans):
            if f and f > 0:
                per_samples.append(fs / f)
        if per_samples:
            median_period = np.median(per_samples)
            min_distance = max(1, int(0.5 * median_period))
    # kwargs = {"prominence": min_prominence, 'height': height}
    kwargs = {'height': height}
    if min_distance is not None:
        kwargs["distance"] = min_distance
    peaks, props = find_peaks(x, **kwargs)

    # if _HAS_SCIPY:
        # kwargs = {"prominence": min_prominence, 'height': height}
        # if min_distance is not None:
            # kwargs["distance"] = min_distance
        # peaks, props = find_peaks(x, **kwargs)
    # else:
        # dx = np.diff(x)
        # sign = np.sign(dx)
        # zc = np.where((np.roll(sign, 1) > 0) & (sign <= 0))[0]
        # peaks = zc
        # props = {}

    peaks_in_wave = [p for p in peaks if 0 <= p < n and wave_mask[p]]
    return np.array(peaks_in_wave, dtype=int), props

def count_wave_peaks_and_plot(x, fs, window_sec=1.0, step_sec=0.5,
                              clf_kwargs=None, min_prominence=0.2, debug=False):
    x = np.asarray(x, float)
    n = len(x)
    if clf_kwargs is None:
        clf_kwargs = dict(min_freq=0.5, peak_ratio_thr=6.0, flatness_thr=0.55, ac_peak_thr=0.2)
    labels, fdoms, spans, metrics = label_windows(x, fs, window_sec, step_sec, **clf_kwargs)
    wave_mask = build_wave_mask(n, spans, labels)
    peaks_idx, props = detect_peaks_in_wave_regions(x, fs, wave_mask, min_prominence, fdoms, spans)
    if debug:
        os.makedirs('debug', exist_ok=True)
        t = np.arange(n) / fs
        plt.figure()
        plt.plot(t, x, label="signal")
        if len(peaks_idx) > 0:
            plt.plot(t[peaks_idx], x[peaks_idx], "x", label="peaks (wave-only)")
        plt.xlabel("Time (s)") 
        plt.ylabel("Amplitude")
        plt.title("Signal with peaks (counted only in wave regions)")
        plt.legend()
        plt.savefig('debug/debug1.pdf', bbox_inches='tight')

        plt.figure()
        plt.plot(t, wave_mask.astype(int), drawstyle="steps-post")
        plt.ylim(-0.2, 1.2); plt.yticks([0,1], ["noise","wave"])
        plt.xlabel("Time (s)")
        plt.ylabel("Mask")
        plt.title("Wave/Noise Timeline (1=wave)")
        plt.savefig('debug/debug2.pdf', bbox_inches='tight')

    diagnostics = {"labels": labels, "spans": spans, "fdoms": fdoms, "metrics_per_window": metrics, "props": props}
    return int(len(peaks_idx)), np.array(peaks_idx, int), wave_mask, diagnostics


# def main():
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", type=str, help="Path to CSV with one column (or use --col).")
#     ap.add_argument("--col", type=int, default=None, help="Column in CSV (0-based).")
#     ap.add_argument("--npy", type=str, help="Path to a .npy file instead of CSV.")
#     ap.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz).")
#     ap.add_argument("--window_sec", type=float, default=1.0, help="Window size (s).")
#     ap.add_argument("--step_sec", type=float, default=0.5, help="Step size (s).")
#     ap.add_argument("--min_prominence", type=float, default=0.25, help="Peak prominence.")
#     ap.add_argument("--show", action="store_true", help="Show plots.")
#     args = ap.parse_args()
#     if args.csv:
#         x = _load_csv(args.csv, args.col)
#     elif args.npy:
#         x = np.load(args.npy)
#     else:
#         raise SystemExit("Provide --csv or --npy")
#     count, peaks_idx, mask, diag = count_wave_peaks_and_plot(
#         x, args.fs, args.window_sec, args.step_sec,
#         clf_kwargs=dict(min_freq=0.5, peak_ratio_thr=6.0, flatness_thr=0.55, ac_peak_thr=0.2),
#         min_prominence=args.min_prominence, show=args.show
#     )
#     print("Peak count (wave-only):", count)
#     print("Peak indices:", peaks_idx.tolist())

