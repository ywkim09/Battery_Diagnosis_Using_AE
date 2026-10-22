"""Script version of data_analysis_fourier.ipynb.

Trains AR / Gaussian NB / XGBoost classifiers on labeled AE signals
(normal / gassing / cracking), classifies the "mixed" cluster, and then
applies the trained models to an unlabeled test folder.

Figures that the notebook displayed inline are saved to result/figures/.
"""
import os, datetime
import time
from pathlib import Path
import pickle
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pywt
from scipy.signal import hilbert
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from scipy.stats import entropy

time_threshold = 0.00005
N = 500
method = 'wavelet'
PULSE_DETECTION_BOOL = True
AIC_BOOL = True
AIC_ENVELOPE_THRESHOLD = 0.2
AIC_SEARCH_HALF_WINDOW = 500
VOLTAGE_THRESHOLD = 0.02

FIGURE_DIR = os.path.join('result', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 

cluster_folders = {
    "gassing": Path("Data/260928/Cluster_0 (gassing)"),
    "cracking": Path("Data/260928/Cluster_1 (cracking)"),
    "mixed": Path("Data/260928/Cluster_2 (mix undecided)")
}

unlabeled_folders = {
    "test": Path("Data/261012/Raw CH4 LFP-Gr Plating 2026-01-21"),
}

AR_ORDER = 100
AR_CLASS_NAMES = ["Normal", "Gassing", "Cracking"]
MODEL_LST = ["GaussNB", "XGBoost"]


_START_TIME = time.time()


def log(msg):
    print(f"[{time.time() - _START_TIME:8.1f}s] {msg}", flush=True)


def save_current_figure(name):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_DIR, name) , dpi=150)
    plt.close()

def onset_anchor_index(signal, time_values=None):
    signal = np.asarray(signal, dtype=float)
    if AIC_BOOL:
        envelope = np.abs(hilbert(signal))
        envelope_max = float(np.max(envelope))
        if envelope_max == 0:
            raise ValueError("Signal envelope is zero; onset cannot be detected.")
        envelope /= envelope_max
        time_values = np.asarray(time_values, dtype=float)
        above = np.flatnonzero(
                                (envelope >= AIC_ENVELOPE_THRESHOLD)
                                & (time_values >= time_threshold)
                            )
        # above = np.flatnonzero(envelope >= AIC_ENVELOPE_THRESHOLD)
    else:
        above_mask = np.abs(signal) >= VOLTAGE_THRESHOLD
        if time_values is None:
            raise ValueError(
                "Time values are required when AIC is disabled."
            )
        time_values = np.asarray(time_values, dtype=float)
        if len(time_values) != len(signal):
            raise ValueError("Signal and time values must have the same length.")
        above = np.flatnonzero(
            above_mask & (time_values >= time_threshold)
        )

    if len(above) == 0:
        if AIC_BOOL:
            raise ValueError(
                f"Normalized envelope has no value at least "
                f"{AIC_ENVELOPE_THRESHOLD}."
            )
        raise ValueError(
            f"No sample has |amplitude| >= {VOLTAGE_THRESHOLD} V at or after "
            f"{time_threshold} s."
        )
    return int(above[0])


def onset_picker(signal, time_values=None):
    signal = np.asarray(signal, dtype=float)
    threshold_index = onset_anchor_index(signal, time_values)
    if not AIC_BOOL:
        return threshold_index

    envelope = np.abs(hilbert(signal))
    envelope_max = float(np.max(envelope))
    envelope /= envelope_max

    lower = max(threshold_index - AIC_SEARCH_HALF_WINDOW, 0)
    upper = min(
        threshold_index + AIC_SEARCH_HALF_WINDOW,
        len(envelope),
    )
    window = envelope[lower:upper]
    if len(window) < 4:
        raise ValueError(
            f"Only {len(window)} samples are available in the AIC window."
        )

    split_indices = np.arange(2, len(window) - 1)
    cumulative_sum = np.cumsum(window)
    cumulative_square_sum = np.cumsum(window**2)
    left_count = split_indices
    right_count = len(window) - split_indices
    left_sum = cumulative_sum[split_indices - 1]
    right_sum = cumulative_sum[-1] - left_sum
    left_square_sum = cumulative_square_sum[split_indices - 1]
    right_square_sum = cumulative_square_sum[-1] - left_square_sum
    left_ssr = np.maximum(
        left_square_sum - left_sum**2 / left_count,
        np.finfo(float).eps,
    )
    right_ssr = np.maximum(
        right_square_sum - right_sum**2 / right_count,
        np.finfo(float).eps,
    )
    aic_scores = (
        left_count * np.log(left_ssr / left_count)
        + right_count * np.log(right_ssr / right_count)
        + 8
    )
    return lower + int(split_indices[np.argmin(aic_scores)])


def onset_segment(dataframe, N, context):
    try:
        onset = onset_picker(
            dataframe["Amplitude (V)"].to_numpy(),
            dataframe["Time (s)"].to_numpy(),
        )
    except ValueError as error:
        log(f"  Skipping {context}: {error}")
        return None

    segment = dataframe.iloc[onset:onset + N].copy()
    if len(segment) < N:
        log(
            f"  Skipping {context}: only {len(segment)} samples remain after "
            f"onset; {N} required."
        )
        return None
    return segment


def first_segment_after_threshold(dataframe, N, time_threshold, context):
    signal = dataframe.loc[dataframe["Time (s)"] >= time_threshold]
    if len(signal) < N:
        log(
            f"  Skipping {context}: only {len(signal)} samples remain after "
            f"time threshold; {N} required."
        )
        return None
    return signal.iloc[:N].copy()


# ---------------------------------------------------------------------------
# Data loading / segmentation
# ---------------------------------------------------------------------------
def load_clusters(folders):
    clusters = {}
    for cluster, folder in folders.items():
        csv_files = sorted(folder.glob("*.csv"))
        log(f"  Loading '{cluster}': {len(csv_files)} CSV files from {folder}")
        clusters[cluster] = {
            csv_file.stem: pd.read_csv(csv_file)
            for csv_file in csv_files
        }
    return clusters


def process_data(cluster_data, N, time_threshold):
    noise_data = {}
    signal_data = {"normal": {}}

    for cluster_name, file_data in cluster_data.items():
        noise_data[cluster_name] = {}
        signal_data[cluster_name] = {}
        for file_name, dataframe in file_data.items():
            noise_mask = dataframe["Time (s)"] < time_threshold
            noise_data[cluster_name][file_name] = [
                dataframe.loc[noise_mask].copy()
            ]
            if PULSE_DETECTION_BOOL:
                segment = onset_segment(
                    dataframe,
                    N,
                    f"{cluster_name}/{file_name}",
                )
                if segment is not None:
                    signal_data[cluster_name][file_name] = [segment]
            else:
                signal_data["normal"][file_name] = [
                    dataframe.loc[noise_mask].copy()
                ]
                segment = first_segment_after_threshold(
                    dataframe,
                    N,
                    time_threshold,
                    f"{cluster_name}/{file_name}",
                )
                if segment is not None:
                    signal_data[cluster_name][file_name] = [segment]
    return noise_data, signal_data


def process_unlabeled_data(cluster_data, N):
    signal_data = {}

    for cluster_name, file_data in cluster_data.items():
        signal_data[cluster_name] = {}
        for file_name, dataframe in file_data.items():
            if PULSE_DETECTION_BOOL:
                segment = onset_segment(
                    dataframe,
                    N,
                    f"{cluster_name}/{file_name}",
                )
                if segment is not None:
                    signal_data[cluster_name][file_name] = [segment]
            else:
                segment = first_segment_after_threshold(
                    dataframe,
                    N,
                    time_threshold,
                    f"{cluster_name}/{file_name}",
                )
                if segment is not None:
                    signal_data[cluster_name][file_name] = [segment]
    return signal_data


def recombine_dfs(dfs):
    recombined_dict = {}
    for k in dfs:
        recombined_dict[k] = pd.concat(
                    [df for df in dfs[k]],
                ignore_index=True,
            )
    return recombined_dict


# ---------------------------------------------------------------------------
# Exploratory plots of the noise (pre-threshold) part
# ---------------------------------------------------------------------------
def plot_noise_moving_average(noise_data):
    k = list(noise_data["cracking"].keys())[0]
    df = noise_data["cracking"][k][0]
    smooth_amplitude = df['Amplitude (V)'].rolling(window=100, center=True, min_periods=1).mean()

    plt.figure(figsize=(10, 5))

    plt.plot(df['Time (s)'], df['Amplitude (V)'], label='Original Noisy', color='lightgray', alpha=0.6)
    plt.plot(df['Time (s)'], smooth_amplitude, label='Smoothed (Moving Average)', color='blue', linewidth=2)

    plt.title('Cracking Noise Signal - Moving Average')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.legend()
    plt.grid(True)
    save_current_figure("noise_moving_average.png")


def plot_noise_signals(noise_data):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    for i, c in enumerate(["gassing", "cracking"]):
        for key in noise_data[c]:
            noise_data[c][key][0].plot(
                x="Time (s)",
                y="Amplitude (V)",
                ax=ax[i],
                legend=False,
                alpha=0.6,
                linewidth=0.8
            )
        ax[i].set_xlim(0.0000, 0.00005)
        ax[i].set_title(c.capitalize() + " Signals")
        ax[i].set_ylabel("Amplitude (V)")
        ax[i].grid(True)

    plt.tight_layout()
    save_current_figure("noise_signals.png")


def plot_noise_mean_std(noise_data):
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    class_lst = ["gassing", "cracking", "mixed"]
    for i, c in enumerate(class_lst):
        signals = np.array([
            df[0]["Amplitude (V)"].values
            for df in noise_data[c].values()
        ])

        time = next(iter(noise_data[c].values()))[0]["Time (s)"].values

        mean_signal = signals.mean(axis=0)
        std_signal = signals.std(axis=0)

        ax[i].plot(time, mean_signal, color="blue", linewidth=2, label="Mean")
        ax[i].fill_between(
            time,
            mean_signal - std_signal,
            mean_signal + std_signal,
            color="blue",
            alpha=0.3,
            label="±1 Std"
        )
        ax[i].set_ylabel("Amplitude (V)")
        ax[i].set_title(c.capitalize() + " Signals")
        ax[i].grid(alpha=0.3)
        ax[i].set_ylim(-0.01, 0.01)
    ax[-1].set_xlabel("Time (s)")
    ax[0].legend(ncol=4)
    save_current_figure("noise_mean_std.png")


# ---------------------------------------------------------------------------
# FFT / wavelet transform
# ---------------------------------------------------------------------------
def run_transform_df(input_df, method='fft', wavelet_w=6.0, total_scales=128):
    time = input_df['Time (s)'].values
    amplitude = input_df['Amplitude (V)'].values

    N = len(amplitude)
    dt = time[1] - time[0]
    fs = 1.0 / dt

    if method == 'fft':
        fft_vals = np.fft.rfft(amplitude)
        freqs = np.fft.rfftfreq(N, d=dt)
        magnitude = np.abs(fft_vals) / N

        if N % 2 == 0:
            magnitude[1:-1] *= 2
        else:
            magnitude[1:] *= 2

        return freqs, magnitude

    elif method == 'wavelet':
        # 1. Hz
        f_min = 1.0 / (N * dt)
        f_max = fs / 2.0
        freqs = np.linspace(f_min, f_max, total_scales)

        # 2. Morlet Wavelet
        # pywt에서 'cmorB-C' (B: bandwidth, C: center frequency)
        wavelet_name = f'cmor1.0-{wavelet_w}'

        # pywt.frequency2scale  Hz -> Scale
        scales = pywt.frequency2scale(wavelet_name, freqs) * fs

        # 3. Continuous Wavelet Transform
        # cwt_matrix shape: (total_scales, N)
        cwt_matrix, _ = pywt.cwt(amplitude, scales, wavelet_name, sampling_period=dt)
        magnitude = np.abs(cwt_matrix)

        return freqs, magnitude

    else:
        raise ValueError("method 옵션은 'fft' 또는 'wavelet' 이어야 합니다.")


def run_transform(input_dict, method='fft', **kwargs):
    freqs_lst_lst = []
    magnitude_lst_lst = []
    key_lst = []
    for key, df_lst in input_dict.items():
        freqs_lst, magnitude_lst = [], []
        for df in df_lst:
            freqs, magnitude = run_transform_df(df, method=method, **kwargs)
            freqs_lst.append(freqs)
            magnitude_lst.append(magnitude)
        key_lst.append(key)
        freqs_lst_lst.append(freqs_lst)
        magnitude_lst_lst.append(magnitude_lst)

    return key_lst, np.array(freqs_lst_lst) /1e3 , np.array(magnitude_lst_lst)


def transform_all(signal_data, method):
    keys_dict = {}
    freq_dict = {}
    magnitude_dict = {}

    for key in signal_data:
        log(f"  {method} transform for '{key}' ({len(signal_data[key])} files)")
        keys, freqs, magnitude = run_transform(signal_data[key], method=method)
        keys_dict[key] = keys
        freq_dict[key] = freqs
        magnitude_dict[key] = magnitude
    return keys_dict, freq_dict, magnitude_dict


def plot_fft_transformation(i, axs, key, freqs, mags):
    freq = freqs[:,i,:].mean(axis=0)
    mag = mags[:,i,:].mean(axis=0)
    mag_std = mags[:,i,:].std(axis=0)
    color_map = {'normal': 'blue', 'gassing': 'green', 'cracking': 'red'}
    if i==0:
        label = r' (0-50$\mu$ Sec)'
    else:
        label = r' (50-100$\mu$ Sec)'

    axs[i].plot(freq, mag, alpha=0.9, label=key.capitalize()+ label, color=color_map.get(key, 'black'), linestyle='-', linewidth=1.5)
    axs[i].fill_between(freq, mag - mag_std, mag + mag_std,
            alpha=0.15, color=color_map.get(key, 'black'), edgecolor='none')
    axs[i].set_xlim(0, 1000)
    axs[i].set_ylim(0, 0.014)
    axs[i].set_title('Spectrum Comparison of Normal, Gassing, and Cracking Signals')
    axs[i].set_xlabel('Frequency (kHz)')
    axs[i].set_ylabel('Magnitude (V)')
    axs[i].grid(True)
    axs[i].legend()


def plot_spectra(freq_dict, magnitude_dict, method):
    if method == "fft":
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))

        for key in freq_dict:
            if key.split("_")[0] == "mixed":
                continue

            plot_fft_transformation(0, ax, key, freq_dict[key], magnitude_dict[key])
            if key != "normal":
                plot_fft_transformation(1, ax, key, freq_dict[key], magnitude_dict[key])

        ax[0].set_title(
            "Spectrum Comparison of Normal, Gassing, and Cracking Signals"
        )
        ax[1].set_title("Spectrum Comparison of Decay Signals")
        plt.tight_layout()
        save_current_figure("fft_spectra.png")

    elif method == "wavelet":
        if PULSE_DETECTION_BOOL:
            figure_shape = (1, 3)
            figure_size = (15, 4)
            panel_specs = [
                (0, 0, "normal", 0, "Normal", 0, 50),
                (0, 1, "gassing", 0, "Gassing onset window", 0, 50),
                (0, 2, "cracking", 0, "Cracking onset window", 0, 50),
            ]
        else:
            figure_shape = (1, 3)
            figure_size = (15, 4)
            panel_specs = [
                (0, 0, "normal", 0, "Normal (0–50 us)", 0, 50),
                (0, 1, "gassing", 0, "Gassing (0–50 us)", 0, 50),
                (0, 2, "cracking", 0, "Cracking (0–50 us)", 0, 50),
            ]
        fig, axes = plt.subplots(
            *figure_shape,
            figsize=figure_size,
            sharey=True,
            constrained_layout=True,
            squeeze=False,
        )

        panels = []
        vmax = 0.0

        for row, col, key, segment, title, t_start, t_end in panel_specs:
            freqs = np.asarray(freq_dict[key])
            mags = np.asarray(magnitude_dict[key])

            if segment >= freqs.shape[1] or segment >= mags.shape[1]:
                raise ValueError(f"{key}에 구간 인덱스 {segment}가 없습니다.")

            #  magnitude
            frequency = freqs[:, segment, :].mean(axis=0)
            image_data = mags[:, segment, :, :].mean(axis=0)

            # 500
            time_us = np.linspace(t_start, t_end, image_data.shape[1])

            vmax = max(vmax, float(np.nanmax(image_data)))
            panels.append((row, col, title, time_us, frequency, image_data))

        mesh = None
        for row, col, title, time_us, frequency, image_data in panels:
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                time_us,
                frequency,
                image_data,
                shading="auto",
                cmap="turbo",
                vmin=0,
                vmax=vmax if vmax > 0 else 1,
            )
            ax.set_title(title)
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Frequency (kHz)")

        active_axes = [axes[row, col] for row, col, *_ in panel_specs]

        fig.colorbar(
            mesh,
            ax=active_axes,
            label="Mean magnitude",
            location="right",
            pad=0.04,
            shrink=0.9,
        )
        fig.suptitle("Wavelet Scalograms")
        ax.set_ylim([0,2000])
        save_current_figure("wavelet_scalograms.png")


def compute_p90_point(magnitude_dict):
    p90_array = []
    for k in magnitude_dict:
        if k != 'mixed':
            image_arr = magnitude_dict[k][:, 0].mean(axis=0)
            p90 = np.percentile(image_arr, 90)
            p90_array.append(image_arr > p90)

    return np.array(p90_array).any(axis=0)


def select_features(magnitude_dict, method, p90_point):
    magnitude_input_dict = {}
    if method == 'fft':
        for k in magnitude_dict:
            magnitude_input_dict[k] = magnitude_dict[k]
    else:
        for k in magnitude_dict:
            magnitude_input_dict[k] = magnitude_dict[k][:, :, p90_point]
    return magnitude_input_dict


def plot_p90_mask(p90_point, N):
    time_us = np.linspace(0, 250, N)

    plt.figure(figsize=(6, 6))

    binary_cmap = ListedColormap(["#1f77b4", "#ff7f0e"])

    plt.imshow(
        p90_point.astype(int),
        aspect="auto",
        origin="lower",
        cmap=binary_cmap,
        extent=[time_us[0], time_us[-1], 0, 5000],
        vmin=0,
        vmax=1,
    )

    plt.ylim([0, 2000])
    plt.xlabel(r"Time ($\mu s$)")
    plt.ylabel("Frequency (kHz)")
    plt.title("P90 Activation Mask over Time (Binary)")

    plt.gca().set_box_aspect(1)

    cbar = plt.colorbar(ticks=[0, 1], shrink=0.8)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    cbar.set_ticklabels(["False (0)", "True (1)"])

    plt.tight_layout()
    save_current_figure("p90_mask.png")


# ---------------------------------------------------------------------------
# AR model
# ---------------------------------------------------------------------------
def conditional_ar_design(sequence, order):
    sequence = np.asarray(sequence, dtype=float)
    target = sequence[order:]
    lagged = [sequence[order - lag : len(sequence) - lag] for lag in range(1, order + 1)]
    design = np.column_stack([np.ones(len(target)), *lagged])
    return design, target


def fit_gaussian_ar_mle(sequences, order):
    designs_and_targets = [conditional_ar_design(sequence, order) for sequence in sequences]
    design = np.vstack([item[0] for item in designs_and_targets])
    target = np.concatenate([item[1] for item in designs_and_targets])

    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    innovation_variance = max(np.mean(residual**2), np.finfo(float).eps)
    return coefficients, innovation_variance


def gaussian_ar_log_likelihood(sequence, coefficients, innovation_variance, order):
    design, target = conditional_ar_design(sequence, order)
    residual = target - design @ coefficients
    mean_squared_error = np.mean(residual**2)
    return -0.5 * (1/len(residual)) * (np.log(2 * np.pi * innovation_variance) + mean_squared_error / innovation_variance)


def run_ar_model(signal_data, cluster_data, sample_key_to_index, train_idx, valid_idx, test_idx):
    full_gassing = recombine_dfs(signal_data["gassing"])
    full_cracking = recombine_dfs(signal_data["cracking"])
    ar_class_sources = [signal_data["normal"], full_gassing, full_cracking]
    merged_cluster_data = cluster_data['gassing'] | cluster_data['cracking']
    ar_sequences = []
    ar_labels = []
    ar_sample_indices = []
    file_key_lst = []
    full_sequences = []
    for class_id, class_sources in enumerate(ar_class_sources):
        for file_key, dataframe in class_sources.items():
            if isinstance(dataframe, list):
                if not dataframe:
                    continue
                if not all(isinstance(part, pd.DataFrame) for part in dataframe):
                    raise TypeError(
                        f"{AR_CLASS_NAMES[class_id]} / {file_key}: "
                        "It is not a list of dataframes."
                    )
                dataframe = pd.concat(dataframe, ignore_index=True)
            sequence = dataframe["Amplitude (V)"].to_numpy(dtype=float)
            if len(sequence) > AR_ORDER and np.isfinite(sequence).all():
                file_key_lst.append(file_key)
                ar_sequences.append(sequence)
                full_sequences.append(merged_cluster_data[file_key])
                ar_labels.append(class_id)
                ar_sample_indices.append(sample_key_to_index[(class_id, file_key)])

    ar_labels = np.asarray(ar_labels)
    ar_sample_indices = np.asarray(ar_sample_indices)
    ar_train_idx = train_idx
    ar_valid_idx = valid_idx
    ar_test_idx = test_idx
    train_sequences = [ar_sequences[index] for index in ar_train_idx]
    test_sequences = [ar_sequences[index] for index in ar_test_idx]
    test_filekeys = [file_key_lst[index] for index in ar_test_idx]
    test_full_sequence = [full_sequences[index] for index in ar_test_idx]
    ar_y_train = ar_labels[ar_train_idx]
    ar_y_test = ar_labels[ar_test_idx]

    ar_class_models = {}
    for class_id, class_name in enumerate(AR_CLASS_NAMES):
        class_train_sequences = [
            sequence for sequence, label in zip(train_sequences, ar_y_train)
            if label == class_id
        ]
        ar_class_models[class_id] = fit_gaussian_ar_mle(class_train_sequences, AR_ORDER)

    ar_log_likelihoods = np.array([
        [
            gaussian_ar_log_likelihood(sequence, *ar_class_models[class_id], AR_ORDER)
            for class_id in range(len(AR_CLASS_NAMES))
        ]
        for sequence in test_sequences
    ])
    ar_y_pred = ar_log_likelihoods.argmax(axis=1)

    print(f"AR({AR_ORDER}) / ARMA({AR_ORDER}, 0) shared file-level split")
    print(f"Training sequences: {len(ar_train_idx)}; test sequences: {len(ar_test_idx)}")
    print(f"Test accuracy: {accuracy_score(ar_y_test, ar_y_pred):.3f}")
    print(classification_report(
        ar_y_test,
        ar_y_pred,
        labels=np.arange(len(AR_CLASS_NAMES)),
        target_names=AR_CLASS_NAMES,
        zero_division=0,
    ))
    for class_id, class_name in enumerate(AR_CLASS_NAMES):
        coefficients, innovation_variance = ar_class_models[class_id]
        print(f"{class_name}: intercept={coefficients[0]:.6g}, AR coefficients={coefficients[1:]}, "
              f"innovation variance={innovation_variance:.6g}")

    return ar_y_test, ar_log_likelihoods, test_sequences, test_filekeys, test_full_sequence


def save_figs(model, y_test, probs, sequences, filekeys, path_default, max_figs=280):
    plt.rcParams['text.usetex'] = False

    log(f" Saving {min(max_figs, len(y_test))} {model} waveform figures")

    test_result = probs.argmax(axis=1)

    for i in range(min(max_figs, len(y_test))):
        fig, ax = plt.subplots(figsize=(8, 4))
        
        if y_test[i] == test_result[i]:
            color = 'green'
        else:
            color = 'red'
            
        label = f"True Class: {AR_CLASS_NAMES[y_test[i]]}, Predicted: {AR_CLASS_NAMES[test_result[i]]}"
        
        time_values = sequences[i]['Time (s)'].to_numpy()
        amplitude_values = sequences[i]['Amplitude (V)'].to_numpy()
        ax.plot(time_values, amplitude_values, label=label, color=color)

        if y_test[i] == 0:
            used_indices = np.flatnonzero(time_values < time_threshold)
            segment_label = "Noise segment used"
            marker_label = "Noise/signal boundary"
            threshold_indices = np.flatnonzero(time_values >= time_threshold)
            marker_time = (
                time_values[threshold_indices[0]]
                if len(threshold_indices)
                else None
            )
        elif PULSE_DETECTION_BOOL:
            try:
                anchor = onset_anchor_index(amplitude_values, time_values)
                onset = onset_picker(amplitude_values, time_values)
            except ValueError as error:
                log(f"  Cannot mark used segment for {filekeys[i]}: {error}")
                used_indices = np.array([], dtype=int)
                marker_time = None
                threshold_marker_time = None
            else:
                used_indices = np.arange(onset, min(onset + N, len(time_values)))
                marker_time = time_values[onset]
                threshold_marker_time = time_values[anchor]
            segment_label = "Onset segment used"
            marker_label = (
                "Detected onset (AIC)"
                if AIC_BOOL
                else "Detected onset (|amplitude| >= 0.02 V after time threshold)"
            )
        else:
            used_indices = np.flatnonzero(time_values >= time_threshold)
            used_indices = used_indices[:N]
            segment_label = "First post-threshold segment used"
            marker_label = "Time threshold"
            marker_time = (
                time_values[used_indices[0]] if len(used_indices) else None
            )
            threshold_marker_time = None

        if len(used_indices):
            start_index = int(used_indices[0])
            end_index = int(used_indices[-1])
            ax.axvspan(
                time_values[start_index],
                time_values[end_index],
                color="gold",
                alpha=0.3,
                label=segment_label,
            )
            if marker_time is not None:
                ax.axvline(
                    marker_time,
                    color="blue",
                    linestyle="--",
                    linewidth=1.2,
                    label=marker_label,
                )
            if y_test[i] != 0 and PULSE_DETECTION_BOOL and threshold_marker_time is not None:
                ax.axvline(
                    threshold_marker_time,
                    color="orange",
                    linestyle=":",
                    linewidth=1.2,
                    label=(
                        "Envelope >= 20% of max"
                        if AIC_BOOL
                        else "0.02 V after time threshold"
                    ),
                )

        ax.set_xlim(0.0, 0.00005*6)
        ax.set_ylim(-0.08, 0.08)
        ax.grid(True)
        ax.legend()
        ax.set_title("Waveform of " + filekeys[i]) # 타이틀 선언 위치를 위로 올렸습니다.
        path = os.path.join(path_default, model, str(y_test[i]))
        os.makedirs(path, exist_ok=True)
        file_name = f"{filekeys[i]}.png"
        save_path = os.path.join(path, file_name)

        # 3. 명확하게 해당 fig 객체만 저장하고 닫습니다.
        fig.savefig(save_path, dpi=300)
        plt.close(fig) 

# ---------------------------------------------------------------------------
# Spectrum classifiers
# ---------------------------------------------------------------------------
def train_gaussian_nb(X_train, y_train, X_test, y_test, confirmed_classes, y, sample_keys, train_idx, test_idx):
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Training Accuracy:", model.score(X_train, y_train))
    print("Test Accuracy:", model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    class_ids = np.arange(len(confirmed_classes))
    class_counts = np.bincount(y, minlength=len(confirmed_classes))

    print("Classes:", dict(enumerate(confirmed_classes)))
    print("Class counts:", dict(zip(confirmed_classes, class_counts)))
    print("Majority-class baseline:", class_counts.max() / len(y))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=class_ids))
    print(classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=confirmed_classes,
        zero_division=0,
    ))

    train_file_keys = {sample_keys[index][1] for index in train_idx}
    test_file_keys = {sample_keys[index][1] for index in test_idx}
    print("Source files present in both splits:", len(train_file_keys & test_file_keys))
    return model


def train_xgboost(X_train, y_train, X_valid, y_valid, X_test, y_test, confirmed_classes):
    xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(np.unique(y_train)),
    n_estimators=1000,        
    max_depth=4,
    learning_rate=0.05,
    early_stopping_rounds=30, 
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2,           
    reg_lambda=1,        
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    )
    xgb_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
    )

    print("XGBoost training accuracy:", xgb_model.score(X_train, y_train))
    print("XGBoost test accuracy:", xgb_model.score(X_test, y_test))
    print(classification_report(
        y_test,
        xgb_model.predict(X_test),
        target_names=confirmed_classes,
        zero_division=0,
    ))
    return xgb_model


def classify_segments(models, magnitude_input_dict, keys_dict, cluster):
    result_dict = {}
    prob_dict = {}
    entropy_dict = {}
    for m in MODEL_LST:
        result_dict[m] = {}
        prob_dict[m] = {}
        entropy_dict[m] = {}
    for i, magnitude_array in enumerate(magnitude_input_dict[cluster]):
        key = keys_dict[cluster][i]
        for m in MODEL_LST:
            prob_dict[m][key] = []
            entropy_dict[m][key] = None
        for spectrum in magnitude_array:
            spectrum = np.array(spectrum).reshape(1, -1)
            for m in MODEL_LST:
                probs = models[m].predict_proba(spectrum)[0]
                prob_dict[m][key].append(probs)
                entropy_dict[m][key] = entropy(probs, base=2)
        for m in MODEL_LST:
            result_dict[m][key] = np.array(prob_dict[m][key]).argmax(axis=1)
    return result_dict, prob_dict, entropy_dict


def plot_classification_regions(key, model, result_dict, entropy_dict, cluster_data, path, N, save = False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    df = cluster_data
    time_arr = df['Time (s)'].values
    amp_arr = df['Amplitude (V)'].values

    signal_line = ax.plot(time_arr, amp_arr, color='black', linewidth=0.8, alpha=0.7, label='AE Signal')

    color_map = {0: 'green', 1: 'orange', 2: 'red'}
    label_map = {0: 'Normal (0)', 1: 'Gassing (1)', 2: 'Cracking (2)'}
    used_labels = set()
    
    if PULSE_DETECTION_BOOL:
        cls_id = int(result_dict[0])
        anchor = onset_anchor_index(amp_arr, time_arr)
        onset = onset_picker(amp_arr, time_arr)
        t_start = time_arr[onset]
        t_end = time_arr[onset + N - 1]
        ax.axvline(
            time_arr[anchor],
            color="orange",
            linestyle=":",
            linewidth=1.2,
            label=(
                "Envelope >= 20% of max"
                if AIC_BOOL
                else "0.02 V after time threshold"
            ),
        )
        ax.axvline(
            time_arr[onset],
            color="blue",
            linestyle="--",
            linewidth=1.2,
            label=(
                "Detected onset (AIC)"
                if AIC_BOOL
                else "Detected onset (|amplitude| >= 0.02 V after time threshold)"
            ),
        )
    else:
        cls_id = int(result_dict[0])
        segment_start = np.flatnonzero(time_arr >= time_threshold)[0]
        t_start = time_arr[segment_start]
        t_end = time_arr[segment_start + N - 1]

    color = color_map.get(cls_id, 'gray')
    label = label_map.get(cls_id, f'Class {cls_id}') if cls_id not in used_labels else None

    ax.axvspan(t_start, t_end, color=color, alpha=0.25, edgecolor='none', label=label)
    used_labels.add(cls_id)
    legend_patches = [
        mpatches.Patch(color=color_map[0], alpha=0.25, label=label_map[0]),
        mpatches.Patch(color=color_map[1], alpha=0.25, label=label_map[1]),
        mpatches.Patch(color=color_map[2], alpha=0.25, label=label_map[2])
    ]
    handles = signal_line + legend_patches  
    ax.set_xlim(0.0, 0.00005*6)
    ax.set_ylim(-0.08, 0.08)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (V)', fontsize=12)
    if PULSE_DETECTION_BOOL:
        title = f'{key} (Window = {N}, Entropy = {entropy_dict:.3f})'
    else:
        title = (
            f'{key} (Window = {N}, Strid = {N}, '
            f'Entropy = {entropy_dict:.3f}'
        )
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(handles=handles, loc='upper right')
    # ax.legend(loc='upper right')
    if save:   
        path_dir = os.path.join(path, model, str(cls_id))
        os.makedirs(path_dir, exist_ok=True)
        file_path = os.path.join(path_dir, key + ".png")
        plt.savefig(file_path)
    else:
        save_current_figure(f"classification_regions_{model}_{key}.png")
    plt.close(fig)


def main():
    # -----------------------------------------------------------------------
    # Training of model
    # -----------------------------------------------------------------------
    log("[1/10] Loading labeled clusters")
    cluster_data = load_clusters(cluster_folders)
    log("[2/10] Splitting into noise / signal segments")
    noise_data, signal_data = process_data(cluster_data, N, time_threshold)

    log("[3/10] Plotting noise figures")
    plot_noise_moving_average(noise_data)
    plot_noise_signals(noise_data)
    plot_noise_mean_std(noise_data)

    signal_data["normal"] = noise_data["cracking"]#| noise_data["mixed"]

    log(f"[4/10] Running {method} transform on labeled data")
    keys_dict, freq_dict, magnitude_dict = transform_all(signal_data, method)
    log("  Plotting spectra")
    plot_spectra(freq_dict, magnitude_dict, method)

    p90_point = None
    if method == 'wavelet':
        log("  Computing P90 feature mask")
        p90_point = compute_p90_point(magnitude_dict)
        plot_p90_mask(p90_point, N)
    magnitude_input_dict = select_features(magnitude_dict, method, p90_point)

    log("[5/10] Building train/test split")
    confirmed_classes = [key for key in signal_data if key.split('_')[0] != 'mixed' and key.split('_')[-1] != 'decay']
    print("Confirmed classes:", confirmed_classes)
    X = np.vstack([magnitude_input_dict[key][:,0,:] for key in confirmed_classes])
    y = np.array([i for i, key in enumerate(confirmed_classes) for _ in range(len(magnitude_input_dict[key]))])

    sample_keys = [
        (class_id, file_key)
        for class_id, class_name in enumerate(confirmed_classes)
        for file_key in signal_data[class_name]
    ]
    if len(sample_keys) != len(y):
        raise ValueError("FFT samples and source-file keys are not aligned")

    sample_key_to_index = {key: index for index, key in enumerate(sample_keys)}
    sample_indices = np.arange(len(X))
    train_valid_idx, test_idx = train_test_split(
        sample_indices,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Train+Valid Train Valid
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=0.20,
        random_state=42,
        stratify=y[train_valid_idx],  # y of train_valid_idx
    )

    # 최종 데이터 및 타겟 분할
    X_train, X_valid, X_test = X[train_idx], X[valid_idx], X[test_idx]
    y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_valid.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # ARMA model
    log(f"[6/10] Training AR({AR_ORDER}) model")
    path_test = os.path.join(FIGURE_DIR, 'dataset-test') 
    ar_y_test, ar_log_likelihoods, test_sequences, test_filekeys, full_sequences = run_ar_model(
        signal_data, cluster_data, sample_key_to_index, train_idx, valid_idx, test_idx
    )
    save_figs('ARMA', ar_y_test, ar_log_likelihoods, full_sequences, test_filekeys, path_test)

    # Gaussian NB
    log("[7/10] Training Gaussian NB")
    model = train_gaussian_nb(X_train, y_train, X_test, y_test, confirmed_classes, y, sample_keys, train_idx, test_idx)
    save_figs('GNB', y_test, model.predict_proba(X_test), full_sequences, test_filekeys, path_test)

    # XGBoost
    log("[8/10] Training XGBoost")
    xgb_model = train_xgboost(X_train, y_train, X_valid, y_valid, X_test, y_test, confirmed_classes)
    save_figs('XGB', y_test, xgb_model.predict_proba(X_test), full_sequences, test_filekeys, path_test)

    models = {"GaussNB": model, "XGBoost": xgb_model}
    path_model = os.path.join(FIGURE_DIR, 'models', 'data.pkl') 
    os.makedirs(os.path.dirname(path_model), exist_ok=True)
    pickle.dump(models, open(path_model, 'wb'))


    # Mixed cluster
    log("[9/10] Classifying 'mixed' cluster")
    result_dict, prob_dict, entropy_dict = classify_segments(models, magnitude_input_dict, keys_dict, 'mixed')
    path_mixed = os.path.join(FIGURE_DIR, 'mixed') 
    mixed_keys = keys_dict['mixed'] if PULSE_DETECTION_BOOL else [keys_dict['mixed'][9]]
    for k in mixed_keys:
        for m in MODEL_LST:
            plot_classification_regions(
                k,
                m,
                result_dict[m][k],
                entropy_dict[m][k],
                cluster_data["mixed"][k],
                path_mixed,
                N,
                save=True,
            )

    # -----------------------------------------------------------------------
    # Unlabeled test data
    # -----------------------------------------------------------------------
    log("[Unlabeled 1/4] Loading unlabeled test data")
    cluster_data = load_clusters(unlabeled_folders)
    signal_data = process_unlabeled_data(cluster_data, N)

    log(f"[Unlabeled 2/4] Running {method} transform")
    keys_dict, freq_dict, magnitude_dict = transform_all(signal_data, method)
    magnitude_input_dict = select_features(magnitude_dict, method, p90_point)

    log("[Unlabeled 3/4] Classifying segments")
    result_dict, prob_dict, entropy_dict = classify_segments(models, magnitude_input_dict, keys_dict, 'test')

    test_keys = keys_dict['test']
    log(f"[Unlabeled 4/4] Saving classification plots for {len(test_keys)} files")
    path_drift = os.path.join(FIGURE_DIR, 'dataset-new') 
    for idx, k in enumerate(test_keys, 1):
        if idx % 50 == 0 or idx == len(test_keys):
            log(f"  {idx}/{len(test_keys)} files plotted")
        for m in MODEL_LST:
            plot_classification_regions(k, m, result_dict[m][k], entropy_dict[m][k], cluster_data["test"][k], path_drift, N, save=True)


if __name__ == "__main__":
    main()
    log("Done")
