from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report\

if __name__ == "__main__":
    N = 500
    cluster_folders = {
        "gassing": Path("Data/Cluster_0 (gassing)"),
        "cracking": Path("Data/Cluster_1 (cracking)"),
        "mixed": Path("Data/Cluster_2 (mix undecided)")
    }

    cluster_data = {
        cluster: {
            csv_file.stem: pd.read_csv(csv_file)
            for csv_file in sorted(folder.glob("*.csv"))
        }
        for cluster, folder in cluster_folders.items()
    }

    time_threshold = 0.00005
    noise_data = {}
    signal_data = {}

    for cluster_name, file_data in cluster_data.items():
        noise_data[cluster_name] = {}
        signal_data[cluster_name] = {}

        for file_name, dataframe in file_data.items():
            noise_mask = dataframe["Time (s)"] < time_threshold
            signal_mask = dataframe["Time (s)"] >= time_threshold
            noise_data[cluster_name][file_name] = dataframe.loc[noise_mask].copy()
            signal_data[cluster_name][file_name] = dataframe.loc[signal_mask].copy()

    all_noise =  noise_data["cracking"]#| noise_data["mixed"]
    all_noise_filtered = {}
    for key in all_noise.keys():
        all_noise_filtered[key] = all_noise[key]#[all_noise[key]["Time (s)"]<=first_crossing_time_lst[percentiles==75][0]]


    AR_ORDER = 8
    AR_CLASS_NAMES = ["Normal", "Gassing", "Cracking"]
    AR_CLASS_SOURCES = [all_noise_filtered, signal_data["gassing"], signal_data["cracking"]]


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
        return -0.5 * (np.log(2 * np.pi * innovation_variance) + mean_squared_error / innovation_variance)


    ar_sequences = []
    ar_labels = []
    ar_file_keys = []
    for class_id, class_sources in enumerate(AR_CLASS_SOURCES):
        for file_key, dataframe in class_sources.items():
            sequence = dataframe["Amplitude (V)"].to_numpy(dtype=float)
            if len(sequence) > AR_ORDER and np.isfinite(sequence).all():
                ar_sequences.append(sequence)
                ar_labels.append(class_id)
                ar_file_keys.append(file_key)

    ar_labels = np.asarray(ar_labels)
    ar_indices = np.arange(len(ar_sequences))
    ar_train_idx, ar_test_idx = train_test_split(
        ar_indices,
        test_size=0.25,
        random_state=42,
        stratify=ar_labels,
    )
    ar_train_sequences = [ar_sequences[index] for index in ar_train_idx]
    ar_test_sequences = [ar_sequences[index] for index in ar_test_idx]
    ar_y_train = ar_labels[ar_train_idx]
    ar_y_test = ar_labels[ar_test_idx]

    ar_class_models = {}
    for class_id, class_name in enumerate(AR_CLASS_NAMES):
        class_train_sequences = [
            sequence for sequence, label in zip(ar_train_sequences, ar_y_train)
            if label == class_id
        ]
        ar_class_models[class_id] = fit_gaussian_ar_mle(class_train_sequences, AR_ORDER)

    ar_log_likelihoods = np.array([
        [
            gaussian_ar_log_likelihood(sequence, *ar_class_models[class_id], AR_ORDER)
            for class_id in range(len(AR_CLASS_NAMES))
        ]
        for sequence in ar_test_sequences
    ])
    ar_y_pred = ar_log_likelihoods.argmax(axis=1)

    print(f"AR({AR_ORDER}) / ARMA({AR_ORDER}, 0) file-level split")
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