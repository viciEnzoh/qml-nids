import os
import pickle
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pennylane as qml
import numpy as np
from torchinfo import summary as model_summary

import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
import time
import copy
from collections import Counter
import math

import argparse

def preprocess_data(X, scaler, N_FEATURES=4, N_PACKETS=10, TIME_THRESHOLD=20000):
    X_proc = X.copy().astype(np.float32)
    N = X_proc.shape[0]

    for i in range(N):
        flow = X_proc[i]

        pad_start = np.where(flow[:, 0] == -1)[0]
        valid_len = pad_start[0] if len(pad_start) > 0 else len(flow)

        # cumulative sum of IAT
        cumsum_iat = np.cumsum(flow[:valid_len, 3])
        exceed = np.where(cumsum_iat >= TIME_THRESHOLD)[0]

        if len(exceed) > 0:
            cut_idx = exceed[0]
            flow[cut_idx:] = -1

    X_proc = scaler.transform(np.reshape(X_proc, [-1, N_FEATURES]))
    X_proc = np.reshape(X_proc, [-1, X.shape[1], N_FEATURES])
    return X_proc[:, :N_PACKETS, :]

class UsedDataset(Dataset):
    def __init__(self, X, y, DEVICE):
        self.X = torch.FloatTensor(X).double().to(DEVICE) # Remove to(DEVICE) if dataset doesn't fit in memory
        self.y = torch.LongTensor(y).to(DEVICE) # Remove to(DEVICE) if dataset doesn't fit in memory

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def evaluate(model, loader, criterion, DEVICE):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def select_model(model_name, NUM_CLASSES, N_QUBITS=5, N_LAYERS=3, N_FEATURES=4, N_PACKETS=10, dev_name='default.qubit', NUM_REUPLOADS=3, N_SHOTS=1024):
    if model_name == "ampe_fixed_probs":
        from qml_models import AmpeDenseModel as QModel
        is_quantum = True
        model = QModel(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            n_features=N_FEATURES,
            n_packets=N_PACKETS,
            num_classes=NUM_CLASSES,
            dev_name = dev_name,
            n_shots=N_SHOTS
        )
    elif model_name == "hybrid_ampe_probs":
        from qml_models import AmpeCNNLSTMModel as QModel
        is_quantum = True
        model = QModel(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            n_features=N_FEATURES,
            n_packets=N_PACKETS,
            num_classes=NUM_CLASSES,
            dev_name = dev_name,
            n_shots=N_SHOTS
        )
    elif model_name == "ange_probs":
        from qml_models import AngeDenseModel as QModel
        is_quantum = True
        model = QModel(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            n_features=N_FEATURES,
            n_packets=N_PACKETS,
            num_classes=NUM_CLASSES,
            dev_name = dev_name,
            n_shots=N_SHOTS
        )
    elif model_name == "hybrid":
        from qml_models import CNNLSTMModel as CModel
        is_quantum = False
        model = CModel(
            n_features=N_FEATURES,
            n_packets=N_PACKETS,
            num_classes=NUM_CLASSES
        )

    return model, is_quantum

def load_config(path):
    config = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("(") and line.endswith(")"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                config[key] = value

    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Main entry point")

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=2025,
        help="Random seed"
    )

    parser.add_argument(
        "--dev-name",
        type=str,
        default="default.qubit",
        choices=[
            "default.qubit",
            "lightning.qubit",
            "lightning.gpu",
            "fake_lagos",
            "fake_nairobi",
            "fake_perth",
            "fake_fractbackend",
            "fake_yorktown",
            "fake_london",
        ],
        help="Name of the device for quantum circuit execution (default: default.qubit)"
    )

    parser.add_argument(
        "--num-shots",
        type=int,
        default=1024,
        help="No. of shots, in case of (quantum) noisy device (default: 1024)"
    )

    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        choices=["0", "1"],
        help="ID of the GPU executing the task (default: 0)"
    )

    parser.add_argument(
        "--exp-path",
        type=str,
        required=True,
        help="Path of the directory where is the experiment"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Name of the folder collecting the results of the analysis"
    )

    parser.add_argument(
        "--thresholds-list",
        nargs='+',
        type=int,
        help="List of integer thresholds"
    )

    return parser.parse_args()

def main():

    args = parse_args()

    dataset_path = args.dataset_path
    dev_name = args.dev_name
    gpu_id = args.gpu_id

    exp_path = args.exp_path
    output_dir = args.output_dir

    model_name = exp_path.split('/')[-1]
    print(f"Model name: {model_name}")
    config = load_config(f"{exp_path}/config.txt")
    print("Train model configuration (dict):")
    print(config)

    RANDOM_SEED = config["RANDOM_SEED"]
    NUM_FOLDS = config["NUM_FOLDS"]
    SUB_SAMPLE_RATIO = config["SUB_SAMPLE_RATIO"]
    BATCH_SIZE = config["BATCH_SIZE"]

    split_seed = 2025


    N_PACKETS = config["N_PACKETS"]
    N_FEATURES = config["N_FEATURES"]

    N_QUBITS = config["N_QUBITS"]
    N_LAYERS = config["N_LAYERS"]
    N_SHOTS = args.num_shots

    # CPU, or CUDA dev
    DEVICE = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    output_dir = f"{args.output_dir}/{model_name}"
    os.makedirs(f"{output_dir}", exist_ok=True)

    cfg_file = f"{output_dir}/config.txt"

    with open(cfg_file, "w") as f:
        f.write(f"model_name = {model_name}\n")
        f.write(f"RANDOM_SEED = {RANDOM_SEED}\n\n")
        f.write(f"NUM_FOLDS = {NUM_FOLDS}\n\n")
        f.write(f"BATCH_SIZE = {BATCH_SIZE}\n\n")
        f.write(f"SUB_SAMPLE_RATIO = {SUB_SAMPLE_RATIO}\n\n")
        f.write(f"(Split seed is fixed to {split_seed})\n\n")
        
        f.write(f"N_PACKETS = {N_PACKETS}\n")
        f.write(f"N_FEATURES = {N_FEATURES}\n\n")
        
        f.write(f"N_QUBITS = {N_QUBITS}\n")
        f.write(f"N_LAYERS = {N_LAYERS}\n")
        f.write(f"(Quantum) dev_name = {dev_name}\n\n")
        
        f.write(f"DEVICE (cuda or gpu) = {DEVICE}\n")


    dev_name = args.dev_name

    thresholds = args.thresholds_list


    # Reproducibilty
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Quantum device: {dev_name}")
    if dev_name.startswith("fake"): print(f"No. of shots: {N_SHOTS}")

    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    VAL_SIZE = 0.2
    TEST_SIZE = 0.2

    print("Loading dataset...")
    with open(dataset_path, "rb") as f:
        X_raw = np.array(pickle.load(f), dtype=np.float32) # Convert to numpy array
        y_raw = np.array(pickle.load(f)) # Convert to numpy array

    print(f"Data shape: {X_raw.shape}")
    print(f"Labels shape: {len(y_raw)}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    NUM_CLASSES = len(le.classes_)
    print(f"Number of classes: {NUM_CLASSES}")
    print("Classes:", le.classes_)
    ordinal_classes = np.unique(y_encoded)
    labels_map_file = f"{output_dir}/labels_map.json"
    dictionary = {}
    for l,c in zip(ordinal_classes, le.inverse_transform(ordinal_classes)):
        dictionary[int(l)] = c
    with open(labels_map_file, 'w') as f:
        json.dump(dictionary, f)


    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=split_seed)
        
    for threshold in thresholds:

        print(f"Started inference for time-to-insight ({threshold}ms)")
        threshold_dir = f"{output_dir}/tti-threshold_{threshold}"
        i = 1    
        for train, test in kfold.split(X_raw, y_encoded):
            print(f"FOLD {i}/{NUM_FOLDS}")
            fold_dir = f"{threshold_dir}/fold_{i}"
            os.makedirs(f"{fold_dir}", exist_ok=True)
                    
            if (SUB_SAMPLE_RATIO < 1.0): train, _= train_test_split(train, train_size=SUB_SAMPLE_RATIO, stratify=y_encoded[train], random_state=split_seed)
            X_temp = X_raw[train]
            y_temp = y_encoded[train]


            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=split_seed)
                    
            X_test = X_raw[test]
            y_test = y_encoded[test]
            print(f"Test shape:  {X_test.shape}  ({len(X_test)/len(X_raw):.1%})")

            print("Preprocessing data...")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(np.reshape(X_temp, [-1, N_FEATURES]))
            X_train_proc = preprocess_data(X_train, scaler, TIME_THRESHOLD=threshold)
            X_val_proc = preprocess_data(X_val, scaler, TIME_THRESHOLD=threshold)
            X_test_proc = preprocess_data(X_test, scaler, TIME_THRESHOLD=threshold)


            print("Preprocessing complete.")
            print("Test shape:", X_test_proc.shape)


            train_dataset = UsedDataset(X_train_proc, y_train, DEVICE)
            val_dataset = UsedDataset(X_val_proc, y_val, DEVICE)
            test_dataset = UsedDataset(X_test_proc, y_test, DEVICE)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            model, is_quantum = select_model(
                model_name,
                NUM_CLASSES,
                N_QUBITS=N_QUBITS,
                N_LAYERS=N_LAYERS,
                N_FEATURES=N_FEATURES,
                N_PACKETS=N_PACKETS,
                dev_name=dev_name,
                N_SHOTS=N_SHOTS
            )
            model = model.to(DEVICE).double()
            print(model.get_model_name())
            print(model)

            start_time = time.time()
            
            model_path = f"{exp_path}/fold_{i}/model.pth"
            state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
            model.load_state_dict(state_dict)

            model.eval()
            all_preds = []
            all_labels = []
            all_soft_outputs = []
            softmax = torch.nn.Softmax(dim=1)

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(DEVICE)
                    outputs = model(inputs)
                    probs = softmax(outputs)
                    soft_max, predicted = probs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_soft_outputs.extend(probs.cpu().numpy())

            with open(f"{fold_dir}/soft_output.dat", "w") as f:
                f.write("Actual\tSoft-output\n")
                for label, soft in zip(all_labels, all_soft_outputs):
                    soft_str = ",".join(f"{x:.10f}" for x in soft)
                    f.write(f"{label}\t{soft_str}\n")

            with open(f"{fold_dir}/hard_output.dat", "w") as f:
                f.write("Actual\tPrediction\n")
                for label, pred in zip(all_labels, all_preds):
                    f.write(f"{label}\t{pred}\n")

            report_dict = classification_report(all_labels, all_preds, target_names=le.classes_, labels=np.arange(NUM_CLASSES), digits=4, zero_division=0)
            with open(f"{fold_dir}/classification_report.txt", "w") as f:
                f.write(report_dict)
            total_time = time.time() - start_time
            print(f"Time to evaluate fold {i}: {total_time}s")
            i=i+1
            print("="*20)

if __name__ == "__main__":
    main()