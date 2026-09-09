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
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import time
import copy
from collections import Counter
import math
import argparse

def preprocess_data(X, scaler, N_FEATURES=4, N_PACKETS=10):
    X_proc = X.copy().astype(np.float32)
    X_proc = X_proc[:, :N_PACKETS, :]
    X_proc = scaler.transform(np.reshape(X_proc, [-1, N_FEATURES]))
    X_proc = np.reshape(X_proc, [-1, N_PACKETS, N_FEATURES])
    return X_proc

class UsedDataset(Dataset):
    def __init__(self, X, y, DEVICE):
        self.X = torch.FloatTensor(X).double().to(DEVICE) # Remove to(DEVICE) if dataset doesn't fit in memory
        self.y = torch.LongTensor(y).to(DEVICE) # Remove to(DEVICE) if dataset doesn't fit in memory

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_epoch(model, loader, criterion, optimizer, DEVICE):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, DEVICE):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels


def select_model(model_name, NUM_CLASSES, N_QUBITS=5, N_LAYERS=3, N_FEATURES=4, N_PACKETS=10, dev_name='default.qubit'):
    if model_name == "ampe_fixed_probs":
        from qml_models import AmpeDenseModel as QModel
        is_quantum = True
        model = QModel(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            n_features=N_FEATURES,
            n_packets=N_PACKETS,
            num_classes=NUM_CLASSES,
            dev_name = dev_name
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
            dev_name = dev_name
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
            dev_name = dev_name
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
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to use"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs considered for training (default: 100)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size considered for training (default: 50)"
    )

    parser.add_argument(
        "--num-packets",
        type=int,
        default=10,
        help="Number of packets considered per-biflow (default: 10)"
    )

    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number K of folds for K-fold cross validation (default: 5)"
    )

    parser.add_argument(
        "--num-qubits",
        type=int,
        default=5,
        help="Number of qubits of quantum layer circuit (default: 5)"
    )

    parser.add_argument(
        "--sub-sample-ratio",
        type=float,
        default=1.0,
        help="Ratio of (train) data set to consider over the total (default: 1.0)"
    )

    parser.add_argument(
        "--dev-name",
        type=str,
        default="default.qubit",
        choices=["default.qubit", "lightning.qubit", "lightning.gpu"],
        help="Name of the device for quantum circuit execution (default: default.qubit)"
    )

    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        # choices=["0", "1"],
        help="ID of the GPU executing the task (default: 0)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Name of the folder collecting the results of the training"
    )

    return parser.parse_args()

def main():

    args = parse_args()

    model_name = args.model_name

    DATA_PATH = args.dataset_path
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = 1e-3
    ENABLE_EARLY_STOPPING = True
    PATIENCE = 10
    RANDOM_SEED = args.random_seed
    NUM_FOLDS = args.num_folds
    SUB_SAMPLE_RATIO = args.sub_sample_ratio

    split_seed = 2025


    N_PACKETS = args.num_packets
    N_FEATURES = 4

    N_QUBITS = args.num_qubits
    N_LAYERS = 3

    dev_name = args.dev_name
    print(f"Quantum device: {dev_name}")

    gpu_id = args.gpu_id
    DEVICE = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    output_dir = f"{args.output_dir}/{model_name}"
    os.makedirs(f"{output_dir}", exist_ok=True)

    cfg_file = f"{output_dir}/config.txt"

    with open(cfg_file, "w") as f:
        f.write(f"model_name = {model_name}\n")
        f.write(f"DATA_PATH = {DATA_PATH}\n")
        f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
        f.write(f"EPOCHS = {EPOCHS}\n")
        f.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
        f.write(f"ENABLE_EARLY_STOPPING = {ENABLE_EARLY_STOPPING}\n")
        f.write(f"PATIENCE = {PATIENCE}\n")
        f.write(f"RANDOM_SEED = {RANDOM_SEED}\n\n")
        f.write(f"NUM_FOLDS = {NUM_FOLDS}\n\n")
        f.write(f"SUB_SAMPLE_RATIO = {SUB_SAMPLE_RATIO}\n\n")
        f.write(f"(Split seed is fixed to {split_seed})\n\n")
        
        f.write(f"N_PACKETS = {N_PACKETS}\n")
        f.write(f"N_FEATURES = {N_FEATURES}\n\n")
        
        f.write(f"N_QUBITS = {N_QUBITS}\n")
        f.write(f"N_LAYERS = {N_LAYERS}\n")
        f.write(f"(Quantum) dev_name = {dev_name}\n\n")
        
        f.write(f"DEVICE (cuda or gpu) = {DEVICE}\n")

    # Reproducibilty
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading dataset...")
    with open(DATA_PATH, "rb") as f:
        X_raw = np.array(pickle.load(f), dtype=np.float32)
        y_raw = np.array(pickle.load(f))

    print(f"Data shape: {X_raw.shape}")
    print(f"Labels shape: {len(y_raw)}")
    print(np.unique(y_raw))

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

    VAL_SIZE = 0.2
    TEST_SIZE = 0.2

    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=split_seed) # maybe split_seed useless, in case replace with RANDOM_SEED
    i = 1
    for train, test in kfold.split(X_raw, y_encoded):
        print(f"FOLD {i}/{NUM_FOLDS}")
        fold_dir = f"{output_dir}/fold_{i}"
        os.makedirs(f"{fold_dir}", exist_ok=True)
                
        if (SUB_SAMPLE_RATIO < 1.0): train, _= train_test_split(train, train_size=SUB_SAMPLE_RATIO, stratify=y_encoded[train], random_state=split_seed)
        X_temp = X_raw[train]
        y_temp = y_encoded[train]
                
        X_test = X_raw[test]
        y_test = y_encoded[test]

        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=split_seed) # preset random_state

        print(f"Train shape: {X_train.shape} ({len(X_train)/len(X_raw):.1%})")
        print(f"Val shape:   {X_val.shape}   ({len(X_val)/len(X_raw):.1%})")
        print(f"Test shape:  {X_test.shape}  ({len(X_test)/len(X_raw):.1%})")


        print("Preprocessing data...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.reshape(X_train, [-1, N_FEATURES]))
        X_train_proc = preprocess_data(X_train, scaler)
        X_val_proc = preprocess_data(X_val, scaler)
        X_test_proc = preprocess_data(X_test, scaler)

        print("Preprocessing completed.")
        print("Train shape:", X_train_proc.shape)
        print("Val shape:", X_val_proc.shape)
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
            dev_name=dev_name
        )

        # model = torch.nn.DataParallel(model)
        model = model.to(DEVICE).double()
        print(model.get_model_name())
        print(model)

        # Draw quantum circuit
        if is_quantum:
            q_circuit = model.q_layer.qnode
            q_input = N_QUBITS if 'ange' in model_name else 2**N_QUBITS
            dummy_x = torch.zeros(q_input)
            q_weights = model.q_layer.weights
            q_circuit_draw, _ = qml.draw_mpl(q_circuit)(dummy_x, q_weights)
            q_circuit_draw.savefig(f"{output_dir}/quantum_circuit.png", dpi=600)


        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters count: {total_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

        print(f"[INFO] Training is started. {model.get_model_name()}")
        start_time = time.time()

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(EPOCHS):
            start_epoch_time = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, DEVICE)
            end_epoch_time = time.time()
            epoch_duration = end_epoch_time - start_epoch_time

            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1}/{EPOCHS} | "
                f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                f" | Time: {epoch_duration:.2f}s (FOLD {i}/{NUM_FOLDS})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"  -> Validation loss improved. Model saved.")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}", f"/ {PATIENCE}" if ENABLE_EARLY_STOPPING else "")


            if patience_counter >= PATIENCE and ENABLE_EARLY_STOPPING:
                print("Early stopping triggered.")
                break

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.2f} minutes.")


        last_model = copy.deepcopy(model)
        model.load_state_dict(best_model_wts)

        df_history = pd.DataFrame(history)
        df_history.to_csv(f"{fold_dir}/training_history.csv", index=False)

        torch.save(model.state_dict(), f"{fold_dir}/model.pth")

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
                all_labels.extend(labels.cpu().numpy()) # Replace with all_labels.extend(labels.numpy()) if using CPU
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

        i=i+1
        print("="*20)


if __name__ == "__main__":
    main()