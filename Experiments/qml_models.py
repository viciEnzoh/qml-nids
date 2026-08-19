import torch
import pennylane as qml
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit_ibm_runtime.fake_provider as fp
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel

def get_padding(kernel, padding='same'):
    pad = kernel - 1
    if padding == 'same':
        if kernel % 2:
            return pad // 2, pad // 2
        else:
            return pad // 2, pad // 2 + 1
    return 0, 0


def get_output_dim(dimension, kernels, strides, dilatation=1, padding='same', return_paddings=False):
    out_dim = dimension
    paddings = []
    if padding == 'same':
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel, padding))
            out_dim = (out_dim + stride - 1) // stride
    else:
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel, padding))
            out_dim = (out_dim - kernel + stride) // stride

    if return_paddings:
        return out_dim, paddings
    return out_dim

def select_quantum_device(dev_name, n_qubits, num_shots=None, random_seed=2026):
    if(dev_name == "default.qubit"):
        return qml.device(
            dev_name,
            wires=n_qubits,
            shots=num_shots,
            seed=random_seed
            )

    elif(dev_name.startswith("fake")):
        if(dev_name == "fake_lagos"):
            from qiskit_ibm_runtime.fake_provider import FakeLagosV2 as FakeBackend
        elif(dev_name == "fake_nairobi"):
            from qiskit_ibm_runtime.fake_provider import FakeNairobiV2 as FakeBackend
        elif(dev_name == "fake_perth"):
            from qiskit_ibm_runtime.fake_provider import FakePerth as FakeBackend
        elif(dev_name == "fake_fractbackend"):            
            from qiskit_ibm_runtime.fake_provider import FakeFractionalBackend as FakeBackend
        elif(dev_name == "fake_yorktown"):            
            from qiskit_ibm_runtime.fake_provider import FakeYorktownV2 as FakeBackend
        elif(dev_name == "fake_london"):            
            from qiskit_ibm_runtime.fake_provider import FakeLondonV2 as FakeBackend

        backend = FakeBackend()
        noise_model = NoiseModel.from_backend(backend)
        return qml.device(
            "qiskit.aer",
            wires=n_qubits, backend="aer_simulator",
            noise_model=noise_model, transpile=True,
            transpile_kwargs={"optimization_level": 3, "routing_method": "sabre"},
            seed=random_seed
            )

    elif dev_name.startswith("ibm"):

        token_path = "/home/user/token.txt" # TODO: indicate the token path
        with open(token_path, 'r') as f:
            lines = f.read().splitlines()
            token = lines[0].strip()
            instance = lines[1].strip()

        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
        backend = service.backend(dev_name)

        return qml.device(
            'qiskit.remote',
            wires=n_qubits,
            backend=backend
            )

def meas_qoutputsize_mapping(n_qubits):
    return {
        "pauli": n_qubits,
        "probs": 2**n_qubits
    }

class AmpeDenseModel(nn.Module):
    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42, dev_name="default.qubit", n_shots=None):

        super(AmpeDenseModel, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.dev_name = dev_name
        self.n_shots = n_shots

        self.dev = select_quantum_device(dev_name, n_qubits)

        q_output_size = meas_qoutputsize_mapping(n_qubits)["probs"]

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)

            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Measurement process
            return qml.probs(wires=range(n_qubits))

        self.qnode = qnode
        if self.dev_name.startswith("fake"): self.qnode = qml.set_shots(self.qnode, self.n_shots)

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.flatten = nn.Flatten()

        input_dim = n_packets * n_features
        self.dense1 = nn.Linear(input_dim, 2**n_qubits)
        self.sigmoid = nn.Sigmoid()

        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        self.dense2 = nn.Linear(q_output_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.q_layer(x)
        return self.dense2(x)

    def get_model_name(self):
        return f"AmpeDense_Q{self.n_qubits}_L{self.n_layers}_{self.n_packets}x{self.n_features}_C{self.num_classes}"

    def get_model_name_short(self):
        return f"AmpeDense_Q{self.n_qubits}_L{self.n_layers}"


class AmpeCNNLSTMModel(nn.Module):
    def __init__(self, n_qubits, n_layers, n_packets, n_features, num_classes, random_seed=42, dev_name="default.qubit", n_shots=None):

        super(AmpeCNNLSTMModel, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.dev_name = dev_name
        self.n_shots = n_shots

        in_channels = 1
        out_features_size = 2**n_qubits
        filters = [32, 64]
        kernel = (4, 2)
        stride = (1, 1)

        features_size0, self.paddings0 = get_output_dim(
            n_packets,
            kernels=[kernel[0], kernel[0]],
            strides=[stride[0], stride[0]],
            padding='valid',
            return_paddings=True
        )
        features_size1, self.paddings1 = get_output_dim(
            n_features,
            kernels=[kernel[1], kernel[1]],
            strides=[stride[1], stride[1]],
            padding='valid',
            return_paddings=True
        )


        self.conv1 = nn.Conv2d(in_channels, filters[0], kernel, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(filters[1])
        lstm_input_size = features_size1 * filters[1]
        hidden_size = max([100, 50])
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features_size)


        self.dev = select_quantum_device(dev_name, n_qubits)

        q_output_size = meas_qoutputsize_mapping(n_qubits)["probs"]

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # Feature map
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)

            # Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Measurement process
            return qml.probs(wires=range(n_qubits))

        self.qnode = qnode
        self.qnode = qml.set_shots(self.qnode, self.n_shots)

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}


        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.fc = nn.Linear(q_output_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.conv1(x))
        out = self.bn1(out)
        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        size_interm = out.size()
        out = out.transpose(1, 2)
        out = out.reshape(size_interm[0], size_interm[3], size_interm[1] * size_interm[2])
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = F.dropout(out, 0.2)
        out = self.fc1(out)
        out = F.relu(out)
        out = F.dropout(out, 0.4)
        out = self.fc2(out)
        out = F.sigmoid(out)
        out = self.q_layer(out)
        out = self.fc(out)
        return out

    def get_model_name(self):
        return f"AmpeCNNLSTM_Q{self.n_qubits}_L{self.n_layers}_{self.n_packets}x{self.n_features}_C{self.num_classes}"

    def get_model_name_short(self):
        return f"AmpeCNNLSTM_Q{self.n_qubits}_L{self.n_layers}"


class CNNLSTMModel(nn.Module):
    def __init__(self, n_packets, n_features, num_classes, random_seed=2026):
        super(CNNLSTMModel, self).__init__()
        self.n_packets = n_packets
        self.n_features = n_features
        self.num_classes = num_classes
        random_seed = random_seed

        in_channels = 1
        out_features_size = 100
        filters = [32, 64]
        kernel = (4, 2)
        stride = (1, 1)

        features_size0, self.paddings0 = get_output_dim(
            n_packets,
            kernels=[kernel[0], kernel[0]],
            strides=[stride[0], stride[0]],
            padding='valid',
            return_paddings=True
        )
        features_size1, self.paddings1 = get_output_dim(
            n_features,
            kernels=[kernel[1], kernel[1]],
            strides=[stride[1], stride[1]],
            padding='valid',
            return_paddings=True
        )

        self.conv1 = nn.Conv2d(in_channels, filters[0], kernel, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(filters[1])

        lstm_input_size = features_size1 * filters[1]
        hidden_size = max([100, 50])
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, out_features_size)
        self.fc = nn.Linear(out_features_size, num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.extract_features(x))
        out = F.dropout(out, 0.4)
        out = self.fc(out)
        return out

    def extract_features(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn1(out)
        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        size_interm = out.size()
        out = out.transpose(1, 2)
        out = out.reshape(size_interm[0], size_interm[3], size_interm[1] * size_interm[2])
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = F.dropout(out, 0.2)
        out = self.fc1(out)
        return out


    def get_model_name(self):
        return f"CNNLSTM_{self.n_packets}x{self.n_features}_C{self.num_classes}"

    def get_model_name_short(self):
        return f"CNNLSTM"
