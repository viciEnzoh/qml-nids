# Quantum Machine Learning Network Intrusion Detection System

A Network Intrusion Detection System (NIDS) based on hybrid Quantum Machine Learning (QML) approaches.

## Project Structure

```
.
├── Experiments/    # Scripts to run the experiments
├── Datasets/       # Place your datasets here
├── Results/        # Output folder for results
├── LICENSE
├── README.md
└── requirements.txt
```

## Quick Start

Clone the repository:
```bash
git clone https://github.com/viciEnzoh/qml-nids.git
```

Create a virtual environment:
```
cd qml-nids
python3 -m venv q_venv
source q_venv/bin/activate
pip install -r requirements.txt
```

Then, go in the **`Experiments`** folder and launch the wished script.

## Info for reproducibility

In the following table, we report all the configurations used for the experiments conducted in this paper:

| Parameter Category | Configuration |
| :--- | :--- |
| Optimization & Training | *Optimizer*: **Adam**<br>*Learning Rate*: **0.001**<br>*Loss Function*: **Categorical Cross-Entropy**<br>*Max Epochs*: **100**<br>*Early Stopping*: **Patience = 10 epochs**<br>*Validation Strategy*: **Stratified 5-Fold Cross-Validation** |
| Quantum Circuit Specs | *Number of qubits*: **5**<br>*Quantum Encoding Schemes*: **AMPlitude (AMPe), ANGle (ANGe)**<br>*Ansatz*: **Strongly Entangling Layers**<br>*Ansatz Depth*: **3** |
| Simulation & Hardware | *Ideal Simulator*: **default.qubit**<br>*Noisy Backends*: **FakeLagos, FakeNairobi, FakeLondon, FakeFractional, FakeYorktown**<br>*Number of shots*: **8, 64, 1024** |
| Operational Evaluation | *Misuse Detection Metric*: **pAUC at 1% FPR**<br>*Attack Classification Metric*: **(macro) F1-score**<br>*Time-to-Insight Deadlines (tau)*: **0.05, 0.1, 0.5, 1, 5, 10 seconds**<br>*Calibration Metric*: **Expected Calibration Error (ECE)** |
---


## Example scripts

- **`qml_train_and_test.py`** — Train and evaluate a DL/QML model.
```bash
  python3 qml_train_and_test.py \
      --model-name ampe_fixed_probs \
      --dev-name default.qubit \
      --dataset-path ../Datasets/iot-nidd.pickle \
      --output-dir ../Results/example \
      --epochs 100 \
      --batch-size 50 \
      --num-packets 10
```

- **`qml_load_and_test.py`** — Evaluate a pre-trained stored model.
```bash
  python3 qml_load_and_test.py \
      --dataset-path ../Datasets/iot-nidd.pickle \
      --exp-path ../Results/example/ampe_fixed_probs \
      --output-dir ../Results/example_test \
      --dev-name fake_london \
      --num-shots 64
```

- **`qml_cross_eval.py`** — Cross-evaluate a trained model on a different (target) dataset than the one it was trained on (source).
```bash
  python3 qml_cross_eval.py \
      --dev-name default.qubit \
      --exp-path ../Results/example/ampe_fixed_probs \
      --output-dir ../Results/example_xeval \
      --source-dataset-path ../Datasets/iot-nidd.pickle \
      --target-dataset-path ../Datasets/edge-iiot.pickle
```

- **`qml_time_resilience.py`** — Evaluate model resilience to temporal deadlines across the values in the fixed list.
```bash
  python3 qml_time_resilience.py \
      --dataset-path ../Datasets/iot-nidd.pickle \
      --dev-name default.qubit \
      --exp-path ../Results/example/ampe_fixed_probs \
      --output-dir ../Results/example_tti \
      --thresholds-list 1000 5000 10000
```

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.