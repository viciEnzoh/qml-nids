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

Then, go in the **`Experiments`** folder and launch the wished script among the following:

- **`qml_train_and_test.py`** — Description of what this script does.
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