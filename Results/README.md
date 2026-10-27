# Results

This folder contains all the experimental results supporting the paper: trained models, raw model outputs, post-processed metrics, and the figures included in the manuscript.

## Directory tree

```
Results/
├── metrics/                  # Post-processed tables used to build the paper's figures
├── models_outputs/           # Trained models and raw (soft and hard) model outputs
│   ├── <dataset_name>/       # One folder per dataset: 'cic-iomt' (IoMT), 'edge-iiot' (Edge), 'iot-nidd' (NIDD)
│   │   ├── trained_models/   # Trained models and results for different training-set proportions (OC1)
│   │   ├── time_to_insight/  # Results on resilience to time constraints (OC3)
│   │   └── noisy_models/     # Results of inference on noisy backends (OC4)
│   └── cross_eval/           # Cross-evaluation results (OC2)
└── plots/                    # Figures shown in the paper, including all reliability diagrams (OC5)
```

All results are stored on a per-fold basis.

## Appendix: model naming

Correspondence between the model identifiers used in the result files and the names used in the paper.

| Identifier          | Name in the paper |
|---------------------|-------------------|
| `ampe_fixed_probs`  | AMPe (Dense)      |
| `ange_probs`        | ANGe (Dense)      |
| `hybrid_ampe_probs` | AMPe (CNN+LSTM)   |
| `hybrid`            | CNN+LSTM          |