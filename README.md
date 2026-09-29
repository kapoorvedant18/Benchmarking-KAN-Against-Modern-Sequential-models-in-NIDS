# Temporal NetFlow Benchmark v3: Real-Sequence Intrusion Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A rigorous, leak-free temporal evaluation framework comparing **Kolmogorov-Arnold Networks (KAN)** against deep sequence architectures (**LSTM**, **GRU**, and **Transformer**) across 5 modern NetFlow v3 network intrusion detection benchmarks.

---

## 📌 Motivation & The Temporal Imperative

Standard NIDS evaluations predominantly treat network traffic as independent tabular records, randomly shuffling flow records prior to train/test splits. In production environments, network attacks exhibit strong sequential correlations, burst dynamics, and temporal drift. Randomly shuffled evaluations introduce severe **temporal data leakage**, yielding over-optimistic test metrics that collapse in live deployments.

**Temporal NetFlow Benchmark v3** addresses this with:
- **Strict Chronological Sequencing:** Flows are strictly sorted by `FLOW_START_MILLISECONDS`.
- **Zero-Overlap Sliding Windows:** Sequences of $W = 10$ consecutive flows capture burst dynamics.
- **Leak-Free Chronological Splits:** Non-overlapping $70\% / 15\% / 15\%$ partitions (Train/Validation/Test) separated by a safety boundary gap ($W = 10$) to prevent boundary bleed.
- **Identifier Masking:** IP addresses and port numbers are purged from input features to prevent identity-based memorization.

---

## 🔬 Core Empirical Findings

The complete benchmark encompasses **50 experimental runs** (5 datasets $\times$ 5 models $\times$ 2 tasks: Binary & Multiclass) with strict parameter matching (~40k–125k parameters). For detailed statistical analysis, see [`TEMPORAL_REPORT.md`](file:///c:/Users/kapoo/OneDrive/Desktop/Coding/imd%20project/temporal_benchmark_v3/TEMPORAL_REPORT.md).

### Key Takeaways:
1. **Feeding Temporal History into KAN Hurts Performance:**
   Flattening a 10-flow sequence ($10 \times 47 = 470$ dimensions) triples KAN's parameter count while consistently degrading performance compared to single-flow `KAN-static`. Standard feed-forward KAN has no temporal inductive bias; expanding input dimensions dilutes the localized B-spline activation functions.
2. **With Equal Temporal History, Sequence Models Outperform KAN:**
   When evaluated on identical 10-flow sequences, genuine temporal models (`Transformer`, `GRU`, `LSTM`) systematically outperform `KAN-window` across all datasets.
3. **Sequence Models Excel in Real-World Large-Scale Traffic:**
   On the 13.9 GB `BigFlow-NIDS` dataset, `GRU` achieves the top performance (**0.9525 Accuracy / 0.8355 Macro F1**), while `Transformer` delivers the strongest ROC-AUC.
4. **Parameter Parity Advantage:**
   Recurrent and attention mechanisms compress sequential information far more effectively than high-dimensional spline grids with equivalent or smaller parameter footprints.

---

## 📊 Summary Results

### Binary Classification Performance

| Dataset | Top Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NF-ToN-IoT-v3** | **KAN-static** | **0.9283** | **0.8953** | 0.9456 | **0.9198** | 0.9279 |
| **NF-ToN-IoT-v3** | Transformer | 0.9188 | 0.8847 | 0.9350 | 0.9092 | **0.9508** |
| **NF-BoT-IoT-v3** | **LSTM** | **0.9988** | 0.9988 | **1.0000** | **0.9994** | **0.8767** |
| **NF-UNSW-NB15-v3** | **KAN-static** | **0.9998** | **0.9974** | 0.9996 | **0.9985** | **0.99995** |
| **NF-CSE-CIC-IDS2018-v3** | **Transformer** | 0.4402 | 0.0000 | 0.0000 | 0.0000 | **0.6641** |
| **BigFlow-NIDS** | **GRU** | **0.9525** | **0.5981** | **0.8346** | **0.6968** | **0.9530** |

### Multiclass Classification Performance

| Dataset | Top Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NF-ToN-IoT-v3** | **KAN-static** | **0.9283** | **0.9275** | **0.9285** | **0.9258** | **0.9303** |
| **NF-BoT-IoT-v3** | **KAN-static** / **LSTM** | **0.9988** | **0.4997** | **0.9982** | 0.4994 | 0.5000 |
| **NF-UNSW-NB15-v3** | **KAN-static** | **0.9655** | **0.4253** | **0.9679** | **0.4026** | **0.6048** |
| **BigFlow-NIDS** | **GRU** | **0.9525** | **0.8355** | **0.9526** | **0.7931** | **0.8976** |

*All tabular metric comparisons and per-class reports are available in [`results/tables/`](file:///c:/Users/kapoo/OneDrive/Desktop/Coding/imd%20project/temporal_benchmark_v3/results/tables/).*

---

## 🧱 Evaluated Architectures

| Model Architecture | Implementation | Parameter Budget | Input Representation |
| :--- | :--- | :---: | :--- |
| **`KAN-static`** | B-Spline Kolmogorov-Arnold Network ($G=5, k=2$) | ~41k | Current flow vector $x_t \in \mathbb{R}^{47}$ |
| **`KAN-window`** | Windowed Kolmogorov-Arnold Network ($G=5, k=2$) | ~125k | Flattened sequence $(x_{t-9}, \dots, x_t) \in \mathbb{R}^{470}$ |
| **`LSTM`** | 2-Layer Bidirectional/Unidirectional LSTM + Linear Head | ~62k | Temporal window sequence $\mathbf{X}_t \in \mathbb{R}^{10 \times 47}$ |
| **`GRU`** | 2-Layer Gated Recurrent Unit + Linear Head | ~47k | Temporal window sequence $\mathbf{X}_t \in \mathbb{R}^{10 \times 47}$ |
| **`Transformer`** | Sinusoidal Positional Encoding + Multi-Head Self-Attention | ~40k | Temporal window sequence $\mathbf{X}_t \in \mathbb{R}^{10 \times 47}$ |

---

## 🗄️ Evaluated Datasets

The suite targets official **NetFlow v3** formats with consistent 47-feature schema:
1. **NF-ToN-IoT-v3** (IoT testbed telemetry, diverse network-layer attacks)
2. **NF-BoT-IoT-v3** (Massive IoT botnet traffic, volumetric DDoS/DoS)
3. **NF-UNSW-NB15-v3** (Synthetic and real modern enterprise attack vectors)
4. **NF-CSE-CIC-IDS2018-v3** (Large-scale AWS network infrastructure traffic)
5. **BigFlow-NIDS** (Real-world backbone capture, 13.9 GB)

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA support recommended (e.g., RTX 4060 or higher)

### 2. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd temporal_benchmark_v3

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scikit-learn matplotlib seaborn tqdm pykan
```

### 3. Dataset Download
Download the official NetFlow v3 datasets into the `data/` folder:
```bash
# Download UQ NetFlow v3 datasets (multi-threaded parallel download)
python download_v3_datasets.py

# BigFlow-NIDS will be downloaded automatically when running main.py,
# or can be retrieved manually via src/download_bigflow.py:
python src/download_bigflow.py
```

*Note: Raw dataset CSVs are excluded from Git via `.gitignore` to prevent multi-gigabyte bloat.*

---

## 💻 Running the Benchmark

The suite includes an incremental orchestrator with automatic checkpointing and resume safety:

### Run Everything
```bash
# Execute all 50 experiments (5 datasets x 5 models x 2 modes)
python main.py
```

### Run Specific Experiments
```bash
# Evaluate a specific dataset
python main.py --dataset BigFlow-NIDS

# Evaluate a specific model architecture
python main.py --model Transformer

# Select evaluation mode (binary or multiclass)
python main.py --mode binary

# Force re-running completed experiments
python main.py --dataset NF-ToN-IoT-v3 --model KAN-static --force
```

---

## 📁 Repository Structure

```
temporal_benchmark_v3/
├── checkpoints/              # Saved model weights (.pt files for all 50 runs)
├── data/
│   ├── .gitkeep              # Preserves directory structure
│   └── NetFlow_v3_Features.csv # Complete feature documentation and metadata
├── results/
│   ├── figures/              # Confusion matrices, ROC curves, loss curves
│   └── tables/               # Benchmark comparison CSVs and per-class metrics
├── src/
│   ├── models/               # PyTorch architectures (KAN, LSTM, GRU, Transformer)
│   ├── config.py             # Hyperparameters, window sizes, split parameters
│   ├── data_loader.py        # Chronological streaming, window slicing, zero-leak splits
│   ├── download_bigflow.py   # Automated BigFlow-NIDS streaming downloader
│   ├── evaluate.py           # Evaluation pipeline (metrics, curves, tables)
│   ├── train.py              # Training loops with early stopping and lr scheduling
│   └── utils.py              # Seeds, metrics, hardware detection helpers
├── download_v3_datasets.py   # Multi-threaded downloader for UQ v3 datasets
├── main.py                   # Master benchmark execution runner
├── TEMPORAL_REPORT.md        # Comprehensive in-depth empirical analysis report
├── .gitignore                # Excludes 28+ GB raw CSVs while tracking models & code
└── README.md                 # Project documentation
```

---

## 📜 Citation

If you use this benchmark suite or find its empirical insights helpful in your research, please cite:

```bibtex
@misc{temporal_netflow_v3,
  title={Temporal NetFlow Benchmark v3: Real-Sequence Intrusion Detection with KAN and Sequence Deep Learning},
  author={Kapoor, Tanmay and Contributors},
  year={2026},
  howpublished={\url{https://github.com/TanmayKapoor/temporal_benchmark_v3}}
}
```
