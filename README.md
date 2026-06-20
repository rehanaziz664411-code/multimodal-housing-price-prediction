#  Multimodal Housing Price Prediction
### Task 3 — Multimodal ML Using CNN Image Features + Tabular Data

---

##  Overview

This project predicts **housing sale prices** by combining two data modalities:

-  **Structured / Tabular Data** — bedrooms, bathrooms, square footage, year built, condition, etc.
-  **Unstructured / Image Data** — exterior or interior photographs of each house

A **Multimodal Deep Learning model** fuses CNN-extracted image features with MLP-processed tabular features to produce a single price regression output. The model is evaluated using **MAE** and **RMSE** as required by the task specification.

---

##  Objective

> *Predict housing prices using both structured data and house images.*

| Requirement | Implementation |
|---|---|
| Use CNNs to extract image features | MobileNetV2 (pre-trained on ImageNet) |
| Combine image features with tabular data | Late fusion via `Concatenate` layer |
| Train a model using both modalities | 2-phase training (frozen then fine-tune) |
| Evaluate with MAE and RMSE | Computed on held-out test set (log and actual scale) |

---

##  Project Structure

```
multimodal-housing/
│
├── Multimodal_Housing_Price_Prediction_UPGRADED.ipynb   # Main notebook
│
├── houses.csv              # Tabular housing dataset (your input)
├── house_images/           # Folder of house images (your input)
│   ├── house_0001.jpg
│   ├── house_0002.jpg
│   └── ...
│
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── outputs/                # Generated after running the notebook
    ├── multimodal_housing_model.keras    # Trained multimodal model
    ├── tabular_only_model.keras          # Baseline: tabular only
    ├── image_only_model.keras            # Baseline: image only
    ├── scaler.pkl                        # StandardScaler
    ├── label_encoders.pkl                # Per-column LabelEncoders
    ├── imputer.pkl                       # Median imputer
    ├── model_results.csv                 # All evaluation metrics
    ├── test_predictions.csv             # Per-sample predictions
    ├── price_distribution.png
    ├── correlation_heatmap.png
    ├── sample_images.png
    ├── model_architecture.png
    ├── training_history.png
    ├── prediction_analysis.png
    ├── ablation_study.png
    └── feature_importance.png
```

---

##  Model Architecture

```
┌──────────────────────────────┐     ┌──────────────────────────┐
│        IMAGE BRANCH          │     │      TABULAR BRANCH       │
│                              │     │                          │
│  Input: (224 x 224 x 3)      │     │  Input: (N features)     │
│         ↓                    │     │         ↓                │
│  MobileNetV2                 │     │  Dense(128) + BN         │
│  (ImageNet weights, frozen   │     │  Dropout(0.3)            │
│   in Phase 1)                │     │         ↓                │
│         ↓                    │     │  Dense(64) + BN          │
│  GlobalAveragePooling2D      │     │  Dropout(0.2)            │
│         ↓                    │     │         ↓                │
│  Dense(512) + BN             │     │  Output: (N, 64)         │
│  Dropout(0.4)                │     └────────────┬─────────────┘
│         ↓                    │                  │
│  Dense(256) + BN             │                  │
│  Dropout(0.3)                │                  │
│         ↓                    │                  │
│  Output: (N, 256)            │                  │
└──────────────┬───────────────┘                  │
               └─────────────────┬────────────────┘
                                 ↓
                      CONCATENATE (N, 320)
                                 ↓
                      Dense(256) + BN + Dropout(0.3)
                                 ↓
                      Dense(128) + Dropout(0.2)
                                 ↓
                      Dense(64)
                                 ↓
                      Dense(1)  → Predicted log(Price)
                                 ↓
                           expm1( · )
                                 ↓
                        Predicted Price ($)
```

**Training strategy — 2-Phase Transfer Learning:**
- **Phase 1** — MobileNetV2 backbone fully frozen; only the fusion and tabular layers are trained (30 epochs, LR = 1e-3)
- **Phase 2** — Last 30 layers of MobileNetV2 unfrozen for fine-tuning (50 epochs, LR = 1e-4)

---

##  Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average dollar deviation from true price |
| **RMSE** | Root Mean Squared Error — penalises large errors more heavily |
| **R²** | Coefficient of Determination — proportion of variance explained (1.0 = perfect) |
| **MAPE** | Mean Absolute Percentage Error — scale-independent error percentage |

All metrics are reported on both the **log-price scale** (training target) and **actual dollars** for interpretability.

---

##  Ablation Study

A 3-way comparison quantifies each modality's individual contribution:

| Model | Input | Notes |
|---|---|---|
| **Tabular Only** | Structured features (MLP) | Strong baseline — structured features carry most pricing signal |
| **Image Only** | House photos (CNN + MLP) | Captures visual quality; weaker alone |
| **Multimodal (Ours)** | Tabular + Images | Best overall — modalities complement each other |

---

##  Setup and Installation

### Prerequisites

- Python **3.9 – 3.11** (recommended)
- pip >= 23.0
- Optional: NVIDIA GPU with CUDA 11.8 + cuDNN 8.6 for faster training

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/multimodal-housing.git
cd multimodal-housing
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# OR using conda
conda create -n housing_ml python=3.10
conda activate housing_ml
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Open `requirements.txt` and replace `tensorflow>=2.12.0` with `tensorflow[and-cuda]>=2.12.0`, then run the install command above.

### Step 4 — (Optional) Install Graphviz for model architecture diagram

```bash
# Ubuntu / Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows — download from https://graphviz.org/download/
```

The notebook handles this gracefully — if Graphviz is not installed it prints a text diagram instead.

---

##  Running the Notebook

### Step 1 — Prepare your dataset

**CSV file** (`houses.csv`) must contain at minimum:
- Numeric feature columns: `bedrooms`, `bathrooms`, `sqft`, `year_built`, etc.
- A `price` column (the regression target)
- An `image` column containing the filename of each house's photo (e.g. `house_001.jpg`)

**Image folder** (`house_images/`) must contain `.jpg`, `.jpeg`, or `.png` files whose names match the values in the `image` column of the CSV.

### Step 2 — Update configuration in Cell 2

```python
TABULAR_DATA_PATH  = 'houses.csv'        # path to your CSV file
IMAGE_DIR          = 'house_images'      # path to image folder (forward slashes)
IMAGE_FILENAME_COL = 'image'             # CSV column that holds image filenames
TARGET_COL         = 'price'             # CSV column with house sale prices
DROP_COLS          = ['id', 'address', 'date']   # columns to exclude
```

### Step 3 — Launch Jupyter and run

```bash
# Jupyter Notebook (classic)
jupyter notebook Multimodal_Housing_Price_Prediction_UPGRADED.ipynb

# OR JupyterLab
jupyter lab
```

Select **Kernel → Restart and Run All Cells**.

> **No dataset yet?** Leave all paths at their defaults. The notebook will automatically generate a synthetic dataset of 500 houses with placeholder images so you can test the complete pipeline immediately without any real data.

---

## 🔁 Notebook Cell-by-Cell Walkthrough

| Cell | Section | What it Does |
|------|---------|-------------|
| 1 | Imports | Loads all libraries; prints TF version and GPU info |
| 2 | Configuration | Set file paths, column names, and hyperparameters |
| 3 | Dataset Setup | Loads real data OR auto-generates 500-sample synthetic dataset |
| 4 | EDA — Tabular | Price histogram, log-price histogram, correlation heatmap, missing values |
| 5 | EDA — Images | Counts images, displays 6 sample house photos |
| 6 | Preprocessing | Encodes categoricals (per-column), imputes NaNs, log-transforms price |
| 7 | Train/Val/Test Split | 70/10/20 split with StandardScaler fitted on train only |
| 8 | Image Loading | Loads images with augmentation (flip, brightness ±20%, rotation ±10°) |
| 9 | Build Model | Constructs multimodal Keras model; saves architecture PNG |
| 10 | Phase 1 Training | Trains with frozen MobileNetV2 backbone; EarlyStopping + ReduceLR |
| 11 | Phase 2 Fine-tuning | Unfreezes last 30 CNN layers; fine-tunes at LR=1e-4 |
| 12 | History Plots | Combined loss and MAE curves with phase-split marker |
| 13 | Evaluation | Computes MAE, RMSE, R², MAPE on test set (log + actual scale) |
| 14 | Visualisations | Actual vs Predicted scatter; residual histogram; residuals vs predicted |
| 15 | Ablation Study | Trains Tabular-Only and Image-Only baselines; 3-way bar chart comparison |
| 16 | Feature Importance | GBM-surrogate ranks tabular features by importance |
| 17 | Save Outputs | Exports models, preprocessors, metrics CSV, predictions CSV |
| 18 | Final Summary | Architecture table, skills gained, complete output file listing |

---

##  Key Design Decisions

**Why MobileNetV2?**
MobileNetV2 is lightweight, fast to fine-tune, and pre-trained on 1.4 million ImageNet images. It extracts rich spatial features (edges, textures, structural patterns) without requiring a large GPU or very long training time.

**Why log-transform the price?**
House prices follow a right-skewed distribution. Applying `log(1 + price)` makes the distribution approximately Gaussian, which allows MSE loss to optimise fairly across both inexpensive and high-end properties.

**Why late fusion (Concatenation)?**
Late fusion allows each branch to specialise on its own modality independently before the final combination step. This is more robust than early fusion when the two input types have very different scales, formats, and information densities.

**Why 2-phase training?**
Fine-tuning a pre-trained CNN on a small domain-specific dataset from scratch risks catastrophic forgetting. Freezing the backbone first lets the regression head stabilise before gradually adjusting CNN weights in Phase 2 at a much lower learning rate.

---

##  Skills Demonstrated

- Multimodal Machine Learning — fusing heterogeneous data types end-to-end
- Convolutional Neural Networks — MobileNetV2 transfer learning and fine-tuning
- Feature Fusion — late fusion via Concatenation layer
- Regression Modelling — continuous price prediction with log-transformed target
- Model Evaluation — MAE, RMSE, R², MAPE on both log and actual scale
- Data Augmentation — horizontal flip, brightness variation, rotation for images
- Ablation Studies — quantifying each modality's independent contribution
- Feature Importance — GBM-surrogate ranking of tabular inputs
- Model Persistence — saving all models and preprocessing artifacts for deployment

---

##  Dependencies at a Glance

| Package | Min Version | Purpose |
|---|---|---|
| `tensorflow` | 2.12.0 | Deep learning framework and Keras API |
| `scikit-learn` | 1.3.0 | Preprocessing, evaluation metrics, GBM baseline |
| `numpy` | 1.24.0 | Numerical array operations |
| `pandas` | 2.0.0 | Tabular data loading and EDA |
| `matplotlib` | 3.7.0 | Plotting and visualisation |
| `seaborn` | 0.12.0 | Heatmaps and statistical plots |
| `Pillow` | 9.5.0 | Image I/O and augmentation |
| `tqdm` | 4.65.0 | Progress bars for image loading |
| `jupyter` | 1.0.0 | Notebook environment |
| `pydot` | 1.4.2 | Model architecture diagram (optional) |

Full pinned versions are in `requirements.txt`.

---

##  Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
```bash
pip install tensorflow>=2.12.0
```

**Images not loading — `No such file or directory`**
Ensure `IMAGE_DIR` in Cell 2 exactly matches your image folder name and that filenames in the `image` column of the CSV match the actual files on disk (including extension).

**Out of memory (OOM) during training**
Reduce `BATCH_SIZE` in Cell 2:
```python
BATCH_SIZE = 16   # or 8 for very limited RAM/VRAM
```

**Training is very slow (no GPU)**
Reduce epochs and use smaller batch for a quick test:
```python
EPOCHS     = 10
BATCH_SIZE = 16
```
Expected time on CPU with the 500-sample synthetic dataset: 3–5 minutes.

**`pydot` or Graphviz errors**
The notebook catches this error automatically and falls back to a text diagram. Safe to ignore unless you specifically need the PNG architecture image.

---

##  Author

**AI / ML Engineer**
Task 3 Submission — Multimodal Machine Learning

---

##  License

This project is submitted as part of a machine learning engineering assessment.
All code is original and written for educational and demonstration purposes.
