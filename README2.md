# Multi-Country GDP Prediction - Complete Workflow Guide

Welcome to the Multi-Country GDP Prediction project! This guide will walk you through the entire process step-by-step, from data preparation to model training and evaluation.

## 📋 Project Overview

This project predicts GDP growth across multiple countries using:

- **LSTM models** for sequential/temporal data (quarterly data)
- **MLP models** for tabular data (yearly data)
- **Transformer & PatchTST models** for advanced time series prediction
- **Light intensity data** as an auxiliary feature for GDP estimation

---

## 🚀 Quick Start - Step-by-Step Guide

### **STEP 1: Environment Setup**

#### 1.1 Install Python Dependencies

```bash
# Navigate to the project root
cd Multi-Country-GDP-Prediction

# Install required packages
pip install -r requirements.txt
```

#### 1.2 Verify Installation

Check that all packages are installed correctly:

```bash
python -c "import torch, pandas, numpy; print('✅ All dependencies installed')"
```

---

### **STEP 2: Data Preparation (Run Notebooks in Order)**

The data preparation process involves loading, processing, and converting raw data into PyTorch tensors for model training.

#### **2.1 Prepare MLP Yearly Dataset with Light Data**

**File:** `process_origin_data/save_mlp_dataset_year_light_sms_sample.ipynb`

**What this does:**

- Loads economic data (2013-2019) from `yearly/integrated_yearly_data.xlsx`
- Loads light intensity satellite data from `light_data/light_data.csv`
- Merges datasets by country and year
- Normalizes features using StandardScaler
- Creates country code mappings
- Generates PyTorch tensors for MLP yearly training

**Output files:**

- `dataset/MLP_data_light_sms_y_13-19.pt` (features: 135 samples × 18 features)
- `dataset/MLP_label_light_sms_y_13-19.pt` (labels: 135 samples × 3 info columns)

**How to run:**

```
1. Open the notebook in Jupyter
2. Click "Run All" or press Ctrl+Shift+Enter
3. Wait for completion (should take ~30 seconds)
4. Verify output files exist in dataset/ folder
```

#### **2.2 Prepare MLP Quarterly Dataset with Light Data**

**File:** `process_origin_data/save_mlp_dataset_sample.ipynb`

**What this does:**

- Processes quarterly economic data
- Aggregates light data by quarter
- Creates MLP-ready quarterly tensors

**Output files:**

- `dataset/MLP_data_light_sms_q_13-19.pt`
- `dataset/MLP_label_light_sms_q_13-19.pt`

#### **2.3 Prepare LSTM Dataset**

**Files:**

- `process_origin_data/save_lstm_dataset_sample.ipynb`

**What this does:**

- Creates sequential time-series data from quarterly economic indicators
- Reshapes data into LSTM input format (samples × timesteps × features)
- Prepares labels for LSTM training

**Output files:**

- `dataset/LSTM_data_light_sms_q_t*.pt` (multiple files for different timesteps)
- `dataset/LSTM_label_light_sms_q_t*.pt`
- `dataset/LSTM_flattened_data_light_sms_q_t*.pt`

---

### **STEP 3: Run everything from `master_notebook.ipynb` (preferred)**

Instead of invoking individual scripts, this project provides a single orchestrating notebook, `master_notebook.ipynb`, that sequences data preparation, training and evaluation for all models (LSTM, MLP, Transformer, and optional advanced models). Use this notebook to run the full pipeline end-to-end with centralized configuration.

Why use the notebook:

- Centralized configuration for all models (hyperparameters, single-output mode, verbose flags)
- Runs data-prep notebooks, training routines and evaluation in the correct order
- Consolidates results into single CSVs / single model checkpoints for easier tracking

How to run `master_notebook.ipynb`:

1. Open the notebook in Jupyter or VS Code (Jupyter extension).
2. Inspect the first configuration cell (Cell 1) and set options as needed (e.g., `single_output = True`, `VERBOSE = False`, date ranges).
3. Run the notebook top-to-bottom (recommended) with "Run All" or run each section manually in this order:
   - Configuration cell (set `single_output` and model toggles)
   - Data preparation sections (these call the data notebooks or run equivalent steps)
   - Model training sections (LSTM → MLP → Transformer as enabled)
   - Evaluation & aggregate results export

Notebook tips:

- If you only want to run a subset, toggle the model-run flags in the configuration cell and then run the relevant cells.
- Use the notebook toolbar "Restart Kernel and Run All" to get a fresh run from scratch.
- Review printed outputs or the aggregated CSVs (`best_params_res.csv`, `linear_*.csv`) in the project root after completion.

Expected outputs when using the notebook:

- `dataset/` tensors (same files produced by data preparation notebooks)
- Aggregated `best_params_res.csv` with hyperparameter search results
- Aggregated result CSVs: `linear_lstm_q_res.csv`, `linear_mlp_q_res.csv`, `linear_mlp_y_res.csv`

Configuration note:

Make sure `master_notebook.ipynb` contains `"single_output": True` in the model configs if you want consolidated outputs rather than per-file outputs.

---

#### When you might still use scripts

- For automated CI or headless runs on servers where notebooks are inconvenient, the individual scripts in `scripts/` can still be used. The notebook wraps the same functionality but provides easier configuration and monitoring.

### **STEP 4: Advanced Models (Optional)**

These models require specific dependencies and configurations.

#### **4.1 PatchTST Model**

**File:** `timesfm&PatchTST&Time-LLM/patchTST/run_patchtst_q.py`

```bash
cd timesfm&PatchTST&Time-LLM/patchTST
python run_patchtst_q.py
```

#### **4.2 Time-LLM Model**

**File:** `timesfm&PatchTST&Time-LLM/Time-LLM/run_timellm_q.py`

```bash
cd ../Time-LLM
python run_timellm_q.py
```

---

## 🗑️ Cleaning Up Old Generated Data

Before running new experiments, clean up old data to avoid conflicts.

### **Manual Cleanup**

```bash
# Remove old dataset tensors
rm dataset/MLP_*.pt
rm dataset/LSTM_*.pt
rm dataset/LSTM_flattened_*.pt

# Remove old results
rm linear_mlp_*.csv
rm linear_lstm_*.csv
```

### **Automatic Cleanup** (Recommended)

Create `scripts/cleanup.py`:

```python
import os
import glob

def cleanup_old_data():
    """Remove old generated dataset and result files"""

    patterns = [
        'dataset/MLP_*.pt',
        'dataset/LSTM_*.pt',
        'dataset/LSTM_flattened_*.pt',
        'linear_*.csv',
    ]

    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"✅ Deleted: {file}")
            except Exception as e:
                print(f"❌ Error deleting {file}: {e}")

if __name__ == "__main__":
    cleanup_old_data()
    print("\n✅ Cleanup complete!")
```

**Run cleanup:**

```bash
python scripts/cleanup.py
```

---

## 📊 Complete Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT WORKFLOW                         │
└─────────────────────────────────────────────────────────────┘

STEP 1: SETUP
├─ Install requirements.txt
└─ Verify Python environment

STEP 2: DATA PREPARATION (Run Notebooks)
├─ save_mlp_dataset_year_light_sms_sample.ipynb
│  └─ Output: MLP_data_light_sms_y_13-19.pt
│
├─ save_mlp_dataset_sample.ipynb (quarterly)
│  └─ Output: MLP_data_light_sms_q_13-19.pt
│
└─ save_lstm_dataset_sample.ipynb
   └─ Output: LSTM_data_light_sms_q_t*.pt

STEP 3: CLEANUP OLD DATA (Optional)
├─ Run: python scripts/cleanup.py
└─ Or manually delete old dataset/ and .csv files

STEP 4: MODEL TRAINING (Run Scripts)
├─ python scripts/run_lstm_q.py
│  └─ Output: single consolidated LSTM model + CSV
│
├─ python scripts/run_mlp_q.py
│  └─ Output: single consolidated MLP quarterly model
│
└─ python scripts/run_mlp_y.py
   └─ Output: MLP yearly model

STEP 5: EVALUATE RESULTS
├─ Check best_params_res.csv for hyperparameters
├─ Check linear_*.csv for evaluation metrics
└─ Load model checkpoints for inference
```

---

## 📁 Directory Structure

```
Multi-Country-GDP-Prediction/
├── README.md                          # Original project README
├── README2.md                         # ← You are here (this guide)
├── requirements.txt                   # Python dependencies
├── master_notebook.ipynb              # Configuration file (edit here!)
│
├── dataset/                           # Generated tensor files (Git ignored)
│   ├── MLP_data_light_sms_y_13-19.pt
│   ├── MLP_label_light_sms_y_13-19.pt
│   ├── LSTM_data_light_sms_q_t*.pt
│   └── ...
│
├── process_origin_data/               # Data preparation notebooks
│   ├── save_mlp_dataset_year_light_sms_sample.ipynb
│   ├── save_mlp_dataset_sample.ipynb
│   ├── save_lstm_dataset_sample.ipynb
│   ├── light_data/                    # Light intensity satellite data
│   │   ├── light_data.csv
│   │   └── IMF_ISO.xlsx
│   └── yearly/                        # Economic data
│       ├── integrated_yearly_data.xlsx
│       └── data/
│
├── scripts/                           # Training scripts
│   ├── run_lstm_q.py                  # LSTM quarterly training
│   ├── run_mlp_q.py                   # MLP quarterly training
│   ├── run_mlp_y.py                   # MLP yearly training
│   ├── run_transformer_y.py           # Transformer yearly training
│   ├── cleanup.py                     # Data cleanup utility
│   └── metrics.py                     # Evaluation metrics
│
├── timesfm&PatchTST&Time-LLM/        # Advanced time series models
│   ├── patchTST/
│   │   └── run_patchtst_q.py
│   ├── Time-LLM/
│   │   └── run_timellm_q.py
│   └── timesfm/
│
└── results/                           # (Optional) Final results folder
    ├── best_params_res.csv
    ├── linear_lstm_q_res.csv
    ├── linear_mlp_q_res.csv
    └── linear_mlp_y_res.csv
```

---

## ⚙️ Configuration

All model configurations are centralized in `master_notebook.ipynb`.

### Key Configurations:

**Cell 1 - Model Configs:**

```python
LSTM_Q_CONFIG = {
    "single_output": True,           # ← Consolidate outputs
    "hyperparameter_search": True,
    "n_splits": 5,                   # 5-fold cross-validation
    "epochs": 100,
    ...
}

MLP_Q_CONFIG = {
    "single_output": True,
    ...
}
```

### What `"single_output": True` means:

- ✅ All results append to **one CSV file** (not per-country files)
- ✅ All models save to **one PTH checkpoint** per directory
- ✅ Cleaner output structure
- ✅ Easier to track results

---

## 📈 Understanding the Results

### Output Files Generated:

| File                    | Content                             | When Created |
| ----------------------- | ----------------------------------- | ------------ |
| `best_params_res.csv`   | Best hyperparameters for each model | After step 4 |
| `linear_lstm_q_res.csv` | LSTM quarterly evaluation metrics   | After step 3 |
| `linear_mlp_q_res.csv`  | MLP quarterly evaluation metrics    | After step 3 |
| `linear_mlp_y_res.csv`  | MLP yearly evaluation metrics       | After step 3 |

### Key Metrics (in result CSVs):

- **R²**: How well the model explains variance (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError" when running scripts

**Solution:**

```bash
pip install -r requirements.txt
python -m pip install --upgrade pip
```

### Issue: "CUDA out of memory" errors

**Solution:**
Edit the script and reduce batch size or number of hyperparameters:

```python
LSTM_Q_CONFIG = {
    "batch_size": 16,  # Reduce from 32
    "n_random_searches": 30,  # Reduce from 54
}
```

### Issue: Old data conflicts with new runs

**Solution:**

```bash
# Clean up old data
python scripts/cleanup.py

# Or manually:
rm dataset/*.pt
rm linear_*.csv
```

### Issue: Notebook kernel crashes during execution

**Solution:**

1. Clear notebook outputs: `Ctrl+Shift+P` → "Clear All Outputs"
2. Restart kernel: `Ctrl+Shift+P` → "Restart Kernel"
3. Run cells one-by-one instead of "Run All"

---

## 📚 Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/

---

## ✅ Verification Checklist

After completing each step, verify:

- [ ] **Step 1**: `python -c "import torch; print(torch.__version__)"`
- [ ] **Step 2**: Check `dataset/` folder for `*.pt` files
- [ ] **Step 3**: Check for `best_params_res.csv` file
- [ ] **Step 4**: Check for `linear_*.csv` result files
- [ ] **Step 5**: Review metrics - R² > 0.5 is good

---

## 🎯 Next Steps

After completing the workflow:

1. **Analyze Results**: Compare metrics across different models
2. **Hyperparameter Tuning**: Modify configs in `master_notebook.ipynb`
3. **Custom Predictions**: Use trained models for inference on new data
4. **Model Deployment**: Package best model for production use

---

## 💡 Tips for Success

✅ **DO:**

- Run data preparation notebooks first
- Clean old data before new experiments
- Check result CSVs for model performance
- Monitor console output for errors

❌ **DON'T:**

- Skip the data preparation step
- Mix different data versions in one run
- Run all training scripts in parallel (GPU memory issues)
- Modify notebook cells without understanding what they do

---

## 📝 Notes

- All notebooks are set to **quiet mode** (`VERBOSE=False`) to reduce output clutter
- Models use **single output mode** by default for cleaner result files
- Light intensity data spans 2012-2024, but analysis focuses on 2013-2019
- LSTM models use 5-fold cross-validation for robust evaluation

---

## 🤝 Contributing

If you make improvements to this workflow:

1. Update this README2.md
2. Document any new configuration options
3. Test the complete workflow before committing

---

**Last Updated:** November 2025
**Project Status:** ✅ Active
