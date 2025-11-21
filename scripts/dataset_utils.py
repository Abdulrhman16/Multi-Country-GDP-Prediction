"""
Dataset Utilities for CSV format with merged labels.

This module provides functions to:
1. Convert PyTorch .pt files to CSV format (with labels merged)
2. Load CSV files and convert back to PyTorch tensors
3. Handle both LSTM (3D sequences) and MLP (2D) data formats
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Tuple, Optional


def convert_lstm_pt_to_csv(data_path: str, label_path: str, output_csv_path: str):
    """
    Convert LSTM PyTorch tensors to CSV format with labels merged.
    
    Args:
        data_path: Path to LSTM data .pt file (shape: n_samples, time_length, n_features)
        label_path: Path to LSTM label .pt file (shape: n_samples, 1)
        output_csv_path: Path to save the merged CSV file
    """
    # Load tensors
    data = torch.load(data_path)
    labels = torch.load(label_path)
    
    # Convert to numpy
    data_np = data.numpy()
    labels_np = labels.numpy()
    
    # Flatten sequences: (n_samples, time_length, n_features) -> (n_samples, time_length * n_features)
    n_samples, time_length, n_features = data_np.shape
    data_flat = data_np.reshape(n_samples, time_length * n_features)
    
    # Create column names for flattened data
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    seq_cols = []
    for t in range(time_length):
        for f in range(n_features):
            seq_cols.append(f"t{t}_{feature_cols[f]}")
    
    # Create DataFrame
    df = pd.DataFrame(data_flat, columns=seq_cols)
    
    # Add label column
    df['target_GDP'] = labels_np.flatten()
    
    # Add metadata columns
    df['sample_id'] = range(n_samples)
    df['time_length'] = time_length
    df['n_features'] = n_features
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Converted LSTM data to CSV: {output_csv_path}")
    print(f"  Shape: {df.shape}, Time length: {time_length}, Features: {n_features}")
    
    return df


def convert_mlp_pt_to_csv(data_path: str, label_path: str, output_csv_path: str):
    """
    Convert MLP PyTorch tensors to CSV format with labels merged.
    
    Args:
        data_path: Path to MLP data .pt file (shape: n_samples, n_features)
        label_path: Path to MLP label .pt file (shape: n_samples, n_label_features)
        output_csv_path: Path to save the merged CSV file
    """
    # Load tensors
    data = torch.load(data_path)
    labels = torch.load(label_path)
    
    # Convert to numpy
    data_np = data.numpy()
    labels_np = labels.numpy()
    
    # Create feature column names
    n_samples, n_features = data_np.shape
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    
    # Create label column names
    n_label_features = labels_np.shape[1] if len(labels_np.shape) > 1 else 1
    if n_label_features == 1:
        label_cols = ['target_GDP']
    else:
        label_cols = [f"target_{i}" for i in range(n_label_features)]
        # If labels have GDP + meta, name them appropriately
        if n_label_features == 4:  # GDP, country_code, year, quarter
            label_cols = ['target_GDP', 'target_country_code', 'target_year', 'target_quarter']
    
    # Create DataFrame
    df_data = pd.DataFrame(data_np, columns=feature_cols)
    df_labels = pd.DataFrame(labels_np, columns=label_cols)
    
    # Merge
    df = pd.concat([df_data, df_labels], axis=1)
    
    # Add metadata
    df['sample_id'] = range(n_samples)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Converted MLP data to CSV: {output_csv_path}")
    print(f"  Shape: {df.shape}, Features: {n_features}, Label features: {n_label_features}")
    
    return df


def load_lstm_from_csv(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load LSTM dataset from CSV and convert back to PyTorch tensors.
    
    Args:
        csv_path: Path to CSV file created by convert_lstm_pt_to_csv
        
    Returns:
        data: Tensor of shape (n_samples, time_length, n_features)
        labels: Tensor of shape (n_samples, 1)
    """
    df = pd.read_csv(csv_path)
    
    # Extract metadata
    time_length = int(df['time_length'].iloc[0])
    n_features = int(df['n_features'].iloc[0])
    n_samples = len(df)
    
    # Extract feature columns (matching pattern 't{time}_feature_{index}' from convert_lstm_pt_to_csv)
    # Exclude metadata columns like 'target_GDP' and 'time_length' that start with 't' but aren't features
    import re
    # Pattern: t followed by digits, then _feature_, then digits
    pattern = re.compile(r'^t\d+_feature_\d+$')
    feature_cols = [col for col in df.columns if pattern.match(col)]
    
    # Sort feature columns to ensure correct order (t0_feature_0, t0_feature_1, ..., t1_feature_0, ...)
    # Sort by time step first, then by feature index
    def sort_key(col):
        # Extract time step and feature index from pattern 't{time}_feature_{index}'
        match = pattern.match(col)
        if match:
            parts = col.split('_')
            if len(parts) == 3:  # ['t0', 'feature', '0']
                try:
                    time_idx = int(parts[0][1:])  # Extract number after 't'
                    feat_idx = int(parts[2])      # Extract feature index
                    return (time_idx, feat_idx)
                except ValueError:
                    pass
        return (999, 999)  # Put invalid columns at end
    
    feature_cols = sorted(feature_cols, key=sort_key)
    
    # Validate expected number of feature columns
    expected_cols = time_length * n_features
    if len(feature_cols) != expected_cols:
        raise ValueError(
            f"Mismatch in feature columns: expected {expected_cols} columns "
            f"(time_length={time_length} * n_features={n_features}), "
            f"but found {len(feature_cols)} columns starting with 't'. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Extract label column
    label_col = 'target_GDP'
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Get data and reshape
    data_flat = df[feature_cols].values
    expected_size = n_samples * time_length * n_features
    actual_size = data_flat.size
    
    if actual_size != expected_size:
        raise ValueError(
            f"Cannot reshape array: expected {expected_size} elements "
            f"({n_samples} samples * {time_length} time steps * {n_features} features), "
            f"but got {actual_size} elements. "
            f"Found {len(feature_cols)} feature columns."
        )
    
    data = data_flat.reshape(n_samples, time_length, n_features)
    
    # Get labels
    labels = df[label_col].values.reshape(n_samples, 1)
    
    # Convert to tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    return data_tensor, labels_tensor


def load_mlp_from_csv(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load MLP dataset from CSV and convert back to PyTorch tensors.
    
    Args:
        csv_path: Path to CSV file created by convert_mlp_pt_to_csv
        
    Returns:
        data: Tensor of shape (n_samples, n_features)
        labels: Tensor of shape (n_samples, n_label_features)
    """
    df = pd.read_csv(csv_path)
    
    # Extract feature columns (all columns starting with 'feature_')
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Extract label columns (all columns starting with 'target_')
    label_cols = [col for col in df.columns if col.startswith('target_')]
    
    # Get data and labels
    data = df[feature_cols].values
    labels = df[label_cols].values
    
    # Convert to tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    return data_tensor, labels_tensor


def convert_all_datasets_to_csv(dataset_dir: str = "./dataset", output_dir: Optional[str] = None):
    """
    Convert all .pt dataset files to CSV format.
    
    Args:
        dataset_dir: Directory containing .pt files
        output_dir: Directory to save CSV files (defaults to same as dataset_dir)
    """
    if output_dir is None:
        output_dir = dataset_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all data files (exclude flattened and already converted CSV files)
    data_files = [
        f for f in os.listdir(dataset_dir) 
        if (f.startswith("LSTM_data_") or f.startswith("MLP_data_")) 
        and f.endswith(".pt")
        and "flattened" not in f.lower()
    ]
    
    converted = []
    for data_file in data_files:
        data_path = os.path.join(dataset_dir, data_file)
        
        # Determine corresponding label file
        if "LSTM_data_" in data_file:
            label_file = data_file.replace("LSTM_data_", "LSTM_label_")
            label_path = os.path.join(dataset_dir, label_file)
            output_file = data_file.replace(".pt", ".csv")
            output_path = os.path.join(output_dir, output_file)
            
            # Skip if CSV already exists
            if os.path.exists(output_path):
                print(f"Skipping {data_file} - CSV already exists: {output_file}")
                converted.append(output_file)
                continue
            
            if os.path.exists(label_path):
                convert_lstm_pt_to_csv(data_path, label_path, output_path)
                converted.append(output_file)
            else:
                print(f"Warning: Label file not found for {data_file}: {label_path}")
        
        elif "MLP_data_" in data_file:
            label_file = data_file.replace("MLP_data_", "MLP_label_")
            label_path = os.path.join(dataset_dir, label_file)
            output_file = data_file.replace(".pt", ".csv")
            output_path = os.path.join(output_dir, output_file)
            
            # Skip if CSV already exists
            if os.path.exists(output_path):
                print(f"Skipping {data_file} - CSV already exists: {output_file}")
                converted.append(output_file)
                continue
            
            if os.path.exists(label_path):
                convert_mlp_pt_to_csv(data_path, label_path, output_path)
                converted.append(output_file)
            else:
                print(f"Warning: Label file not found for {data_file}: {label_path}")
    
    print(f"\n✅ Converted {len(converted)} datasets to CSV format")
    return converted


def load_dataset_from_csv_or_pt(data_path: str, label_path: Optional[str] = None, dataset_type: str = "auto"):
    """
    Load dataset from CSV (preferred) or PyTorch .pt files.
    
    Args:
        data_path: Path to data file (.csv or .pt)
        label_path: Path to label file (.pt) - only used if data_path is .pt
        dataset_type: "auto", "lstm", or "mlp" - auto-detects from filename if "auto"
        
    Returns:
        data: Tensor
        labels: Tensor
    """
    # Check if CSV exists
    csv_path = data_path.replace(".pt", ".csv")
    if os.path.exists(csv_path):
        # Auto-detect type from filename
        if dataset_type == "auto":
            if "LSTM" in csv_path or "lstm" in csv_path.lower():
                dataset_type = "lstm"
            elif "MLP" in csv_path or "mlp" in csv_path.lower():
                dataset_type = "mlp"
            else:
                raise ValueError(f"Cannot auto-detect dataset type from path: {csv_path}")
        
        if dataset_type == "lstm":
            return load_lstm_from_csv(csv_path)
        elif dataset_type == "mlp":
            return load_mlp_from_csv(csv_path)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Fall back to .pt files
    if label_path is None:
        if "LSTM_data" in data_path:
            label_path = data_path.replace("LSTM_data", "LSTM_label")
        elif "MLP_data" in data_path:
            label_path = data_path.replace("MLP_data", "MLP_label")
        else:
            raise ValueError(f"Cannot determine label path from: {data_path}")
    
    if os.path.exists(data_path) and os.path.exists(label_path):
        data = torch.load(data_path)
        labels = torch.load(label_path)
        return data, labels
    else:
        raise FileNotFoundError(
            f"Could not find dataset files. Tried CSV: {csv_path}, "
            f"PT data: {data_path}, PT labels: {label_path}"
        )

