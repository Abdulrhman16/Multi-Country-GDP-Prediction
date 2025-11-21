import numpy as np
import pandas as pd
from utils.metrics import metric
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import itertools
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Import dataset utilities for CSV loading
try:
    from dataset_utils import load_lstm_from_csv, load_mlp_from_csv
except ImportError:
    # If import fails, define basic CSV loading functions here
    def load_lstm_from_csv(csv_path):
        """Load LSTM dataset from CSV."""
        df = pd.read_csv(csv_path)
        time_length = int(df['time_length'].iloc[0])
        n_features = int(df['n_features'].iloc[0])
        n_samples = len(df)
        feature_cols = [col for col in df.columns if col.startswith('t')]
        data_flat = df[feature_cols].values
        data = data_flat.reshape(n_samples, time_length, n_features)
        labels = df['target_GDP'].values.reshape(n_samples, 1)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
    
    def load_mlp_from_csv(csv_path):
        """Load MLP dataset from CSV."""
        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        label_cols = [col for col in df.columns if col.startswith('target_')]
        data = df[feature_cols].values
        labels = df[label_cols].values
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# Control noisy prints from training/hp-search. Set True to enable detailed output.
VERBOSE = False

# NOTE: Do not import `master_config` at module import time. The notebook defines
# configs in its top cell and will import functions from this module. Import
# `master_config` only when running this file as a script (CLI).


def load_lstm_dataset(data_path: str, label_path: Optional[str] = None, prefer_csv: bool = True):
    """
    Load LSTM dataset from CSV (preferred) or PyTorch .pt files.
    
    Args:
        data_path: Path to data file (.csv or .pt) or CSV file with merged data+labels
        label_path: Path to label file (.pt) - only used if data_path is .pt
        prefer_csv: If True, prefer CSV format; if False, prefer .pt format
        
    Returns:
        data: Tensor of shape (n_samples, time_length, n_features)
        labels: Tensor of shape (n_samples, 1)
    """
    # Check if CSV exists
    csv_path = data_path.replace(".pt", ".csv")
    if prefer_csv and os.path.exists(csv_path):
        print(f"Loading from CSV: {csv_path}")
        return load_lstm_from_csv(csv_path)
    
    # Fall back to .pt files
    if label_path is None:
        label_path = data_path.replace("LSTM_data_", "LSTM_label_")
    
    if os.path.exists(data_path) and os.path.exists(label_path):
        print(f"Loading from PyTorch files: {data_path}, {label_path}")
        data = torch.load(data_path)
        labels = torch.load(label_path)
        return data, labels
    else:
        raise FileNotFoundError(
            f"Could not find dataset files. Tried CSV: {csv_path}, "
            f"PT data: {data_path}, PT labels: {label_path}"
        )


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 计算最小值和最大值, 并进行归一化
def norm_lstm_tensor(data, labels, freq="quarter"):
    # Validate inputs
    if data.numel() == 0:
        raise ValueError("Data tensor is empty")
    if labels.numel() == 0:
        raise ValueError("Labels tensor is empty")
    
    if freq == "quarter":
        drop_index = -3
    else:
        drop_index = -2

    # 将最后一维的数据分开
    data_to_norm = data[:, :, :drop_index]  # 除了最后一个维度
    
    # Handle labels: check if labels have same structure as data or only GDP
    # If labels have fewer features, assume they only contain GDP (target feature)
    label_features_original = labels.shape[-1]
    labels_only_gdp = label_features_original + drop_index <= 0
    
    # Check if labels can be sliced with drop_index without becoming empty
    if labels_only_gdp:
        # Labels have fewer features than drop_index suggests (e.g., only GDP)
        # In this case, labels likely only contain GDP, so we use it directly
        labels_to_norm = labels  # Use all labels (just GDP)
        
        # Extract GDP feature from data for normalization
        # GDP is typically the last feature before the meta columns (year, quarter, country_code)
        # For quarterly: data has features [..., GDP, country_code, year, quarter]
        # So GDP is at index -4 (or we need to find it)
        # Actually, if drop_index=-3, data_to_norm excludes last 3, so GDP should be in data_to_norm
        # But we need to find which feature is GDP in data_to_norm
        
        # Since labels only have GDP, we'll normalize data features separately
        # and use GDP from data for min/max calculation
        # Extract GDP from data: it's the feature before the last 3 (country_code, year, quarter)
        # So GDP should be at index -4 in original data, which is -1 in data_to_norm
        data_gdp_feature = data_to_norm[:, :, -1]  # Last feature in data_to_norm (should be GDP)
        
        # Calculate min/max for data features (excluding GDP for now)
        data_features_to_norm = data_to_norm[:, :, :-1]  # All features except GDP
        max_vals_data, _ = torch.max(data_features_to_norm, dim=0)
        max_vals_data, _ = torch.max(max_vals_data, dim=0)
        min_vals_data, _ = torch.min(data_features_to_norm, dim=0)
        min_vals_data, _ = torch.min(min_vals_data, dim=0)
        
        # Calculate min/max for GDP from both data and labels
        data_gdp_min = data_gdp_feature.min().item()
        data_gdp_max = data_gdp_feature.max().item()
        label_gdp_min = labels.min().item()
        label_gdp_max = labels.max().item()
        
        gdp_min = min(data_gdp_min, label_gdp_min)
        gdp_max = max(data_gdp_max, label_gdp_max)
        
        # Combine: data features + GDP
        min_value_all = torch.cat([min_vals_data, torch.tensor([gdp_min], device=data.device)])
        max_value_all = torch.cat([max_vals_data, torch.tensor([gdp_max], device=data.device)])
        
    else:
        # Labels have same structure as data - normal case
        labels_to_norm = labels[:, :drop_index]  # 除了最后一个维度
        
        # Validate that data and labels have compatible feature dimensions
        data_features = data_to_norm.shape[-1]
        label_features = labels_to_norm.shape[-1]
        if data_features != label_features:
            raise ValueError(
                f"Feature dimension mismatch: data has {data_features} features "
                f"but labels has {label_features} features after slicing. "
                f"Data shape: {data.shape}, Labels shape: {labels.shape}"
            )

        # 计算data的最小值和最大值
        max_vals_data, _ = torch.max(data_to_norm, dim=0)  # 对第一个维度求最大值
        max_vals_data, _ = torch.max(max_vals_data, dim=0)  # 再对第二个维度求最大值

        min_vals_data, _ = torch.min(data_to_norm, dim=0)  # 对第一个维度求最小值
        min_vals_data, _ = torch.min(min_vals_data, dim=0)  # 再对第二个维度求最小值

        # 计算labels的最小值和最大值
        # labels_to_norm shape: (num_samples, num_features)
        # Reduce to 1D by taking min/max across all samples
        min_vals_label = labels_to_norm.min(
            dim=0, keepdim=False
        ).values  # shape: (num_features,)
        max_vals_label = labels_to_norm.max(
            dim=0, keepdim=False
        ).values  # shape: (num_features,)

        # min_vals_data and max_vals_data are already 1D (num_features,), safe to compare
        min_value_all = torch.minimum(min_vals_data, min_vals_label)
        max_value_all = torch.maximum(max_vals_data, max_vals_label)

    min_value = min_value_all[-1]
    max_value = max_value_all[-1]

    # 计算 Min-Max 归一化
    # 对 data (去除最后一个维度的部分) 进行 Min-Max 归一化
    normalized_data_to_norm = (data_to_norm - min_value_all) / (
        max_value_all - min_value_all + 1e-8  # Add small epsilon to avoid division by zero
    )

    # Handle label normalization based on label structure
    if labels_only_gdp:
        # Labels only have GDP - normalize using GDP min/max
        gdp_min_val = min_value_all[-1]
        gdp_max_val = max_value_all[-1]
        normalized_labels_to_norm = (labels_to_norm - gdp_min_val) / (
            gdp_max_val - gdp_min_val + 1e-8
        )
        # Labels don't have meta columns, so just return normalized GDP
        normalized_label = normalized_labels_to_norm
    else:
        # Labels have same structure as data - normal case
        # 对 label (去除最后一个维度的部分) 进行 Min-Max 归一化
        normalized_labels_to_norm = (labels_to_norm - min_value_all) / (
            max_value_all - min_value_all + 1e-8
        )
        # cat the last dim (meta columns)
        normalized_label = torch.cat(
            [normalized_labels_to_norm, labels[:, drop_index:]], dim=1
        )

    # cat the last dim for data (meta columns)
    normalized_data = torch.cat(
        [normalized_data_to_norm, data[:, :, drop_index:]], dim=2
    )

    return normalized_data, normalized_label, min_value, max_value


def split_lstm_dataset_by_year(data, labels, year, freq="quarter"):
    if freq == "quarter":
        dim_index = -2  # Year is at -2 in data (before quarter at -1)
        data_year_index = -2  # Year position in data tensor
    else:
        dim_index = -1
        data_year_index = -1

    # Check if labels have enough features to contain year
    # If labels only have GDP (1 feature), get year from data instead
    labels_have_year = labels.shape[-1] > abs(dim_index)
    
    train_index_list = []
    test_index_list = []
    
    if labels_have_year:
        # Normal case: labels have year column
        for i in range(len(labels)):
            if labels[i, dim_index] >= year:
                test_index_list.append(i)
            else:
                train_index_list.append(i)
    else:
        # Labels only have GDP - get year from data
        # For quarterly: data shape is (n_samples, seq_len, n_features)
        # Year is at the last time step, at position data_year_index
        # We need to get year from the last time step of each sequence
        for i in range(len(labels)):
            # Get year from the last time step of sequence i
            year_value = data[i, -1, data_year_index].item()
            if year_value >= year:
                test_index_list.append(i)
            else:
                train_index_list.append(i)

    # Split data and labels
    train_data = data[train_index_list, :, : dim_index - 1]
    test_data = data[test_index_list, :, : dim_index - 1]
    
    # Handle labels: if labels only have GDP, don't slice
    if labels_have_year:
        train_targets = labels[train_index_list, : dim_index - 1]
        test_targets = labels[test_index_list, : dim_index - 1]
    else:
        # Labels only have GDP - use all labels (just GDP)
        train_targets = labels[train_index_list]
        test_targets = labels[test_index_list]
    
    return train_data, test_data, train_targets, test_targets


def reverse_norm(row):
    # Deprecated: explicit scaling required. Use `reverse_norm_with_scale` instead.
    raise RuntimeError(
        "reverse_norm is deprecated. Call reverse_norm_with_scale(row, min_value, max_value) instead."
    )


def reverse_norm_with_scale(row, min_value, max_value):
    # Handle None values (fall back to identity scaling)
    if min_value is None or max_value is None:
        # If original scale is unknown, assume normalized data in [0,1]
        min_val = 0.0
        max_val = 1.0
    else:
        # Accept tensors or numeric scalars for min/max
        try:
            min_val = float(min_value)
            max_val = float(max_value)
        except Exception:
            min_val = min_value.item()
            max_val = max_value.item()

    gap = max_val - min_val
    if gap == 0:
        gap = 1.0

    if len(row.shape) == 2:
        return row[:, -1] * gap + min_val
    else:
        return row * gap + min_val


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # initial hidden and cell
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_dim)

        # Decode the last time step of the hidden state
        out = self.fc(out[:, -1, :])
        return out


# no train loss
import time


def no_train_loss(model, train_loader, criterion, weight, device, min_value, max_value):
    model.eval()
    total_loss = 0
    total_gdp_loss = 0

    with torch.no_grad():
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # Forward
            outputs = model(batch_data)

            loss = criterion(outputs, batch_labels, weight)

            gdp_loss = criterion(
                reverse_norm_with_scale(outputs, min_value, max_value),
                reverse_norm_with_scale(batch_labels, min_value, max_value),
                weight,
            )

            total_loss += loss.item() * batch_data.size(0)
            total_gdp_loss += gdp_loss.item() * batch_data.size(0)

    total_loss = total_loss / len(train_loader.dataset)
    total_gdp_loss = total_gdp_loss / len(train_loader.dataset)
    return total_loss, total_gdp_loss


def loss_weight(outputs, targets, weight):
    # print(outputs.shape)
    # Determine device from tensor (avoid relying on an external 'device' name)
    if isinstance(outputs, torch.Tensor):
        _device = outputs.device
    else:
        _device = torch.device("cpu")

    if len(outputs.shape) != 1:
        criterion_ = nn.MSELoss(reduction="none")

        # create weights directly on correct device
        weights = torch.tensor([1.0] * outputs.shape[-1], device=_device)
        weights[-1] = weight

        # Compute the element-wise MSE loss
        loss_temp = criterion_(outputs, targets)

        # Apply the weights
        weighted_loss = loss_temp * weights

        # Reduce the weighted loss (e.g., take the mean or sum)
        loss = weighted_loss.mean()

    elif len(outputs.shape) == 1:  # GDP use MSE
        criterion_ = nn.MSELoss()
        loss = criterion_(outputs, targets)
    else:
        raise ValueError("output shape Value Wrong!")

    return loss


# train and eval
def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    weight,
    device,
    min_value,
    max_value,
):
    best_model_wts = None
    best_val_gdp_loss = float("inf")
    best_epoch = 0  # for save best epoch

    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(
        model, train_loader, criterion, weight, device, min_value, max_value
    )
    if VERBOSE:
        print(
            "initial train loss: ",
            no_train_loss_res,
            "initial gdp train loss: ",
            no_train_gdp_loss_res,
        )

    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(
        model, val_loader, criterion, weight, device, min_value, max_value
    )
    if VERBOSE:
        print(
            "initial val loss: ",
            no_train_loss_res,
            "initial gdp val loss: ",
            no_train_gdp_loss_res,
        )

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_gdp_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, weight)

            gdp_loss = criterion(
                reverse_norm_with_scale(outputs, min_value, max_value),
                reverse_norm_with_scale(targets, min_value, max_value),
                weight,
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_gdp_loss += gdp_loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_gdp_loss = running_gdp_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        running_val_gdp_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets, weight)
                gdp_loss = criterion(
                    reverse_norm_with_scale(outputs, min_value, max_value),
                    reverse_norm_with_scale(targets, min_value, max_value),
                    weight,
                )

                running_val_loss += loss.item() * inputs.size(0)
                running_val_gdp_loss += gdp_loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_gdp_loss = running_val_gdp_loss / len(val_loader.dataset)

        # 如果当前验证损失小于最佳损失，则更新最佳模型权重和最佳 epoch
        if epoch_val_gdp_loss < best_val_gdp_loss:
            best_val_gdp_loss = epoch_val_gdp_loss
            best_model_wts = model.state_dict()
            best_epoch = epoch + 1  # 保存最佳 epoch（从 1 开始）

    # 返回最佳模型权重、最佳验证损失和最佳 epoch
    return best_model_wts, best_val_gdp_loss, best_epoch


# 主函数，执行超参数搜索和交叉验证
def hyperparameter_search(
    X,
    y,
    param_grid,
    k_folds=5,
    device="cpu",
    min_value=None,
    max_value=None,
    checkpoint_dir=None,
    file_item=None,
):
    # 将数据转换为TensorDataset
    # Use as_tensor to avoid unnecessary copy warnings when X/y are already tensors
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    )

    # 创建超参数组合
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    total_combinations = len(param_combinations)
    
    # Print search space info
    if file_item:
        tqdm.write(f"[LSTM-Q] Starting hyperparameter search for {file_item}")
    tqdm.write(f"[LSTM-Q] Search space: {total_combinations} combinations × {k_folds} folds = {total_combinations * k_folds} total evaluations")

    best_overall_loss = float("inf")
    best_params = None
    best_model_wts = None

    # 遍历每个超参数组合
    for combo_idx, param_values in enumerate(tqdm(param_combinations, desc="HP Search", unit="combo"), 1):
        params = dict(zip(param_names, param_values))
        
        # Show current params being tested
        key_params = f"h{params.get('hidden_dim', '?')}_l{params.get('num_layers', '?')}_lr{params.get('lr', '?'):.0e}"
        tqdm.write(f"[LSTM-Q] [{combo_idx}/{total_combinations}] Testing: {key_params}")

        # 存储每个折的验证损失
        val_losses = []

        # 创建KFold对象
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # 遍历每个折，记录每个折里最优的折和epoch
        record_best_val_gdp_loss = float("inf")
        record_best_epoch = 0
        record_best_fold = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
            if VERBOSE:
                print(f"Fold {fold}/{k_folds}")

            # 创建数据加载器
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(
                train_subset, batch_size=params["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=params["batch_size"], shuffle=False
            )

            # 初始化模型、损失函数和优化器
            model = LSTMModel(
                input_dim=X.shape[-1],
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                output_dim=X.shape[-1],
                dropout_rate=params["dropout_rate"],
            ).to(device)
            criterion = loss_weight
            optimizer = optim.AdamW(
                model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
            )

            # 训练和验证
            best_model_wts, best_val_gdp_loss, best_epoch = train_and_evaluate(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                params["num_epochs"],
                params["weight"],
                device,
                min_value,
                max_value,
            )
            val_losses.append(best_val_gdp_loss)
            if VERBOSE:
                print(
                    f"Best Validation GDP Loss for fold {fold}: {best_val_gdp_loss:.4f}"
                )
                print(f"Best epoch: {best_epoch}")

            if best_val_gdp_loss < record_best_val_gdp_loss:
                record_best_val_gdp_loss = best_val_gdp_loss
                record_best_epoch = best_epoch
                record_best_fold = fold
                best_valid_model = model

        # 计算当前超参数组合的平均验证损失
        avg_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)
        
        # Always show result for this combo
        tqdm.write(f"[LSTM-Q] [{combo_idx}/{total_combinations}] {key_params} → avg_loss={avg_val_loss:.4f} (±{std_val_loss:.4f})")

        # 如果当前平均验证损失小于整体最佳损失，则更新最佳参数和模型权重
        if avg_val_loss < best_overall_loss:
            best_overall_loss = avg_val_loss
            best_params = params
            best_params["record_best_epoch"] = record_best_epoch
            best_params["record_best_val_gdp_loss"] = record_best_val_gdp_loss
            best_params["record_best_fold"] = record_best_fold
            tqdm.write(f"[LSTM-Q] ✓ New best! loss={best_overall_loss:.4f}")
            # Save the best model for this hyperparameter combo if we have a target path
            if checkpoint_dir is None:
                checkpoint_dir = "checkpoints_lstm/"
            # Ensure checkpoint directory exists
            os.makedirs(checkpoint_dir, exist_ok=True)
            if file_item is None:
                model_save_path = os.path.join(
                    checkpoint_dir, "lstm_best_valid_model.pth"
                )
            else:
                model_save_path = os.path.join(
                    checkpoint_dir,
                    file_item.replace(".pt", "_") + "lstm_best_valid_model.pth",
                )
            try:
                torch.save(model.state_dict(), model_save_path)
                if VERBOSE:
                    print(f"\nBest valid model saved to {model_save_path}")
            except Exception:
                if VERBOSE:
                    print("Warning: could not save best model to disk.")

    # Always print final summary
    if best_params:
        tqdm.write(f"\n[LSTM-Q] ✓ Hyperparameter search complete!")
        tqdm.write(f"[LSTM-Q] Best params: h{best_params.get('hidden_dim', '?')}, l{best_params.get('num_layers', '?')}, lr{best_params.get('lr', '?'):.0e}, bs{best_params.get('batch_size', '?')}")
        tqdm.write(f"[LSTM-Q] Best avg loss: {best_overall_loss:.4f}")
    return best_params, best_overall_loss


# 使用最佳超参数在整个训练数据集上训练最终模型，并计算在测试集的performance
def train_and_evaluate_final(
    train_data,
    test_data,
    train_targets,
    test_targets,
    best_params,
    device="cpu",
    file_item=None,
    checkpoint_dir="checkpoints_lstm/",
    min_value=None,
    max_value=None,
):

    # 创建TensorDataset和DataLoader
    train_dataset = TensorDataset(train_data, train_targets)
    test_dataset = TensorDataset(test_data, test_targets)
    train_dataloader = DataLoader(
        train_dataset, batch_size=best_params["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=best_params["batch_size"], shuffle=False
    )

    final_model = LSTMModel(
        input_dim=train_data.shape[-1],
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        output_dim=train_data.shape[-1],
        dropout_rate=best_params["dropout_rate"],
    ).to(device)
    final_criterion = loss_weight
    final_optimizer = optim.AdamW(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    weight = best_params["weight"]

    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(
        final_model,
        train_dataloader,
        final_criterion,
        weight,
        device,
        min_value,
        max_value,
    )
    if VERBOSE:
        print(
            "initial train loss: ",
            no_train_loss_res,
            "initial gdp train loss: ",
            no_train_gdp_loss_res,
        )

    no_train_loss_res, no_train_gdp_loss_res = no_train_loss(
        final_model,
        test_dataloader,
        final_criterion,
        weight,
        device,
        min_value,
        max_value,
    )
    if VERBOSE:
        print(
            "initial val loss: ",
            no_train_loss_res,
            "initial gdp val loss: ",
            no_train_gdp_loss_res,
        )

    # 训练最终模型
    train_losses = []
    test_losses = []
    train_gdp_losses = []
    test_gdp_losses = []
    num_epochs = best_params["record_best_epoch"]
    for epoch in range(num_epochs):
        total_loss = 0
        total_gdp_loss = 0
        for batch_data, batch_labels in train_dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # 前向传播
            final_model.train()
            outputs = final_model(batch_data)

            loss = final_criterion(outputs, batch_labels, weight)

            gdp_loss = final_criterion(
                reverse_norm_with_scale(outputs, min_value, max_value),
                reverse_norm_with_scale(batch_labels, min_value, max_value),
                weight,
            )

            final_optimizer.zero_grad()
            loss.backward()
            final_optimizer.step()

            total_loss += loss.item() * batch_data.size(0)
            total_gdp_loss += gdp_loss.item() * batch_data.size(0)

        train_loss = total_loss / len(train_dataloader.dataset)
        train_gdp_loss = total_gdp_loss / len(train_dataloader.dataset)

        train_losses.append(train_loss)
        train_gdp_losses.append(train_gdp_loss)

        preds = []
        trues = []
        # 评估模型
        final_model.eval()
        with torch.no_grad():
            test_loss = 0
            test_gdp_loss = 0
            for batch_data, batch_targets in test_dataloader:
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)

                outputs = final_model(batch_data)

                loss = final_criterion(outputs, batch_targets, weight)

                gdp_loss = final_criterion(
                    reverse_norm_with_scale(outputs, min_value, max_value),
                    reverse_norm_with_scale(batch_targets, min_value, max_value),
                    weight,
                )

                test_loss += loss.item() * batch_data.size(0)
                test_gdp_loss += gdp_loss.item() * batch_data.size(0)

                outputs = reverse_norm_with_scale(outputs, min_value, max_value)
                batch_targets = reverse_norm_with_scale(
                    batch_targets, min_value, max_value
                )

                for item in outputs:
                    preds.append(item.detach().cpu().numpy())
                for item in batch_targets:
                    trues.append(item.detach().cpu().numpy())

            test_loss = test_loss / len(test_dataloader.dataset)
            test_gdp_loss = test_gdp_loss / len(test_dataloader.dataset)

            test_losses.append(test_loss)
            test_gdp_losses.append(test_gdp_loss)

        preds = torch.Tensor(np.array(preds))
        trues = torch.Tensor(np.array(trues))

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

    if VERBOSE:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, GDP Train Loss: {train_gdp_loss:.4f}, GDP Test Loss: {test_gdp_loss:.4f}"
        )
        print("mae, mse, rmse, mape: ", mae, mse, rmse, mape)
        print("Training complete!")

    # # 保存最终模型
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if file_item is not None:
        model_save_path = os.path.join(
            checkpoint_dir, file_item.replace(".pt", "_") + "lstm_best_final_model.pth"
        )
    else:
        model_save_path = os.path.join(checkpoint_dir, "lstm_best_final_model.pth")
    torch.save(final_model.state_dict(), model_save_path)
    if VERBOSE:
        print(f"\nFinal model saved to {model_save_path}")

    best_params["final model mae"] = mae
    best_params["final model mse"] = mse
    best_params["final model rmse"] = rmse
    best_params["final model mape"] = mape
    return best_params


# 使用checkpoint在测试集上测试
def eval_model(
    test_data,
    test_targets,
    model_path,
    best_params,
    device="cpu",
    min_value=None,
    max_value=None,
):
    test_dataset = TensorDataset(test_data, test_targets)
    test_dataloader = DataLoader(
        test_dataset, batch_size=best_params["batch_size"], shuffle=False
    )

    final_model = LSTMModel(
        input_dim=test_data.shape[-1],
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        output_dim=test_data.shape[-1],
        dropout_rate=best_params["dropout_rate"],
    ).to(device)
    final_criterion = loss_weight
    final_optimizer = optim.AdamW(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    # 3. 加载模型权重
    # If the checkpoint file does not exist, skip evaluation with a warning.
    if not os.path.exists(model_path):
        if VERBOSE:
            print(
                f"Warning: model checkpoint not found at {model_path}. Skipping evaluation."
            )
        return best_params

    # Try to load checkpoint; if shapes mismatch, load matching keys and warn.
    try:
        checkpoint = torch.load(model_path, map_location=device)
        try:
            final_model.load_state_dict(checkpoint)
        except RuntimeError as e:
            # Shape mismatch: attempt to load only matching parameters
            model_state = final_model.state_dict()
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                ckpt_state = checkpoint["state_dict"]
            else:
                ckpt_state = checkpoint

            matched = {}
            skipped = []
            for k, v in ckpt_state.items():
                if k in model_state and v.size() == model_state[k].size():
                    matched[k] = v
                else:
                    skipped.append(k)

            if len(matched) > 0:
                model_state.update(matched)
                final_model.load_state_dict(model_state)
                if VERBOSE:
                    print(
                        f"Warning: loaded {len(matched)} matching params from checkpoint; skipped {len(skipped)} params due to shape mismatch."
                    )
                    if len(skipped) > 0:
                        print(f"Skipped keys (example): {skipped[:10]}")
            else:
                raise RuntimeError(
                    f"Checkpoint at {model_path} has no parameters matching the current model shapes. Original error: {e}"
                )
    except FileNotFoundError:
        raise RuntimeError(f"Model checkpoint not found at {model_path}")
    except Exception as e:
        # Re-raise unexpected exceptions
        raise

    # 4. 设置模型为评估模式
    test_losses = []
    test_gdp_losses = []
    preds = []
    trues = []
    weight = best_params["weight"]
    # 评估模型
    final_model.eval()
    with torch.no_grad():
        test_loss = 0
        test_gdp_loss = 0
        for batch_data, batch_targets in test_dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)

            outputs = final_model(batch_data)
            loss = final_criterion(outputs, batch_targets, weight)

            gdp_loss = final_criterion(
                reverse_norm_with_scale(outputs, min_value, max_value),
                reverse_norm_with_scale(batch_targets, min_value, max_value),
                weight,
            )

            test_loss += loss.item() * batch_data.size(0)
            test_gdp_loss += gdp_loss.item() * batch_data.size(0)

            outputs = reverse_norm_with_scale(outputs, min_value, max_value)
            batch_targets = reverse_norm_with_scale(batch_targets, min_value, max_value)

            for item in outputs:
                preds.append(item.detach().cpu().numpy())

            for item in batch_targets:
                trues.append(item.detach().cpu().numpy())

        test_loss = test_loss / len(test_dataloader.dataset)
        test_gdp_loss = test_gdp_loss / len(test_dataloader.dataset)

        test_losses.append(test_loss)
        test_gdp_losses.append(test_gdp_loss)

    preds = torch.Tensor(np.array(preds))
    trues = torch.Tensor(np.array(trues))

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

    if VERBOSE:
        print(f"Test Loss: {test_loss:.4f}, GDP Test Loss: {test_gdp_loss:.4f}")
        print("mae, mse, rmse, mape: ", mae, mse, rmse, mape)

    best_params["best val model mae"] = mae
    best_params["best val model mse"] = mse
    best_params["best val model rmse"] = rmse
    best_params["best val model mape"] = mape
    return best_params


def run_with_config(lstm_q_config, seed=1, device=None):
    data_dir = lstm_q_config["dataset_path"]
    file_item_list = [
        f for f in os.listdir(data_dir) if lstm_q_config["file_pattern"] in f
    ]

    if VERBOSE:
        print("file item list length: ", len(file_item_list))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Option: when single_output is True, use a single CSV and (optionally) a single pth
    single_output = bool(lstm_q_config.get("single_output", False))

    # Ensure checkpoint dir exists
    checkpoint_dir = lstm_q_config.get("checkpoint_dir", "checkpoints_lstm/")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Path for aggregated CSV when single_output is enabled
    agg_csv_path = (
        os.path.join(checkpoint_dir, "best_params_res.csv") if single_output else None
    )

    for file_item in tqdm(file_item_list[:]):
        if VERBOSE:
            print(file_item)
        start_time = time.time()
        data_path = os.path.join(data_dir, file_item)
        label_path = os.path.join(
            data_dir, file_item.replace("LSTM_data", "LSTM_label")
        )

        set_seed(seed)
        data = torch.load(data_path)
        labels = torch.load(label_path)

        data, labels, _min_value, _max_value = norm_lstm_tensor(
            data, labels, lstm_q_config.get("freq", "quarter")
        )

        # Keep min/max available locally and pass explicitly to functions
        min_value = _min_value
        max_value = _max_value

        # determine test year using centralized config
        if "13-19" in file_item and "13-19" in lstm_q_config.get("test_year", {}):
            year = lstm_q_config["test_year"]["13-19"]
        else:
            year = lstm_q_config["test_year"].get("default", 2018)

        train_data, test_data, train_targets, test_targets = split_lstm_dataset_by_year(
            data, labels, year, freq=lstm_q_config.get("freq", "quarter")
        )

        set_seed(seed)

        # use centralized param grid
        param_grid = lstm_q_config["param_grid"]

        # 执行超参数搜索
        # If single_output is requested, pass file_item=None so hyperparameter_search
        # will save model to a single common filename inside checkpoint_dir.
        hp_file_item = None if single_output else file_item
        best_params, best_overall_loss = hyperparameter_search(
            train_data,
            train_targets,
            param_grid,
            k_folds=lstm_q_config.get("k_folds", 5),
            device=device,
            min_value=min_value,
            max_value=max_value,
            checkpoint_dir=checkpoint_dir,
            file_item=hp_file_item,
        )

        best_params["best_overall_loss_average"] = best_overall_loss

        set_seed(seed)
        # Train final model on full train set. Respect single_output flag.
        final_file_item = None if single_output else file_item
        best_params = train_and_evaluate_final(
            train_data,
            test_data,
            train_targets,
            test_targets,
            best_params,
            device=device,
            file_item=final_file_item,
            checkpoint_dir=checkpoint_dir,
            min_value=min_value,
            max_value=max_value,
        )

        # Determine which model file to evaluate: if single_output, use common filename
        if single_output:
            model_path = os.path.join(checkpoint_dir, "lstm_best_valid_model.pth")
        else:
            model_path = os.path.join(
                checkpoint_dir,
                file_item.replace(".pt", "_") + "lstm_best_valid_model.pth",
            )
        best_params = eval_model(
            test_data,
            test_targets,
            model_path,
            best_params,
            device,
            min_value,
            max_value,
        )
        best_params["train_data shape"] = ", ".join([str(x) for x in train_data.shape])
        best_params["test_data shape"] = ", ".join([str(x) for x in test_data.shape])
        best_params["train_targets shape"] = ", ".join(
            [str(x) for x in train_targets.shape]
        )
        best_params["test_targets shape"] = ", ".join(
            [str(x) for x in test_targets.shape]
        )
        # Save best_params: either append to an aggregated CSV (single_output)
        # or write a per-file CSV as before.
        if single_output and agg_csv_path is not None:
            write_header = not os.path.exists(agg_csv_path)
            pd.DataFrame([best_params]).to_csv(
                agg_csv_path, index=False, mode="a", header=write_header
            )
        else:
            pd.DataFrame([best_params]).to_csv(
                os.path.join(
                    checkpoint_dir,
                    file_item.replace(".pt", "_") + "best_params_res.csv",
                )
            )
        if VERBOSE:
            print("cost time: ", time.time() - start_time)
            print("\n===================Next=====================")


def _main():
    try:
        from master_config import LSTM_Q_CONFIG, SEED, CUDA_VISIBLE_DEVICES

        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_with_config(LSTM_Q_CONFIG, seed=SEED, device=device)
    except Exception as e:
        raise RuntimeError(
            "When running as a script, provide a `master_config.py` or call `run_with_config` from a notebook."
            + str(e)
        )


if __name__ == "__main__":
    _main()
