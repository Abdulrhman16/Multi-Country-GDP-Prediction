# Wandb Logging - Enhanced Metrics

This document describes all the metrics and variables being logged to wandb.

## Metrics Logged During Training

### Per Epoch (in train_and_evaluate):
- `epoch`: Current epoch number
- `train_loss`: Training loss (normalized)
- `train_gdp_loss`: Training GDP loss (denormalized)
- `val_loss`: Validation loss (normalized)
- `val_gdp_loss`: Validation GDP loss (denormalized)
- `val_mae`: Validation Mean Absolute Error
- `val_mse`: Validation Mean Squared Error
- `val_rmse`: Validation Root Mean Squared Error
- `val_mape`: Validation Mean Absolute Percentage Error
- `val_r2`: Validation R² Score
- `val_rse`: Validation Root Squared Error
- `val_corr`: Validation Correlation
- `learning_rate`: Current learning rate

### Per Epoch (in train_and_evaluate_final):
- `epoch`: Current epoch number
- `train_loss`: Training loss
- `train_gdp_loss`: Training GDP loss
- `test_loss`: Test loss
- `test_gdp_loss`: Test GDP loss
- `test_mae`: Test Mean Absolute Error
- `test_mse`: Test Mean Squared Error
- `test_rmse`: Test Root Mean Squared Error
- `test_mape`: Test Mean Absolute Percentage Error
- `test_mspe`: Test Mean Squared Percentage Error
- `test_rse`: Test Root Squared Error
- `test_corr`: Test Correlation
- `test_r2`: Test R² Score
- `learning_rate`: Current learning rate

### Final Metrics:
- `final_mae`: Final Mean Absolute Error
- `final_mse`: Final Mean Squared Error
- `final_rmse`: Final Root Mean Squared Error
- `final_mape`: Final Mean Absolute Percentage Error
- `final_mspe`: Final Mean Squared Percentage Error
- `final_rse`: Final Root Squared Error
- `final_corr`: Final Correlation
- `final_r2`: Final R² Score
- `best_epoch`: Best epoch number
- `train_samples`: Number of training samples
- `test_samples`: Number of test samples

## Configuration Logged:
- `hidden_dim`: Hidden dimension
- `num_layers`: Number of layers
- `dropout_rate`: Dropout rate
- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `weight_decay`: Weight decay
- `use_batch_norm`: Whether batch normalization is used
- `use_residual`: Whether residual connections are used
- `dataset_file`: Dataset file name
- `train_samples`: Number of training samples
- `test_samples`: Number of test samples
- `input_features`: Number of input features
- `test_year`: Test year split
- `best_hidden_dim`: Best hidden dimension from hyperparameter search
- `best_num_layers`: Best number of layers
- `best_dropout_rate`: Best dropout rate
- `best_learning_rate`: Best learning rate
- `best_batch_size`: Best batch size
- `best_weight_decay`: Best weight decay
- `best_overall_loss`: Best overall validation loss
- `best_val_loss`: Best validation loss
- `best_epoch`: Best epoch number

## Visualizations Logged:
- `training_curves`: Training and validation loss curves
- `predictions_vs_actual`: Predictions vs actual values plots
- `residual_analysis`: Residual plots
- `metrics_summary`: Metrics summary bar chart

