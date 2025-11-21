# Improved MLP Model for GDP Forecasting

This folder contains an improved MLP (Multi-Layer Perceptron) model designed to address common problems with traditional MLP architectures for GDP prediction.

## Model Architecture Improvements

The improved model includes several enhancements over the baseline MLP:

### 1. **Residual Connections**
- Skip connections between layers to help with gradient flow
- Enables training of deeper networks without vanishing gradient problems
- Improves model capacity and learning efficiency

### 2. **Batch Normalization**
- Normalizes inputs to each layer for faster and more stable training
- Reduces internal covariate shift
- Allows for higher learning rates

### 3. **GELU Activation Function**
- Gaussian Error Linear Unit (GELU) instead of ReLU
- Smoother activation function that often performs better in practice
- Better gradient flow compared to ReLU

### 4. **Feature Interaction Layers**
- Captures interactions between features using element-wise products
- Helps the model learn non-linear feature relationships
- Improves model expressiveness

### 5. **Gradient Clipping**
- Prevents exploding gradients during training
- Improves training stability
- Helps with convergence

### 6. **Early Stopping**
- Prevents overfitting by stopping training when validation loss stops improving
- Saves computational resources
- Improves generalization

## Architecture Details

The model consists of:
- **Input Projection Layer**: Maps input features to hidden dimension
- **Feature Interaction Layer**: Captures feature interactions
- **Residual Blocks**: Multiple residual blocks with batch normalization
- **Output Projection Layers**: Gradual reduction to output dimension

## Usage

1. Open the `improved_mlp_model.ipynb` notebook
2. Run all cells sequentially
3. The model will:
   - Load the MLP dataset from the `../dataset` folder
   - Perform hyperparameter search with k-fold cross-validation
   - Train the final model with best hyperparameters
   - Evaluate on test set and save results

## Configuration

The model can be configured in the CONFIG dictionary:
- `dataset_path`: Path to dataset folder
- `file_pattern_suffix`: Use `"_q_"` for quarterly data or `"_y_"` for yearly data
- `freq`: "quarter" or "year"
- `param_grid`: Hyperparameter search space

## Expected Improvements

Compared to the baseline MLP, this model should show:
- Better generalization (lower test error)
- More stable training (fewer training issues)
- Better feature learning (captures complex patterns)
- Faster convergence (due to batch normalization)

## Output

The model saves:
- Best model checkpoint: `../checkpoints_improved_mlp/improved_mlp_best_final_model.pth`
- Results CSV: `../checkpoints_improved_mlp/[dataset_name]_best_params_res.csv`

## Requirements

Same as the main project:
- PyTorch
- NumPy
- Pandas
- scikit-learn
- tqdm

## Notes

- The model uses the same dataset format as the baseline MLP models
- Results will be compared with the baseline models after training
- The hyperparameter search may take some time depending on the search space

