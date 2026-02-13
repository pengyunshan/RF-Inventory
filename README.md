# RF Contract Inventory Forecasting Model Framework

This is a machine learning model framework for Reach & Frequency (RF) contract inventory forecasting, supporting training and evaluation of multiple baseline models.

## Dataset Description

This dataset contains basic information about RF contracts, including:
- **Targeting Conditions**: Geography, age, gender, etc.
- **Delivery Schedule**: Number of delivery days
- **Frequency Capping Rules**: Frequency limits based on time windows (e.g., "3 times in 7 days")
- **Budget-Performance Curves**: Reach (UV) and impressions (PV) under different CPM thresholds

## Project Structure

```
rf_git/
├── data/                          # Data files
│   ├── df_train1.pq              # Training data 1 (Schedule split)
│   ├── df_test1.pq               # Test data 1
│   ├── df_train2.pq              # Training data 2 (CPM split)
│   ├── df_test2.pq               # Test data 2
│   ├── df_train3.pq              # Training data 3 (Joint split)
│   ├── df_test3.pq               # Test data 3
│   └── cache/                    # Cached preprocessed data
├── models/                       # Model implementations
│   ├── __init__.py               # Module initialization
│   ├── base_model.py             # Abstract base class for all models
│   ├── model_factory.py          # Model factory and management
│   ├── ridge_model.py            # Ridge regression
│   ├── gbdt_model.py             # Gradient Boosting Decision Tree
│   ├── mlp_model.py              # Multi-Layer Perceptron
│   ├── monotonic_mlp_model.py    # MLP with monotonicity constraints
│   ├── logistic_regression_model.py  # Logistic regression
│   ├── nearest_fc_model.py       # Nearest neighbor retrieval baseline
│   ├── tf_base_model.py          # TensorFlow model base wrapper
│   ├── tf_model_factory.py       # TensorFlow model factory
│   ├── tf_mlp_model.py           # TensorFlow MLP
│   ├── tf_posnn_model.py         # TensorFlow Positive NN (monotonic)
│   ├── tf_minmax_model.py        # MinMax model with monotonicity
│   ├── tf_smoothed_minmax_model.py  # Smoothed MinMax model
│   ├── tf_constrained_model.py   # Constrained model
│   ├── tf_hint_model.py          # Hint learning model
│   ├── tf_pwl_model.py           # Piecewise linear model
│   ├── tf_gcm_model.py           # Generative constraint model
│   ├── tf_igcm_model.py          # Improved GCM
│   ├── tf_scalable_model.py      # Scalable model
│   └── tf_utils.py               # TensorFlow utilities
├── utils/                        # Utility modules
│   ├── __init__.py               # Module initialization
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── evaluation.py             # Metric calculation
│   ├── visualization.py          # Visualization tools
│   └── config.py                 # Configuration utilities
├── scripts/                      # Script files
│   ├── __init__.py               # Module initialization
│   └── train_and_evaluate.py     # Training and evaluation script
├── experiments/                  # Experiment output directory
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies
└── run.sh                        # Batch run script
```

some model base code from https://github.com/tyxaaron/GCM

## Feature Engineering

### Feature Processing Strategy
- **Categorical Features**: `age`, `gender`, `fre_region_province` are encoded using LabelEncoder
- **Numerical Features**: 
  - `freq_limit`: Split by underscore into `freq_limit_days` (days) and `freq_limit_count` (frequency cap)
  - `periods`: Delivery schedule length
  - `cpm_thr`: CPM threshold
- **Excluded Features**: 
  - `fre_region_country` is not used as a model feature
  - `budget` is not used as a model feature

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Single Experiment

```bash
# Using pre-split data - Schedule extrapolation split (train on short schedules, test on long schedules)
python scripts/train_and_evaluate.py \
    --train_path data/df_train1.pq \
    --test_path data/df_test1.pq \
    --split_type schedule \
    --output_dir experiments/schedule_split

# CPM extrapolation split (train on low CPM, test on high CPM)
python scripts/train_and_evaluate.py \
    --train_path data/df_train2.pq \
    --test_path data/df_test2.pq \
    --split_type cpm \
    --output_dir experiments/cpm_split

# Joint extrapolation split (train on low CPM + short schedules, test on high CPM + long schedules)
python scripts/train_and_evaluate.py \
    --train_path data/df_train3.pq \
    --test_path data/df_test3.pq \
    --split_type joint \
    --output_dir experiments/joint_split
```

### 2. Train Specific Models

```bash
# Train only GBDT model
python scripts/train_and_evaluate.py \
    --train_path data/df_train3.pq \
    --test_path data/df_test3.pq \
    --split_type joint \
    --models gbdt \
    --output_dir experiments/gbdt_only

# Train TensorFlow models
python scripts/train_and_evaluate.py \
    --train_path data/df_train3.pq \
    --test_path data/df_test3.pq \
    --split_type joint \
    --models tf_smoothed_minmax tf_posnn \
    --output_dir experiments/tf_models
```

### 3. Batch Run

Use the provided `run.sh` script to run multiple experiments:

```bash
bash run.sh
```

## Key Features

### 📊 Budget-Performance Curve Visualization
- Automatically generate budget-performance curve charts
- Visualize model prediction quality during evaluation
- Separate visualization output for each model

```bash
# Generate visualization during evaluation
python scripts/train_and_evaluate.py \
    --train_path data/df_train3.pq \
    --test_path data/df_test3.pq \
    --split_type joint \
    --visualize \
    --n_vis_contexts 3
```

### 🚀 Training Progress Visualization
- Use `tqdm` progress bars to display training status
- Real-time display of PV_RMSE metrics for each model
- More transparent and controllable training process

### 🧠 TensorFlow Deep Learning Models (Optional)
- Support for TensorFlow neural network models with various architectures
- Models with monotonicity constraints (TF_PosNN, TF_MinMax, etc.)
- Shared evaluation framework with sklearn models

**Available TensorFlow Models:**
- `tf_mlp`: Standard MLP
- `tf_posnn`: Positive Neural Network (monotonic constraints)
- `tf_minmax`: MinMax model
- `tf_smoothed_minmax`: Smoothed MinMax model
- `tf_constrained`: Constrained model
- `tf_hint`: Hint learning model
- `tf_pwl`: Piecewise linear model
- `tf_gcm`: Generative constraint model
- `tf_igcm`: Improved generative constraint model
- `tf_scalable`: Scalable model

```bash
# Train TensorFlow models
python scripts/train_and_evaluate.py \
    --train_path data/df_train3.pq \
    --test_path data/df_test3.pq \
    --split_type joint \
    --models tf_smoothed_minmax tf_posnn
```

### 📝 Detailed Logging
- Detailed logs with file names and line numbers
- Easier debugging and issue location
- Support for different log levels

## Supported Models

Models are designed in a modular fashion, with each model as a separate file:

### Sklearn-based Models
1. **Ridge** (`models/ridge_model.py`): Ridge regression model
2. **GBDT** (`models/gbdt_model.py`): Histogram-based Gradient Boosting Decision Tree
3. **MLP** (`models/mlp_model.py`): Multi-Layer Perceptron
4. **Monotonic_MLP** (`models/monotonic_mlp_model.py`): MLP with monotonicity constraints
5. **Logistic_Regression** (`models/logistic_regression_model.py`): Logistic regression baseline
6. **Nearest_FC** (`models/nearest_fc_model.py`): Nearest neighbor retrieval baseline

### TensorFlow Models (Optional)
7. **TF_MLP** (`models/tf_mlp_model.py`): TensorFlow Multi-Layer Perceptron
8. **TF_PosNN** (`models/tf_posnn_model.py`): Positive Neural Network with monotonicity constraints
9. **TF_MinMax** (`models/tf_minmax_model.py`): MinMax model ensuring monotonicity
10. **TF_Smoothed_MinMax** (`models/tf_smoothed_minmax_model.py`): Smoothed MinMax with LogSumExp
11. **TF_Constrained** (`models/tf_constrained_model.py`): Model with special activation functions
12. **TF_Hint** (`models/tf_hint_model.py`): Hint learning model
13. **TF_PWL** (`models/tf_pwl_model.py`): Piecewise linear model with gradient regularization
14. **TF_GCM** (`models/tf_gcm_model.py`): Generative constraint model
15. **TF_IGCM** (`models/tf_igcm_model.py`): Improved generative constraint model
16. **TF_Scalable** (`models/tf_scalable_model.py`): Scalable model

### Model Base Classes
- **BaseModel** (`models/base_model.py`): Abstract base class for all models
- **ModelFactory** (`models/model_factory.py`): Sklearn model management and factory
- **TFModelWrapper** (`models/tf_base_model.py`): TensorFlow model wrapper for integration
- **TFModelFactory** (`models/tf_model_factory.py`): TensorFlow model factory

## Supported Data Split Types

1. **random**: Random split
2. **schedule**: Schedule extrapolation split (train on short schedules, test on long schedules)
   - Uses `df_train1.pq` / `df_test1.pq`
3. **cpm**: CPM extrapolation split (train on low CPM, test on high CPM)
   - Uses `df_train2.pq` / `df_test2.pq`
4. **joint**: Joint extrapolation split (train on low CPM + short schedules, test on high CPM + long schedules)
   - Uses `df_train3.pq` / `df_test3.pq`

## Evaluation Metrics

### Point-level Prediction Accuracy
- **PV_MAE**: Page views mean absolute error
- **PV_RMSE**: Page views root mean squared error
- **UV_MAE**: Unique visitors mean absolute error
- **UV_RMSE**: Unique visitors root mean squared error

### Structural Consistency
- **Violation_Rate**: Monotonicity violation rate
- **Violation_Magnitude**: Monotonicity violation magnitude
- **Constraint_Violation_Rate**: Theoretical constraint violation rate

## Custom Models

To add a new model, inherit from the `BaseModel` class:

```python
from models.base_model import BaseModel

class YourModel(BaseModel):
    def __init__(self, your_params):
        super().__init__("YourModel")
        # Initialize your model
    
    def fit(self, X, y_pv, y_uv, theoretical_max=None):
        # Implement training logic
        pass
    
    def predict(self, X):
        # Implement prediction logic
        return y_pv_pred, y_uv_pred
```

Register your model in `models/model_factory.py` to use it with the training script.

## Configuration

### Model Parameters

**GBDT Configuration:**
- The project includes a GBDT configuration system in `models/model_factory.py`
- Default optimized configuration is used automatically
- Supports monotonicity constraints and categorical feature handling

**Common Parameters:**
- `alpha`: Regularization parameter for Ridge model
- `max_iter`: Maximum iterations for GBDT
- `max_depth`: Maximum tree depth for GBDT
- `learning_rate`: Learning rate for GBDT
- `hidden_layer_sizes`: Hidden layer structure for MLP models

### Data Split Parameters
- `test_size`: Test set proportion (for random split)
- `random_state`: Random seed

### Visualization Parameters
- `--visualize` / `--no_visualize`: Enable/disable visualization
- `--n_vis_contexts`: Number of contexts to visualize (default: 3)
- `--save_predictions` / `--no_save_predictions`: Save prediction CSV files

### Output Parameters
- `--save_models` / `--no_save_models`: Save trained models
- `--output_dir`: Directory for experiment outputs
- `--model_dir`: Directory for saved models (default: output_dir/models)

## Output Results

Experiment results are saved in the specified output directory:

**Files Generated:**
- `results_{split_type}.csv`: Evaluation metrics summary for all models
- `{model_name}_predictions.csv`: Detailed prediction results (features + true values + predictions)
- `visualizations_{model_name}/`: Visualization charts for each model (if enabled)
  - `context_*.png`: Budget-performance curves for different contexts
- `models/`: Saved model files (if `--save_models` is enabled)
  - `{model_name}.pkl`: Pickled model objects

**Example Output Structure:**
```
experiments/joint_split/
├── results_joint.csv
├── gbdt_predictions.csv
├── ridge_predictions.csv
├── models/
│   ├── gbdt_model.pkl
│   └── ridge_model.pkl
└── visualizations_gbdt/
    ├── context_1_age0_gender1_....png
    └── context_2_age1_gender0_....png
```

## Paper Task Mapping

### Task 1: Point-level Performance Prediction
Use point-level prediction accuracy metrics (MAE, RMSE) to evaluate model performance.

### Task 2: Multi-dimensional Monotonic Delivery Result Prediction
Use structural consistency metrics (Violation Rate, Constraint Violation Rate) to evaluate model monotonicity.

## Contact

For questions or suggestions, please contact the project maintainers.# RF-Inventory
