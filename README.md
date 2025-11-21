
# RENTT: Runtime Efficient Network to Tree Transformation

This repository contains code for training neural networks (NN) and converting them to equivalent decision trees (DT) for regression and classification tasks. The RENTT framework enables the exact and scalable transformation of neural networks into multivariate decision trees, which can also be used to calculate ground-truth feature importance (FI) explanations with RENTT-FI.

## Overview

For details on the underlying theory and evaluation, see our paper:
["Efficiently Transforming Neural Networks into Decision Trees: A Path to Ground Truth Explanations with RENTT"](https://arxiv.org/abs/2511.09299)

## Features

  - Framework Support: TensorFlow and PyTorch neural networks
  - Tree Variants: Compact (pruned) or complete decision trees. Pruning removes the neuron activations that never fired during training.
  - Task Types: Regression and classification
  - Feature Importance:
      - RENTT-FI: Local (for one sample), regional (for a specific input region), and global (for the entire model) feature effects and contributions
      - SOTA comparison methods: 
          - local: LIME, SHAP, Break-Down (implemented via [dalex](https://dalex.drwhy.ai/)) [1]
          - global: Shapley Effects, SAGE (implemented via the [original implementation](https://github.com/iancovert/sage)) [2]
      - Agreement analysis: RMSE and Krippendorff's Alpha metrics (implementation [fast-krippendorff](https://github.com/pln-fing-udelar/fast-krippendorff/tree/main)) [3]
  - Flexible Configuration: Uses [Hydra](https://github.com/facebookresearch/hydra) [4] for flexible configuration management
  - Multiple data sets: Pre-configured support for various data sets


## Installation

Requirements:

- Python >=3.10, <3.13

```bash
pip install -e .
```

## Usage

### 1. Train/load neural network and transform them into decision trees with:

```bash
python -m scripts.main_RENTT
```

Configuration: 

Edit `config/RENTT.yaml` to customize:

  - dataset: Choose data set
  - framework: tensorflow or pytorch
  - activation_function: e.g., relu
  - load_nn_model: True to load existing model, False to train new model
  - train_test_split_ratio: Training/test split
  - use_complete_tree: True for complete tree, False for pruned
  - hidden_layers: List of hidden layer sizes
  - random_state: Random seed for reproducibility

Edit `config/dataset/*.yaml` to customize:

- model_filename: Model name consisting of hidden layers, date, and time, e.g. 8-4_20250611-1311


The resulting models are saved in `Models/`

### 2. Calculate Feature Importances with:

```bash
python -m scripts.main_RENTT-FI
```

Configuration: 

Edit `config/RENTT-FI.yaml` to enable/disable:

  - plot_analysis: True to generate violin plots of Krippendorff's Alpha and RMSE analysis

RENTT-FI Methods:
  - compute_local_fe: Local feature effects
  - compute_local_fc: Local feature contributions
  - compute_regional_fe: Regional feature effects
  - compute_regional_fc: Regional feature contributions
  - compute_global_fe: Global feature effects
  - compute_global_fc: Global feature contributions

Feature effects are the weighting of the different input features (how much the prediction changes for each unit increase in the feature), representing the data transformation of the neural network/decision tree. 
Feature contributions are the product of input values and effects, showing how much a feature contributes to the final prediction, and are commonly used for explanations.

Comparison Analyses:
  - compute_local_comparison_sota_nn_dt: Compare SOTA methods between NN and DT (local)
  - compute_local_comparison_sota_rentt: Compare SOTA with RENTT-FI (local)
  - compute_global_comparison_sota_nn_dt: Compare SOTA methods between NN and DT (global)
  - compute_global_comparison_sota_rentt: Compare SOTA with RENTT-FI (global)

Additional Parameters:
  - num_samples: Number of samples for FI calculation (null for all samples)



The Feature Importance results and comparison results are saved in `Feature_Importance/`


## Supported Data Sets

**Regression**

- **Absolute**  
  Synthetic function y=∣x∣: 1 feature, 1 output

- **Linear**  
  Synthetic function y=4⋅x1​+x2​+0.001⋅x3​: 3 features, 1 output

- **Diabetes (regression)**  
  [scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset): 442 samples, 10 features, 1 output

- **California Housing**  
  [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html): 20,640 samples, 8 features, 1 output

**Classification**

- **Iris**  
  [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html): 150 samples, 4 features, 3 classes

- **Diabetes (classification)**  
  [Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set): 768 samples, 8 features, 2 classes

- **Wine Quality**  
  [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html): 178 samples, 13 features, 3 classes

- **Car Evaluation**  
  [UCI](https://archive.ics.uci.edu/ml/datasets/car): 1,728 samples, 6 features, 4 classes

- **Forest Cover Type**  
  [UCI](https://archive.ics.uci.edu/ml/datasets/covertype): 581,012 samples, 54 features, 7 classes


## Directory Structure

```
project/
│
├── config/                          # Hydra configuration files
│   ├── RENTT.yaml                   # Main transformation config
│   └── RENTT-FI.yaml                # Feature importance config
│
├── scripts/                         # Main execution scripts
│   ├── main_RENTT.py                # NN-to-DT transformation
│   └── main_RENTT-FI.py             # Feature importance calculation
│
├── src/                             # Source code
│   ├── config.py                    # Configuration
│   ├── DT.py                        # Decision tree classes (Tree, CompleteTree)
│   ├── feature_importance.py        # FI calculation methods
│   ├── nn_tensorflow.py             # TensorFlow NN implementation
│   ├── nn_pytorch.py                # PyTorch NN implementation
│   ├── read_data.py                 # Data set loading and preprocessing
│   └── utils.py                     # Utility functions and enums
│
├── Data/                            # Data set storage
├── Models/                          # Saved models and accuracy results
├── Feature_Importance/              # FI results and plots
│
├── setup.py                         # Package installation
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Key Components
### Neural Network Training (main_RENTT.py)

  1. Load/prepare data set with train-test split
  2. Create neural network (train new model or load existing)
  3. Transform to decision tree (pruned or complete)
  4. Evaluate both models and save results

### Feature Importance Analysis (main_RENTT-FI.py)

  1. Load pretrained NN and transformed DT
  2. Calculate RENTT-FI metrics:
      - Local: Point-wise feature effects/contributions
      - Regional: Region-specific feature effects/contributions, where a region is determined as the part of the input space with the same neuron activation pattern 
      - Global: Data set-wide aggregated feature effects/contributions
  3. Compute SOTA baselines (LIME, SHAP, BD, Shapley Effects, SAGE)
  4. Perform comparative analysis:
      - RMSE: Measure differences between methods
      - Krippendorff Alpha: Assess inter-rater agreement (ordinal & interval)
  5. Generate visualizations and save results

## References
[1] Baniecki, H., Kretowicz, W., Piatyszek, P., Wisniewski, J., & Biecek, P. (2021). dalex: Responsible Machine Learning with Interactive Explainability and Fairness in Python. Journal of Machine Learning Research, 22(214), 1-7. http://jmlr.org/papers/v22/20-1473.html

[2] Covert, I., Lundberg, S., & Lee, S.-I. (2020). Understanding global feature contributions with additive importance measures. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

[3] Castro, S. (2017). Fast Krippendorff: Fast computation of Krippendorff's alpha agreement measure. GitHub. https://github.com/pln-fing-udelar/fast-krippendorff

[4] Yadan, O. (2019). Hydra: A framework for elegantly configuring complex applications. GitHub. https://github.com/facebookresearch/hydra


## Citation

If you use this code, please cite:
```bibtex
@misc{monke2025efficientlytransformingneuralnetworks,
      title={Efficiently Transforming Neural Networks into Decision Trees: A Path to Ground Truth Explanations with RENTT}, 
      author={Helena Monke and Benjamin Fresz and Marco Bernreuther and Yilin Chen and Marco F. Huber},
      year={2025},
      eprint={2511.09299},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.09299}, 
}
```