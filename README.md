# N-Gram Language Modeling and Evaluation

## Overview
This project implements and evaluates various N-gram Language Models (LMs) to understand the trade-offs between different N-gram orders, and the critical role of smoothing and backoff techniques in handling data sparsity.

## Files Description
- `main.py` - Main implementation with all N-gram models and evaluation
- `data_setup.py` - Script to download and set up Penn Treebank dataset
- `ptb.train.txt` - Training data (Penn Treebank)
- `ptb.valid.txt` - Validation/Development data
- `ptb.test.txt` - Test data
- `README.md` - This file
- `REPORT.md` - Detailed analysis and discussion report

## Requirements
- Python 3.7+
- numpy
- kagglehub (for dataset download)

## Installation

### 1. Install Required Packages
```bash
pip install numpy kagglehub
```

### 2. Download Dataset
Run the data setup script to download the Penn Treebank dataset:
```bash
python data_setup.py
```

## Running the Code

### Quick Start
```bash
python main.py
```

### What the Code Does
The script will automatically:

1. **Load Data**: Load Penn Treebank training, validation, and test sets
2. **Train MLE Models**: Train 1-gram, 2-gram, 3-gram, and 4-gram models using Maximum Likelihood Estimation
3. **Train Smoothed Models**: 
   - Add-1 (Laplace) smoothing for trigram model
4. **Tune Hyperparameters**: 
   - Find optimal interpolation weights using validation data
   - Find optimal alpha parameter for Stupid Backoff
5. **Evaluate Models**: Calculate perplexity on test set for all models
6. **Generate Text**: Generate sample sentences using the best performing model
7. **Display Results**: Show formatted tables with all results

## Model Implementations

### 1. Maximum Likelihood Estimation (MLE) Models
- **Unigram (1-gram)**: P(w)
- **Bigram (2-gram)**: P(w|w-1)
- **Trigram (3-gram)**: P(w|w-2,w-1)
- **4-gram**: P(w|w-3,w-2,w-1)

### 2. Smoothing Techniques
- **Add-1 (Laplace) Smoothing**: P(w|context) = (Count(context,w) + 1) / (Count(context) + V)

### 3. Backoff and Interpolation
- **Linear Interpolation**: P(w|w1,w2) = λ1*P(w) + λ2*P(w|w1) + λ3*P(w|w1,w2)
- **Stupid Backoff**: Backoff from trigram → bigram → unigram with discount factor α

## Key Features

### Hyperparameter Tuning
- **Interpolation Weights**: Tests 8 different weight combinations to find optimal λ1, λ2, λ3
- **Backoff Alpha**: Tests 7 different alpha values (0.1 to 0.7) for optimal backoff

### Text Generation
- Generates sample sentences using the best performing model
- Uses probability sampling for natural text generation

### Comprehensive Evaluation
- Perplexity calculation for all models
- Proper handling of zero probabilities (INF perplexity)
- Development set used for hyperparameter tuning
- Test set used for final evaluation
