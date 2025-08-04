# Housing Price Prediction: Preprocessing Impact Demo

This repository contains a demonstration project showcasing how better preprocessing techniques can improve machine learning model performance. The project compares three approaches to housing price prediction using the Sberbank Housing dataset.

## ⚠️ Important Disclaimer

**This project is NOT about achieving the lowest possible prediction error.** The absolute RMSE values (in rubles) are high because:

- **Focus**: The goal is to demonstrate the **relative improvements** achieved through different preprocessing approaches
- **Methodology**: All models use the same Ridge regression algorithm
- **Purpose**: Show that systematic data preprocessing can deliver improvements even with basic models
- **Learning Objective**: Emphasize that clean data and proper preprocessing often trump complex algorithms and hyperparameter tuning

**Key Takeaway**: The improvements demonstrate that investing in data quality and preprocessing can deliver immediate benefits to any ML project.

## 🎯 Project Overview

The goal of this project is to demonstrate the impact that proper data preprocessing can have on model performance. We compare three approaches:

- **Basic Model** (`base.ipynb`): Minimal preprocessing approach
- **Advanced Model** (`advanced.ipynb`): Better preprocessing pipeline
- **Fine-tuned Basic Model** (`base_fine_tuned.ipynb`): Basic preprocessing with hyperparameter optimization

**Key Result**: The advanced preprocessing approach achieves better performance than both the basic model and the fine-tuned basic model, demonstrating that systematic data preprocessing can outperform hyperparameter optimization alone.

### Business Context
This case study addresses a critical industry challenge: **70% of AI project failures are due to data quality issues**, resulting in an estimated **$5T annual revenue loss globally**. This project demonstrates how strategic data preprocessing can deliver immediate, substantial improvements in AI project success rates.

## 📊 Dataset

The project uses the **Sberbank Housing Dataset**, which contains real estate data from Moscow, Russia. The dataset includes:

- **Target Variable**: `price_doc` (house price in rubles)
- **Features**: 20 variables including square footage, location data, building characteristics, and proximity to amenities
- **Size**: ~27,000 records after stripping the dataset down (for compute optimization reasons)

### Data Files
- `Data/sberbank_housing.csv` - Main dataset

### Initial Data State
The raw dataset exhibited typical real-world data quality issues:
- **Missing values**: Up to 50% in some features (life_sq: 5,537, build_year: 12,869, state: 13,067)
- **Inconsistent data types**: Numeric values stored as strings
- **No proper normalization**: Features on different scales
- **Outliers and noise**: In target variable and feature distributions
- **Categorical variables**: Requiring proper encoding strategies

## 🏗️ Project Structure

```
DAMA/
├── base.ipynb                    # Basic model with minimal preprocessing
├── advanced.ipynb                # Advanced model with comprehensive preprocessing
├── base_fine_tuned.ipynb         # Basic model with hyperparameter optimization
├── requirements.txt              # Python package dependencies
├── baseline_rmse.txt             # Basic model RMSE result
├── baseline_rmse_fine_tuned.txt  # Fine-tuned basic model RMSE result
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore file
└── Data/
    └── sberbank_housing.csv
```

**Note**: This project focuses on the Jupyter notebooks as the primary demonstration files. The notebooks contain complete analysis with outputs, visualizations, and detailed explanations.

## 🔬 Methodology

### Basic Approach (`base.ipynb`)
- **Minimal preprocessing**: Only basic cleaning and log transformation of target
- **Feature selection**: Uses only 6 complete numeric features (out of 17 total features)
- **Model**: Ridge regression with alpha=10.0
- **Limitations**: Ignores categorical data, cannot handle missing values effectively
- **Dataset size**: 27,000 records (21,600 training, 5,400 test)

### Advanced Approach (`advanced.ipynb`)
- **Comprehensive preprocessing pipeline**:
  - **Data cleaning**: Outlier removal, duplicate elimination, data validation
  - **Missing value imputation**: Hybrid approach (KNN for numeric, mode for categorical)
  - **Feature engineering**: Living efficiency, room size, floor ratio, log transformations, amenity score
  - **Categorical encoding**: Smart encoding based on cardinality (frequency, one-hot, ordinal)
  - **Scaling**: StandardScaler for all features
- **Model**: Ridge regression with alpha=10.0
- **Dataset size**: 11,826 records (9,460 training, 2,366 test) after cleaning

### Fine-tuned Basic Approach (`base_fine_tuned.ipynb`)
- **Same minimal preprocessing** as basic approach
- **Hyperparameter optimization**: Grid search over Ridge alpha values
- **Model**: Ridge regression with optimized alpha
- **Dataset size**: Same as basic approach (27,000 records)

## 📈 Results

| Model | RMSE (Log Scale) | RMSE (Rubles) | Features Used | Approach |
|-------|------------------|----------------|---------------|----------|
| Basic | 0.551 | 4,595,983 | 6 features | Minimal preprocessing |
| Fine-tuned Basic | 0.540 | 4,320,000 | 6 features | Basic preprocessing + hyperparameter tuning |
| Advanced | 0.502 | 3,340,000 | 17 features | Comprehensive preprocessing |

**Key Findings**:
- **Advanced preprocessing** achieves the best performance (27.30% improvement over basic)
- **Fine-tuning alone** provides modest improvement (6.0% over basic)
- **Advanced preprocessing** outperforms fine-tuning by 22.7%

### Key Performance Metrics:
- **Basic Model**: Uses only 6 complete numeric features, ignores 11 features with missing values
- **Fine-tuned Basic Model**: Same features as basic, but with optimized hyperparameters
- **Advanced Model**: Utilizes all 17 features through better preprocessing
- **Feature Engineering**: Creates 5 new predictive features (living efficiency, room size, floor ratio, log transformations, amenity score)
- **Data Quality**: Advanced model removes outliers and duplicates, resulting in cleaner but smaller dataset

### Business Impact
For a housing market application:
- **Basic RMSE**: ₽4.6M average prediction error
- **Fine-tuned Basic RMSE**: ₽4.3M average prediction error
- **Advanced RMSE**: ₽3.3M average prediction error
- **Practical benefit**: Advanced preprocessing provides ₽1.3M more accurate predictions than fine-tuning alone

This improvement translates to:
- More accurate property valuations
- Better investment decisions
- Reduced financial risk in real estate transactions

## 🚀 Getting Started

### Prerequisites

**Option 1: Install from requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: Install packages individually**
```bash
pip install pandas numpy scikit-learn matplotlib scipy jupyter ipykernel
```

### Required Packages
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing
- **matplotlib**: Data visualization
- **scipy**: Statistical functions
- **jupyter**: Jupyter notebook environment
- **ipykernel**: Python kernel for Jupyter

### Running the Models

**Jupyter Notebooks** (Primary Method):
1. **Basic Model**: Open `base.ipynb` and run all cells
2. **Advanced Model**: Open `advanced.ipynb` and run all cells
3. **Fine-tuned Basic Model**: Open `base_fine_tuned.ipynb` and run all cells

**Note**: The Jupyter notebooks contain the complete analysis with outputs, visualizations, and detailed explanations. They are the main demonstration files for this project.

## 🔧 Key Preprocessing Techniques Demonstrated

### 1. Missing Value Imputation
- **KNN Imputation**: For numeric features using 5 nearest neighbors
- **Mode Imputation**: For categorical features
- **Domain-specific imputation**: Living space ratios based on neighborhood

### 2. Feature Engineering
- **Living Efficiency**: `life_sq / full_sq` (living space ratio)
- **Average Room Size**: `full_sq / num_room` (luxury indicator)
- **Floor Ratio**: `floor / max_floor` (floor desirability)
- **Log Transformations**: `log1p(full_sq)` for skewed features
- **Amenity Score**: Combined accessibility to schools, parks, metro, etc.

### 3. Categorical Encoding
- **Frequency Encoding**: For high-cardinality features (sub_area)
- **One-Hot Encoding**: For low-cardinality features
- **Ordinal Encoding**: For medium-cardinality features

### 4. Data Cleaning
- **Outlier Removal**: Based on domain knowledge
- **Duplicate Elimination**: Remove exact duplicates
- **Data Type Conversion**: Proper numeric types

### 5. Hyperparameter Optimization
- **Grid Search**: Systematic exploration of Ridge alpha values
- **Cross-validation**: Ensures robust parameter selection
- **Performance comparison**: Demonstrates limits of tuning alone

## 💡 Key Takeaways

1. **Advanced preprocessing outperforms hyperparameter tuning**: 22.7% better performance than fine-tuning alone
2. **Clean data trumps complex algorithms**: 27.30% improvement through preprocessing alone
3. **Domain knowledge guides feature engineering**: Understanding housing markets led to effective feature creation
4. **Pipeline design matters**: Proper architecture prevents common pitfalls like data leakage
5. **Categorical data is valuable**: Smart encoding preserves important information
6. **ROI is immediate**: High-impact improvements with reasonable time investment
7. **Preprocessing ROI is exceptional**: Time investment vs. performance gain ratio is very high

**Note**: The Jupyter notebooks contain the complete analysis with outputs, visualizations, and detailed explanations. They are the primary files for understanding and running the demonstration.

## 🤝 Contributing

This is a demonstration project.

---

**Presented by**: This case study was presented by GroupLabs at the "AI-Ready or AI-Hopeful?" workshop for DAMA, demonstrating practical approaches to improving AI project success rates through systematic data quality practices. 