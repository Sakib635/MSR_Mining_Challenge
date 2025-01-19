# Maven Package Analysis

## Overview
This repository contains scripts and notebooks for analyzing Maven packages. The analysis includes:

1. Distribution of Maven packages across different ranges of star counts.
2. Correlation matrix of popularity metrics for Maven packages.
3. Comparison of features across the top and bottom 20% of packages, along with P-value and Cohen’s d for effect size.
4. Hierarchical clustering to handle multi-collinearity among features, with selected metrics listed.
5. Logistic regression analysis to generate final results.

## Prerequisites
- Python 3.8 or higher
- All required Python packages listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Distribution of Maven Packages by Star Count
To find the distribution of Maven packages across different ranges of star counts, run the following command:
```bash
python .\star_count_distribution.py
```
This will generate **Figure 1**.

### 2. Correlation Matrix of Popularity Metrics
To generate the correlation matrix for Maven package popularity metrics, run the following command:
```bash
python .\cluster_corelation.py
```
This will generate **Figure 2**.

### 3. Feature Comparison Across Top and Bottom 20%
To compare features across the top and bottom 20% of packages, including P-value and Cohen’s d for effect size, run the following command:
```bash
python .\minmaxmedian.py
```

### 4. Hierarchical Clustering for Feature Selection
To apply hierarchical clustering and handle multi-collinearity among features:
1. Open `hierarchical_clustering.ipynb` in a Jupyter Notebook environment.
2. Run all the cells in the notebook.

This step will identify the following metrics:
- License
- Commits Count
- Readme Exists
- About Info
- Dependencies
- Usages
- Closed Issues Percentage
- Release Frequency
- Vulnerabilities

**Figure 3** will also be generated during this process.

### 5. Logistic Regression Analysis
To perform logistic regression analysis and generate the final results, run the following command:
```bash
python .\Logistic_Regression.py
```

## Output
- **Figure 1**: Star count distribution.
![Image](https://github.com/user-attachments/assets/56025f63-acb2-4301-89ff-b3e091fb5f9d)
- **Figure 2**: Correlation matrix.
![Image](https://github.com/user-attachments/assets/fb9b3b19-8eb0-4939-b4ba-23c809c15be8)
- **Figure 3**: Hierarchical clustering visualization.
![Image](https://github.com/user-attachments/assets/1dfc5e21-6d84-4869-81d1-943ff3bccdda)
- Logistic regression results.

## Contact
For any questions or issues, please reach out to the repository maintainer.

