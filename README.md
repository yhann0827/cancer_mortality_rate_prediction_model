# Cancer Mortality Rate Prediction Model
**#MachineLearning #Python #Numpy #Scikitlearn #pandas #LinearRegression #Lasso #Ridge**
## Predicting Cancer Mortality Rates in US Counties

### Data Exploration and Preprocessing
- Read in the training data and targets files.
- Plot histograms of all features to visualize their distributions and identify outliers.
- Compute correlations of all features with the target variable and sort them according to the strength of correlations.
- **Top five features with strongest correlations to the targets:**
  1. PctBachDeg25_Over
  2. incidenceRate
  3. PctPublicCoverageAlone
  4. medIncome
  5. povertyPercent
- **Correlated Feature Sets:**
  - PctBachDeg25_ and medianIncome positively correlated.
  - PovertyPercent positively correlated with PctPublicCoverageAlone.

### Machine Learning Pipeline
- Create an ML pipeline using scikit-learn to pre-process the training data.
- Remove outliers from the MedianAge feature using the Interquartile range method (IQR).

### Regression Modeling
- Fit linear regression models to the pre-processed data using Ordinary least squares (OLS), Lasso, and Ridge models.
- Choose suitable regularization weights for Lasso and Ridge regression using k-fold cross-validation method.
- **Best Performing model:**
  - OLS model has the highest R-squared value.
  - Lasso model has the lowest root mean squared error (RMSE).
  - Ridge model is identified as the best performing and most consistent model across all the models.
