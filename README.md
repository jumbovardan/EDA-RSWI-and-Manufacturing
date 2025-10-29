EDA on RSWI & Manufacturing Datasets

Project Overview

This project focuses on Exploratory Data Analysis (EDA) of two datasets — Retail Website and Manufacturing Efficiency.
The primary objective is to uncover hidden patterns, analyze relationships between production metrics, and understand operational efficiency through statistical and visual analysis.

 1️⃣ Dataset 1: Retail Website

 Analytical Focus

Handling missing values in workInProgress using imputation and filtering.
Understanding production flow between units.
Detecting units with high defect rates.
Correlation between machine efficiency and production output.

 Techniques Used

Data Cleaning: Removing nulls and invalid rows (df[df['workInProgress'].notnull()]).
Imputation: Filling missing numerical values with mean/median.
Visualization: Heatmaps, pairplots, and distribution plots to show trends in production.
Encoding: Label Encoding for categorical variables (LabelEncoder).

2️⃣ Dataset 2: Manufacturing Efficiency Dataset

 Description

This dataset captures department-wise performance metrics across a manufacturing setup, such as Stitching and Finishing units.

 Key Columns
Column	Description
productionDept	Department name (e.g., Stitching, Finishing).
efficiencyScore	Numerical metric reflecting departmental efficiency.
sessionID	Unique identifier for a production session.
operatorPerformance	Average efficiency of the operator in a given session.
machineRuntime	Duration for which machines were operational.

 Analytical Focus

Comparing department-wise efficiency scores.
Identifying trends in operational performance.
Detecting underperforming units.
Understanding how machine runtime affects efficiency.

 Visualizations

Bar Charts: Efficiency comparison between Stitching and Finishing units.
Box Plots: Distribution of efficiency scores per department.
Correlation Heatmaps: To detect interdependencies between variables.

Example:
sns.barplot(x='productionDept', y='efficiencyScore', data=df)
plt.title('Efficiency by Department')
plt.show()

 Preprocessing Techniques

Missing Value Handling

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna('Unknown')


Label Encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))


Filtering

df = df[df['workInProgress'].notnull()]

 Modeling Experiments

Basic regression and tree-based models were applied to predict efficiency or output rate:
Random Forest Regressor
GridSearchCV for hyperparameter tuning

Example snippet:

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

params = {'n_estimators': [100, 200],
          'max_depth': [5, 10, None],
          'max_features': ['sqrt', 'log2']}

grid = GridSearchCV(RandomForestRegressor(), params, cv=3, scoring='r2')
grid.fit(X, y)
print("Best Parameters:", grid.best_params_)

 Key Insights

Missing values in workInProgress had to be imputed to maintain data consistency.
Efficiency in Finishing Units tends to be slightly higher due to better machine utilization.
Random Forest Regression performed best for predicting efficiency scores.
Gradient Boosting gave interpretable feature importance scores, helping identify top factors influencing productivity.

 Tools & Libraries

Python, Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn (for modeling, encoding, and tuning)
Jupyter Notebook for development and visualization

 Learning Outcomes

Hands-on experience with real-world industrial data cleaning and EDA.
Improved understanding of model hyperparameter tuning and cross-validation.
Practical exposure to identifying efficiency bottlenecks in manufacturing setups.
