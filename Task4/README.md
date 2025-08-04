Task 4: Complex Data Munging & Statistical Modeling 

Objective 

This task focuses on real-world data preprocessing, feature engineering, and statistical modeling using Python libraries like pandas, numpy, and statsmodels. The goal is to predict customer churn based on multiple features after thorough cleaning and transformation.

---

Files Included 

- 'task4_data_prep.ipynb': Missing value imputation, outlier removal, and encoding 
- 'task4_modeling.ipynb': Fitting and interpreting logistic regression 
- 'churn_dataset.csv': Original dataset  
- 'Task 4 Complex data munging and statistical modeling in pandas.pptx': Slide deck summarizing the task 

---

Steps Performed 
1. Data Cleaning 
- Filled missing values using mean/median
- Removed outliers using IQR method
- Encoded categorical features

2. Feature Engineering 
- Created polynomial interaction terms
- Binned 'Total Spend' into categories 
- Normalized numerical features

3. Modeling 
- Built logistic regression using 'statsmodels' 
- Analyzed p-values and confidence intervals
- Evaluated accuracy (~82%) and feature importance

---

Model Summary 

- Model Type: Logistic Regression
- Accuracy: 82%
- Key Influencers: Tenure, Gender, Usage Frequency, Spend Category

---

Learnings 

> Learned how to prepare real-life messy data for modeling, engineer useful features, and build interpretable models using Python. Also understood how each variable impacts customer churn.

---

How to Run 

1. Install dependencies  
 
   pip install pandas numpy statsmodels matplotlib

   Copy the path of churn_dataset given in the directory into the task4_data_prep.ipynb file and run it.

   After running tesk4_data_prep.ipynb file, run task4_modeling.ipynb file.
