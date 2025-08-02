# WEEK 2

This repository contains 4 advanced-level tasks completed as part of a assignment. The tasks involve deep dives into neural networks, NLP, high-performance analytics, and statistical modeling. All code is modular, visualized, and well-documented.

---

## Task List

### **Task 1: From-Scratch Neural Network Regression (NumPy Only)**  
**Goal:** Build a fully connected neural network for a regression task from scratch using only NumPy.  
- Implements: Dense layers, ReLU, MSE Loss, SGD optimizer  
- Dataset: Synthetic cubic function  
- Architecture: 1 → 64 → ReLU → 64 → ReLU → 1  
- Result: Achieved R² ≈ 0.99  


---

### **Task 2: NLTK-Powered Text Analytics Web App**  
**Goal:** Perform interactive text analysis using NLTK via a Streamlit app.  
- Features: Tokenization, Lemmatization, POS Tagging, NER, Frequency plots  
- Input: Any raw text via frontend   
- Run via: `streamlit run streamlit_app.py`  
 

---

### **Task 3: High-Performance Time Series Transformation**  
**Goal:** Benchmark time-series operations using pandas vs NumPy on 1M+ rows.  
- Operations: Rolling mean, variance, covariance, EWMA, FFT filters  
- Visuals: Runtime & memory usage plots  
- Code: `task3/benchmark.py`, `timeseries.py`, `synthetic_data_gen.py`  
- Output: CSV + PNG charts  
- Highlights: NumPy significantly faster for large-scale ops

---

### **Task 4: Complex Data Munging & Statistical Modeling (Customer Churn)**  
**Goal:** Clean real-world churn data and build a logistic regression model.  
- Techniques: Outlier removal, imputation, polynomial features  
- Model: Logistic Regression + R² & accuracy metrics  
- Code: `task4/data_prep.py`, `modeling.py`  
- Result: ~82% accuracy  
- Data: Customer subscription dataset

---

## Setup Instructions

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

Dependencies installation in the individual folders

