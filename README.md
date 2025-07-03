# Credit Card Fraud Detection

## 📌 Overview

### Objective  
The goal of this project is to develop a machine learning model capable of accurately detecting fraudulent credit card transactions using historical data. By analysing transaction patterns, the model aims to distinguish between normal and fraudulent activities, assisting financial institutions in flagging suspicious behaviour early and reducing potential risks.

### Challenges  
- Handling highly imbalanced data where fraud cases represent a small fraction of total transactions.  
- Maintaining high precision to minimise false positives (valid transactions wrongly flagged as fraud).  
- Achieving high recall to detect as many fraudulent transactions as possible.

---

## 🛠️ Workflow  
- Data handling and preprocessing  
- Exploratory Data Analysis (EDA) and visualisation  
- Model training, evaluation, and result interpretation  

---

## DataFrame Description  

| Feature | Description                                                                                   |
|---------|-----------------------------------------------------------------------------------------------|
| ⏳ Time   | Seconds elapsed since the first transaction in the dataset                                    |
| 🛡️ V1-V28 | Anonymised features engineered to protect sensitive original data                            |
| 💷 Amount | Transaction amount                                                                           |
| 🎯 Class  | Target variable (0 = normal transaction, 1 = fraudulent transaction)                          |

---

## 🔍 Analysing Class Distribution (Fraudulent vs Normal Transactions)

To better understand the dataset, we calculated the ratio of fraudulent to valid transactions. This involved counting the number of transactions in each class and computing their percentages relative to the total dataset size.

- **Total fraudulent transactions:** 492  
- **Percentage of fraudulent transactions:** 0.17%  
- **Total valid transactions:** 284,315  
- **Percentage of valid transactions:** 99.83%

---

### 📊 Distribution Observations

The results reveal a significant class imbalance — a key challenge in fraud detection.  
There is approximately **1 fraud for every 578 valid transactions**, making precision and recall critical for model success.

---

### 📈 Data Statistics

We now explore the transaction **amounts** within both fraudulent and valid classes.

<div style="display: flex; gap: 40px;">

<table>
  <caption><strong>💳 Valid Transactions</strong></caption>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Count</td><td>284,315</td></tr>
  <tr><td>Mean</td><td>88.29</td></tr>
  <tr><td>Std Dev</td><td>250.11</td></tr>
  <tr><td>Min</td><td>0.00</td></tr>
  <tr><td>25%</td><td>5.65</td></tr>
  <tr><td>50% (Median)</td><td>22.00</td></tr>
  <tr><td>75%</td><td>77.05</td></tr>
  <tr><td>Max</td><td>25,691.16</td></tr>
</table>

<table>
  <caption><strong>⚠️ Fraudulent Transactions</strong></caption>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Count</td><td>492</td></tr>
  <tr><td>Mean</td><td>122.21</td></tr>
  <tr><td>Std Dev</td><td>256.68</td></tr>
  <tr><td>Min</td><td>0.00</td></tr>
  <tr><td>25%</td><td>1.00</td></tr>
  <tr><td>50% (Median)</td><td>9.25</td></tr>
  <tr><td>75%</td><td>105.89</td></tr>
  <tr><td>Max</td><td>2,125.87</td></tr>
</table>

</div>

---

## 🧩 Correlation Matrix

![Image](https://github.com/user-attachments/assets/dbb1de27-0c6d-4c8e-8d6b-945c1ab90da0)

By examining the correlation matrix, we aim to identify any features that may be redundant or unlikely to contribute meaningfully to the analysis.  

The findings show that **most features in the dataset exhibit very weak linear correlations**, with coefficients clustering around ±0.00.  

🔍 **Conclusion:**  
There is no significant linear relationship between the features, which suggests low redundancy and no immediate need for feature removal based on correlation alone.

---

## 📊 Model Evaluation Metrics and Conclusions

Given the significant class imbalance in the dataset, relying on accuracy alone can be misleading. Therefore, a set of more robust metrics — Precision, Recall, F1-Score, and Matthews Correlation Coefficient — was used to evaluate the model’s true effectiveness. The model evaluated here is a **Random Forest Classifier**.

| Metric                            | Value     |
|----------------------------------|-----------|
| ✅ Accuracy                      | 0.9996    |
| 🎯 Precision                    | 0.9873    |
| 🔁 Recall                       | 0.7959    |
| ⚖️ F1-Score                     | 0.8814    |
| 🧮 Matthews Correlation Coefficient | 0.8863    |

### 🧠 Interpretation:

- **Accuracy (99.96%)**: High, but potentially misleading in imbalanced datasets. A model always predicting "not fraud" would also score highly.
- **Precision (98.73%)**: Excellent. When the model predicts fraud, it’s almost always correct — minimising false positives.
- **Recall (79.59%)**: Indicates the model detects nearly 80% of all actual fraud cases. Higher recall reduces missed frauds.
- **F1-Score (88.14%)**: Balances precision and recall well, showing overall model reliability.
- **Matthews Correlation Coefficient (0.8863)**: A robust metric for imbalanced binary classification. This strong score suggests balanced and accurate predictions across both classes.

📌 **Conclusion**:  
While high accuracy is encouraging, the combination of strong precision, recall, F1-Score, and MCC confirms that the model performs reliably and responsibly in detecting fraudulent transactions, even under heavy class imbalance.

---

## 🧪 Technology Used

This project was developed using the following Python libraries and tools:

- 📊 **Data manipulation and analysis**: `pandas`, `numpy`
- 📈 **Visualisation**: `matplotlib`, `seaborn`
- 🤖 **Machine Learning**: `scikit-learn`
  - `train_test_split` for splitting datasets  
  - `RandomForestClassifier` for modelling  
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `matthews_corrcoef`, `confusion_matrix` for evaluation  

---

📁 Project Structure
```bash
├── data/
│   └── global-data-on-sustainable-energy.csv
├── notebooks/
│   └── main.ipynb
├── images/
├── src/
│   └── modules
│       └── Modules.py
├── README.md
└── requirements.yml
```
---


## 🚀 Future Improvements

- 🧪 **Hyperparameter tuning** for improved model performance (e.g. GridSearchCV, RandomisedSearchCV)
- 📊 **Feature engineering** and dimensionality reduction (e.g. PCA)
- ⚖️ **Advanced resampling techniques** to better address class imbalance (SMOTE, ADASYN, etc.)
- 🔄 **Experiment with alternative models**, such as XGBoost, LightGBM, or deep learning architectures
- 🌐 **Deploy as a web service** using Flask or Streamlit for real-time fraud detection
- 📉 **Integrate cost-sensitive evaluation**, focusing on the real-world impact of false positives and false negatives

---

## 👨‍💻 Author


👨‍💻 Author
Developed with ❤️ by Alberto AJ, AI/ML Engineer 
📌 [Visit my GitHub](https://github.com/AlbertoGhub/AlbertoGhub) • [LinkedIn](https://www.linkedin.com/in/engineeralbertoac/)

---

