🫀 Heart Disease Prediction Using Logistic Regression

🧠 Overview

This project uses supervised machine learning to predict whether a patient is likely to develop **heart disease** based on clinical and demographic features. The model is trained using **Logistic Regression**, and evaluated with **precision**, **recall**, and **confusion matrices**, prioritizing medical safety by minimizing false negatives.


📂 Dataset

* Source: Heart Disease UCI Dataset (Kaggle)
* Type: Tabular
* Target: `0` = No heart disease, `1` = Heart disease

Tools & Libraries

* Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
* Model: `LogisticRegression()`
* Scaler: `RobustScaler()` — handles outliers in medical data effectively

 📈 Data Preprocessing

* Missing values handled
* Categorical variables encoded
* Features scaled using `RobustScaler`
* Dataset split into training and testing sets

🔍 Model Training
python
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

📊 Evaluation Metrics

✅ Classification Report (Test Set):
Precision: 0.80 (class 1)
Recall:    0.90 (class 1)
F1-Score:  0.84 (class 1)
Accuracy:  82%
```

✅ Confusion Matrices:

Training Set

|                 | Predicted: NO | Predicted: YES |
| --------------- | ------------- | -------------- |
| **Actual: NO**  | 264           | 43             |
| **Actual: YES** | 46            | 335            |

**Test Set**

|                 | Predicted: NO | Predicted: YES |
| --------------- | ------------- | -------------- |
| **Actual: NO**  | 74            | 29             |
| **Actual: YES** | 13            | 114            |

✅ The model correctly identified **89.8%** of patients with heart disease in the test set.

---
🔁 Cross-Validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(log_model, X_train, y_train, cv=5, scoring='recall')
```

📊 Results:
Fold Recalls: [0.8816, 0.8947, 0.8052, 0.8289, 0.8947]
Average Recall: 86.1%


➡️ This shows the model generalizes well and is not overfitting.

🧠 Key Insights

* The model achieves **high recall**, which is crucial in healthcare — it's better to falsely flag a healthy patient than to miss a sick one.
* Logistic Regression, although simple, performs consistently and is interpretable.
* `RobustScaler` helped maintain stability in the presence of outliers.


📌 Conclusion

This project demonstrates that with clean data and careful evaluation, **even a baseline logistic regression model** can make meaningful predictions in life-impacting domains like healthcare.

✍️ Author

AYORINDE SAHEED OLAMILEKAN
Aspiring Data Scientist | Passionate about machine learning for health
