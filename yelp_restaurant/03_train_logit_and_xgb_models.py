
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 0. You already have:
#    X = features_df   (linguistic features)
#    y = target        (0/1 engagement label)
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import xgboost as xgb
import shap

# ---------------------------------------------------------
# 1. Train/Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. Standardize for Logistic Regression
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. Logistic Regression (interpretable baseline)
# ---------------------------------------------------------
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = logreg.predict(X_test_scaled)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression Performance ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 3))
print("Precision:", round(precision_score(y_test, y_pred_lr), 3))
print("Recall:", round(recall_score(y_test, y_pred_lr), 3))
print("F1:", round(f1_score(y_test, y_pred_lr), 3))
print("AUC:", round(roc_auc_score(y_test, y_prob_lr), 3))

# ---------------------------------------------------------
# 4. XGBoost (nonlinear model)
# ---------------------------------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n=== XGBoost Performance ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_xgb), 3))
print("Precision:", round(precision_score(y_test, y_pred_xgb), 3))
print("Recall:", round(recall_score(y_test, y_pred_xgb), 3))
print("F1:", round(f1_score(y_test, y_pred_xgb), 3))
print("AUC:", round(roc_auc_score(y_test, y_prob_xgb), 3))

# ---------------------------------------------------------
# 5. Logistic Regression Coefficients (interpretability)
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coef": logreg.coef_[0]
}).sort_values("coef", ascending=False)

print("\n=== Top Positive LR Coefficients ===")
print(coef_df.head(10))

print("\n=== Top Negative LR Coefficients ===")
print(coef_df.tail(10))

# ---------------------------------------------------------
# 6. XGBoost Feature Importance
# ---------------------------------------------------------
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== Top XGBoost Features ===")
print(importance_df.head(10))

# ---------------------------------------------------------
# 7. SHAP Analysis (global + local)
# ---------------------------------------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Global summary plot
shap.summary_plot(shap_values, X_test)

# Optional: dependence plot for a key feature
shap.dependence_plot("token_count", shap_values, X_test)

# Optional: local explanation for one review
idx = 0
shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx])
