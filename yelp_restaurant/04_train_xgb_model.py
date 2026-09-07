

#%% 
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 0. You already have:
#    X = features_df   (linguistic features)
#    y = target        (0/1 engagement label)
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import xgboost as xgb
import shap


#%% 
# ---------------------------------------------------------
# 1. Train/Test Split
# ---------------------------------------------------------

Xy_5pct = pd.read_parquet("03_merged_data.parquet") 
print(Xy_5pct.head()) 



#%%
Xy_5pct = Xy_5pct.loc[Xy_5pct['engagement_label'].isin([0,2])]
Xy_5pct["engagement_binary"] = (Xy_5pct["engagement_label"] == 2).astype(int) 

feature_cols = ['token_count', 'unique_tokens', 'unique_lemmas', 'type_lemma_ratio', 'avg_word_len', 'hapax_ratio', 'noun_count', 
 'verb_count', 'adj_count', 'adv_count', 'avg_dep_depth', 'flesch_reading_ease', 'smog_index', 'ari',
 'coleman_liau', 'dale_chall', 'sentiment_proxy', 'sent_count', 'avg_sent_len', 'char_count', 'punctuation_count',
 'burstiness', 'concreteness_proxy', 'action_ratio']


X = Xy_5pct.loc[:, feature_cols]
y = Xy_5pct["engagement_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# ---------------------------------------------------------
# 2. XGBoost (nonlinear model)
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


import joblib
joblib.dump(xgb_model, "04_xgb_model.joblib")

#load the model for later use 
#xgb_model = joblib.load("04_xgb_model.joblib")



# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n=== XGBoost Performance ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_xgb), 3))
print("Precision:", round(precision_score(y_test, y_pred_xgb), 3))
print("Recall:", round(recall_score(y_test, y_pred_xgb), 3))
print("F1:", round(f1_score(y_test, y_pred_xgb), 3))
print("AUC:", round(roc_auc_score(y_test, y_prob_xgb), 3))


df_metrics = pd.DataFrame({
    "model": ["XGBoost"],
    "accuracy": [round(accuracy_score(y_test, y_pred_xgb), 5)],
    "precision": [round(precision_score(y_test, y_pred_xgb), 5)],
    "recall": [round(recall_score(y_test, y_pred_xgb), 5)],
    "f1": [round(f1_score(y_test, y_pred_xgb), 5)],
    "auc": [round(roc_auc_score(y_test, y_prob_xgb), 5)]
})

df_metrics.to_excel("04_xgb_metrics.xlsx", index=False)
print("\n=== Metrics saved to 04_xgb_metrics.xlsx ===") 



#%% 
# ---------------------------------------------------------
# 3. XGBoost Feature Importance
# ---------------------------------------------------------
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== Top XGBoost Features ===")
print(importance_df.head(10))

#xgb's built-in tree-based importance (gain-based importance), 
# behaves differet from permulation importance and shap importance
#gain-based, computed during training. not after. purely tree split gain. 

#use gain importance only as a quick sanity check
#use shap imp as primary imp metric
#use permulation imp as a robust check 


#%% 
# ---------------------------------------------------------
# 4. SHAP Analysis (global + local)
# ---------------------------------------------------------
import shap
shap.initjs()

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Global summary plot
shap.summary_plot(shap_values, X_test)

#%% 
# shap_values is (n_samples, n_features)
shap_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "shap_importance": shap_importance
}).sort_values("shap_importance", ascending=False)
print(importance_df.head(10))
print(importance_df.shape)
importance_df.to_excel("04_xgb_shap_importance.xlsx")
importance_df.to_parquet("04_xgb_shap_importance.parquet")


#%%
# Optional: dependence plot for a key feature
shap.dependence_plot("token_count", shap_values, X_test)

# Optional: local explanation for one review
idx = 0
shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx])


# %% legacy shap dependence plot 
# highreadability increase engagement. 
shap.dependence_plot(
    "flesch_reading_ease",
    shap_values,
    X_test,
    feature_names=X_test.columns
)



# %% 
""" 
 ============================================================================
 do not run this part until we have trained both xgb and logit models 
 ============================================================================
"""

import shap
import matplotlib.pyplot as plt

# Ensure you have your trained XGB model and logistic models, and identify the shared high-importance features from both models. 
# model_xgb = ...
# X_test = ...

# Compute SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# List of shared high-importance features
top_shared_features = [
    "unique_lemmas",
    "punctuation_count",
    "char_count",
    "avg_dep_depth"
]

# Plot dependence for each feature
for feature in top_shared_features:
    shap.dependence_plot(
        feature,
        shap_values,
        X_test,
        interaction_index=None  # pure dependence, no interaction coloring
    )

# %% 2 by 2 grid plots for 4 common top features

import shap
import matplotlib.pyplot as plt

# Compute SHAP values for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Shared high-importance features
features = [
    "unique_lemmas",
    "punctuation_count",
    "char_count",
    "avg_dep_depth"
]


#%% 
# Create 2x2 grid
# Generate and save each plot
for feature in features:
    shap.dependence_plot(
        feature,
        shap_values,
        X_test,
        interaction_index=None,
        show=False
    )
    plt.title(f"SHAP Dependence: {feature}")
    plt.savefig(f"{feature}.png", dpi=200, bbox_inches="tight")
    plt.close()

# Load the saved images and place them into a 2×2 grid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, feature in zip(axes, features):
    img = mpimg.imread(f"{feature}.png")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"SHAP Dependence: {feature}", fontsize=12)

plt.tight_layout()
plt.show()



# %%
