
# %%
import pandas as pd 
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

import xgboost as xgb



#%% prepare data  
Xy_5pct = pd.read_parquet("03_merged_data.parquet") 
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



#%% 
"""===================== xgb ========================="""
# Ensure you have your trained XGB model and logistic models, and identify the shared high-importance features from both models. 
# model_xgb = ...
# X_test = ...

#load the model
xgb_model = joblib.load("04_xgb_model.joblib")

# Probability-based performance metrics

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
xgb_metrics = compute_metrics(y_test, xgb_prob)
print("XGBoost test metrics:", xgb_metrics)



# Compute SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# List of shared high-importance features
# total 24 features, if a feature is ranked top 1/3 then its top by one model
# common features ranked as top 1/3 by both models are considered shared high-importance features 
top_shared_features = [
    "unique_lemmas",
    "punctuation_count",
    "char_count",
    "avg_dep_depth"
]

# Plot dependence for each feature
""" 
for feature in top_shared_features:
    shap.dependence_plot(
        feature,
        shap_values,
        X_test,
        interaction_index=None  # pure dependence, no interaction coloring
    )
""" 

# %% 2 by 2 grid plots for 4 common top features

import shap
import matplotlib.pyplot as plt

# Compute SHAP values for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)


# Create 2x2 grid
# Generate and save each plot
for feature in top_shared_features:
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

for ax, feature in zip(axes, top_shared_features):
    img = mpimg.imread(f"{feature}.png")
    ax.imshow(img)
    ax.axis("off")
    #no title. already exists
    #ax.set_title(f"SHAP Dependence: {feature}", fontsize=16)

plt.tight_layout()
plt.savefig("05_xgb_top_features_shap_dependence.png")





# %%
"""=================== logistic =========================="""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#load the model
log_clf = joblib.load("04_logit_model_log_clf.joblib") 

# Logistic Regression model: logit_model
# X_test: dataframe with feature columns
log_prob = log_clf.predict_proba(X_test)[:, 1]
log_metrics = compute_metrics(y_test, log_prob)
print("Logistic regression test metrics:", log_metrics)

# Extract LR coefficients into a dict
coef_map = dict(zip(X_test.columns, log_clf.coef_[0]))

# Compute LR contributions for each feature
lr_contrib = {}
for f in top_shared_features:
    beta = coef_map[f]
    lr_contrib[f] = X_test[f] * beta

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, feature in zip(axes, top_shared_features):
    ax.scatter(
        X_test[feature],
        lr_contrib[feature],
        alpha=0.4,
        s=10,
        color="steelblue"
    )
    ax.set_xlabel(feature, fontsize=14)
    ax.set_ylabel(f"LR contribution: β * {feature}", fontsize=14)
    ax.set_title(f"LR Dependence: {feature}", fontsize=16)

plt.tight_layout()
plt.savefig("05_LR_top_features_xbeta.png")




# %% 
""" create final table and plots for paper """
# concatenate performance metrics for both models
metrics_df = pd.DataFrame({
    "XGBoost": xgb_metrics,
    "Logistic Regression": log_metrics
}).T    
print(metrics_df) 

# output to excel file
metrics_df.to_excel("05_performance_metrics.xlsx") 




#%% 
"""================ do not use. column widths are not optimzed =========================""" 
 
# concatenate plots for both models into a single figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

n = len(top_shared_features)

# Create figure and GridSpec with a wider left column and very small gap
fig = plt.figure(figsize=(20, 4.5 * n))
gs = GridSpec(
    n, 2,
    figure=fig,
    width_ratios=[1.6, 0.9],
    wspace=0.0,
    hspace=0.35
)

for idx, feature in enumerate(top_shared_features):

    # Left panel: XGBoost SHAP dependence plot
    ax_left = fig.add_subplot(gs[idx, 0])
    xgb_img = mpimg.imread(f"{feature}.png")
    ax_left.imshow(xgb_img)
    ax_left.axis("off")
    ax_left.set_title(f"XGB SHAP dep: {feature}", fontsize=18, pad=8)

    # Right panel: Logistic regression contribution plot
    ax_right = fig.add_subplot(gs[idx, 1])
    beta = coef_map[feature]
    ax_right.scatter(
        X_test[feature],
        X_test[feature] * beta,
        alpha=0.45,
        s=12,
        color="steelblue"
    )
    ax_right.set_xlabel(feature, fontsize=15)
    ax_right.set_ylabel("LR contribution: xβ", fontsize=15)
    ax_right.set_title(f"Logistic regression: {feature}", fontsize=16)
    ax_right.tick_params(axis='both', labelsize=13)

plt.show()



