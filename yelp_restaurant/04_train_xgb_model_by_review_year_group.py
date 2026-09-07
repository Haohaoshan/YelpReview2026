

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
import joblib 

#%% 
# ---------------------------------------------------------
# 1. Train/Test Split
# ---------------------------------------------------------
Xy_5pct = pd.read_parquet("03_merged_data.parquet") 

#%%
col_to_view = ['review_id', 'date', 'year', 'name', 'address', 'city', 'state', 'stars_review', 'useful', 'funny', 'cool', 'engagement_label']
print(Xy_5pct[col_to_view].head(10))   
print(Xy_5pct['year'].value_counts(dropna=False).sort_index()) 
print(pd.crosstab(Xy_5pct['year'], Xy_5pct['engagement_label'], margins=True, normalize='index').round(3))
#%% 

Xy_5pct = Xy_5pct.loc[Xy_5pct['engagement_label'].isin([0,2])]
Xy_5pct["engagement_binary"] = (Xy_5pct["engagement_label"] == 2).astype(int) 

Xy_5pct['review_year_grp'] = np.select(
    [
        Xy_5pct['year'] < 2016,
        Xy_5pct['year'].isin([2016, 2017, 2018]),
        Xy_5pct['year'] >= 2019
    ],
    [
        'group1_up_to_2015',
        'group2_2016_2018',
        'group3_2019_2022'
    ],
    default='group4_other'
)

print(Xy_5pct['review_year_grp'].value_counts(dropna=False).sort_index()) 
print(pd.crosstab(Xy_5pct['review_year_grp'], Xy_5pct['engagement_binary'], margins=True, normalize='index').round(3))

#%% 
# =========================================================
# SEGMENT-LEVEL MODELING BY review_year_grp
# =========================================================

feature_cols = ['token_count', 'unique_tokens', 'unique_lemmas', 'type_lemma_ratio', 'avg_word_len', 'hapax_ratio', 'noun_count', 
 'verb_count', 'adj_count', 'adv_count', 'avg_dep_depth', 'flesch_reading_ease', 'smog_index', 'ari',
 'coleman_liau', 'dale_chall', 'sentiment_proxy', 'sent_count', 'avg_sent_len', 'char_count', 'punctuation_count',
 'burstiness', 'concreteness_proxy', 'action_ratio']


segments = Xy_5pct['review_year_grp'].unique()

all_metrics = []

for grp in segments:
    print(f"\n=== Training model for segment: {grp} ===")

    df_seg = Xy_5pct[Xy_5pct['review_year_grp'] == grp].copy()
    print(f"df_seg.shape: {df_seg.shape}")  

    X_seg = df_seg.loc[:, feature_cols]
    y_seg = df_seg["engagement_binary"]

    # Train/test split WITHIN segment
    X_train, X_test, y_train, y_test = train_test_split(
        X_seg, y_seg, test_size=0.2, random_state=42, stratify=y_seg
    )

    # ---------------------------------------------------------
    # 1. Train XGBoost model for this segment
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

    # Save model
    joblib.dump(xgb_model, f"04_xgb_model_{grp}.joblib")

    # ---------------------------------------------------------
    # 2. Predictions + metrics
    # ---------------------------------------------------------
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    metrics = {
        "segment": grp,
        "accuracy": round(accuracy_score(y_test, y_pred), 5),
        "precision": round(precision_score(y_test, y_pred), 5),
        "recall": round(recall_score(y_test, y_pred), 5),
        "f1": round(f1_score(y_test, y_pred), 5),
        "auc": round(roc_auc_score(y_test, y_prob), 5)
    }

    all_metrics.append(metrics)
    print(metrics)

    # ---------------------------------------------------------
    # 3. SHAP importance for this segment
    # ---------------------------------------------------------
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # FIX: XGBClassifier returns list of arrays → pick class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_importance = np.abs(shap_values).mean(axis=0)

    df_shap_imp = pd.DataFrame({
        "feature": X_test.columns,
        "shap_importance": shap_importance
    }).sort_values("shap_importance", ascending=False)

    df_shap_imp.to_excel(f"04_xgb_shap_importance_{grp}.xlsx", index=False)
    df_shap_imp.to_parquet(f"04_xgb_shap_importance_{grp}.parquet")

    print(f"Saved SHAP importance for segment {grp}")

# Save all segment metrics
df_all_metrics = pd.DataFrame(all_metrics)
df_all_metrics.to_excel("04_xgb_segment_metrics.xlsx", index=False)
print("\n=== All segment-level metrics saved ===")




# %%
