
#%% 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% --- Build merged importance table ---
logit_df = pd.read_parquet("04_logreg_coeff.parquet")
logit_df['logit_importance']= abs(logit_df['coef'])
print(logit_df.head())

#%% 
xgb_df=pd.read_parquet("04_xgb_shap_importance.parquet")
xgb_df = xgb_df.rename(columns={"shap_importance": "xgb_shap_importance"})
print(xgb_df.head())

#%% 
merged = (
    logit_df
    .merge(xgb_df, on="feature")
    .sort_values("logit_importance", ascending=False)
)
print(merged.head())

#%% issue - both importances's scales are not comparable 
# --- Melt for plotting ---
plot_df = merged.melt(
    id_vars="feature",
    value_vars=["logit_importance", "xgb_shap_importance"],
    var_name="model",
    value_name="importance"
)

# --- Plot ---
plt.figure(figsize=(12, 10))
sns.barplot(
    data=plot_df,
    x="importance",
    y="feature",
    hue="model",
    palette="viridis"
)

plt.title("Logistic Regression vs XGBoost SHAP Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.legend(title="Model")
plt.tight_layout()
plt.show()



# %%
from scipy.stats import rankdata

# Rank logistic regression importance (higher importance → rank 1)
merged["rank_logit"] = rankdata(-merged["logit_importance"], method="dense")

# Rank XGB SHAP importance (higher importance → rank 1)
merged["rank_xgb"] = rankdata(-merged["xgb_shap_importance"], method="dense")
#%% 
print(merged.head())

#%% 
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

plt.scatter(merged["rank_logit"], merged["rank_xgb"], s=120)

for _, row in merged.iterrows():
    plt.text(row["rank_logit"]+0.1, row["rank_xgb"]+0.1, row["feature"], fontsize=9)

plt.xlabel("Logistic Regression Rank")
plt.ylabel("XGBoost SHAP Rank")
plt.title("Rank Comparison: Logit vs XGB SHAP")
plt.grid(True)
plt.show()

# %%
