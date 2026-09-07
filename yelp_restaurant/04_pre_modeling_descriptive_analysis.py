

#%% 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) 
print("=== Script Directory ===")
print(script_dir) 

#%% 
Xy_5pct = pd.read_parquet(os.path.join(script_dir, "03_merged_data.parquet")) 
Xy_5pct = Xy_5pct.loc[Xy_5pct['engagement_label'].isin([0,2])] #remove rows with engagement_label 1 (neutral) 
Xy_5pct["engagement_binary"] = (Xy_5pct["engagement_label"] == 2).astype(int) 
print("=== Engagement Label Distribution ===")
print(Xy_5pct["engagement_binary"].value_counts(dropna=False))

#%% 
feature_cols = ['token_count', 'unique_tokens', 'unique_lemmas', 'type_lemma_ratio', 'avg_word_len', 'hapax_ratio', 'noun_count', 
 'verb_count', 'adj_count', 'adv_count', 'avg_dep_depth', 'flesch_reading_ease', 'smog_index', 'ari',
 'coleman_liau', 'dale_chall', 'sentiment_proxy', 'sent_count', 'avg_sent_len', 'char_count', 'punctuation_count',
 'burstiness', 'concreteness_proxy', 'action_ratio']

df = Xy_5pct.loc[:, feature_cols + ["engagement_binary"]]


#%% 
# ---------------------------------------------------------
# 1. Summary Statistics: High vs Low Engagement
# ---------------------------------------------------------

summary = df.groupby("engagement_binary")[feature_cols].mean().T
summary.columns = ["low_engagement_mean", "high_engagement_mean"]
summary["difference"] = summary["high_engagement_mean"] - summary["low_engagement_mean"]

print("\n=== Mean Differences (High - Low Engagement) ===")
print(summary.sort_values("difference", ascending=False))

#%%
summary.to_excel(os.path.join(script_dir, "04_summary_statistics_high_vs_low_engagement.xlsx"), index=True)

#%% 
# ---------------------------------------------------------
# 2. Statistical Significance (t-tests)
# ---------------------------------------------------------

ttest_results = []

for feature in feature_cols:
    high = df[df["engagement_binary"] == 1][feature]
    low  = df[df["engagement_binary"] == 0][feature]
    t, p = ttest_ind(high, low, equal_var=False)
    ttest_results.append((feature, t, p))

ttest_df = pd.DataFrame(ttest_results, columns=["feature", "t_stat", "p_value"])
print("\n=== T-test Results ===")
print(ttest_df.sort_values("p_value"))

# ---------------------------------------------------------
# 3. Visual Comparison (Boxplots for ALL features)
# ---------------------------------------------------------

sns.set(style="whitegrid")

for feature in feature_cols:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df["engagement_binary"], y=df[feature])
    plt.title(f"{feature}: High vs Low Engagement")
    plt.xlabel("High Engagement (1 = Yes, 0 = No)")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 4. Top Differences (Largest absolute differences)
# ---------------------------------------------------------

top_differences = summary["difference"].abs().sort_values(ascending=False).head(10)
print("\n=== Top 10 Features with Largest Differences ===")
print(top_differences)

# %%
""" 
=== Top 10 Features with Largest Differences ===
char_count           311.726219
token_count           57.384553
lemma_count           57.384553
unique_lemmas         28.958335
noun_count            11.977689
punctuation_count      8.156460
verb_count             7.349504
adj_count              4.963577
adv_count              3.993402
sent_count             3.662461
Name: difference, dtype: float64
""" 