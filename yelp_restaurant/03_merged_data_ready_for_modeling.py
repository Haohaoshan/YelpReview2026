
#%% 
import pandas as pd
import numpy as np 

#%% 
# Load feature and review data
X_features_5pct = pd.read_parquet("02_features_df_5pct.parquet")
reviews = pd.read_parquet("01_df_review_business_user.parquet")
print(X_features_5pct.head())


#%% 
# Filter reviews to the same 5% sample
print(reviews.shape)   #5_245_324
reviews_5pct = reviews.loc[reviews["rand"] < 0.05].copy()
print(reviews_5pct.shape)   #5% ==>261_882



#%% 
#Merge features with engagement fields
X_features_5pct_add = X_features_5pct.merge(
    reviews_5pct[["review_id", "date", "year", 'name', 'address', 'city', 'state', "stars_review",                   
                  "useful", "funny", "cool"]],
    on="review_id",
    how="left"
)

print(X_features_5pct_add)
print(X_features_5pct_add.head())


#%% 
# df must contain: useful, funny, cool

def define_engagement_label(df):
    # High-engagement
    high = (
        (df["useful"] >= 2) |
        (df["funny"] >= 1) |
        (df["cool"] >= 1)
    )

    # Low-engagement
    low = (
        (df["useful"] == 0) &
        (df["funny"] == 0) &
        (df["cool"] == 0)
    )

    # Assign labels:
    # 0 = low
    # 1 = intermediate
    # 2 = high
    df["engagement_label"] = np.where(
        high, 
        2,
        np.where(
            low,
            0,
            1   # intermediate
        )
    )

    return df


df = define_engagement_label(X_features_5pct_add)

print(df['engagement_label'].value_counts(dropna=False)) 

# %%
df.to_parquet("03_merged_data.parquet")
# %%

print(df.columns.tolist())
# %%
['review_id', 
 'token_count', 'unique_tokens', 'unique_lemmas', 'type_lemma_ratio', 'avg_word_len', 'hapax_ratio', 'noun_count', 
 'verb_count', 'adj_count', 'adv_count', 'avg_dep_depth', 'flesch_reading_ease', 'smog_index', 'ari',
 'coleman_liau', 'dale_chall', 'sentiment_proxy', 'sent_count', 'avg_sent_len', 'char_count', 'punctuation_count',
 'burstiness', 'concreteness_proxy', 'action_ratio',
 'date', 'year', 'name', 'address', 'city', 'state', 'stars_review', 'useful', 'funny', 'cool', 'engagement_label']


#%% 
"""======= derive more features by normalizing some count features by token count ===========""" 

df_old = pd.read_parquet("03_merged_data.parquet") 
print(df_old['token_count'].isna().sum(), df_old['token_count'].min(),df_old['token_count'].max(),df_old['token_count'].median()) 

#%%
print(df_old.loc[df_old['token_count'] == 0].head())


#%% ------------ didnt not pursue since we already have concreteness and action ration features 
df_new = df_old.copy()
df_new["noun_ratio"] = df_old["noun_count"] / df_old["token_count"].replace(0, np.nan)
df_new["verb_ratio"] = df_old["verb_count"] / df_old["token_count"].replace(0, np.nan)
df_new["adj_ratio"]  = df_old["adj_count"]  / df_old["token_count"].replace(0, np.nan)
df_new["adv_ratio"]  = df_old["adv_count"]  / df_old["token_count"].replace(0, np.nan)
 
_