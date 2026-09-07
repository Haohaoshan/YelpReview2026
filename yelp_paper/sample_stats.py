
#%% 
import pandas as pd 

data_path= r"C:\Users\Wu\College_App_2028\SciFi_20260616\yelp\yelp_restaurant"
total_reviews = pd.read_parquet(data_path + r"\01_df_review_business_user.parquet")
sampled_reviews = pd.read_parquet(data_path + r"\02_features_df_5pct.parquet")
sampled_final = pd.read_parquet(data_path + r"\03_merged_data.parquet") 

print(len(total_reviews), len(sampled_reviews), len(sampled_final))  
#5_245_324 261_882 261_882

#%% 
print(sampled_final['engagement_label'].value_counts(dropna=False)) 

# %%
