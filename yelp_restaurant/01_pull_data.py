
""" 
https://business.yelp.com/data/resources/open-dataset/

https://data.clawrxiv.org/sources/yelp-open-dataset  


other links 
https://github.com/hemachandarn/Yelp-Dataset


""" 

#%% 
import pandas as pd 
import json
import os
print("gzip and json work!")

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "data", "Yelp JSON") 


#%% ============================step 1: load yelp files =========================
business_file = os.path.join(base_dir, "yelp_academic_dataset_business.json")
review_file   = os.path.join(base_dir, "yelp_academic_dataset_review.json")
user_file =  os.path.join(base_dir, "yelp_academic_dataset_user.json")
print(business_file)   # this prints the path only
print(review_file)
print(user_file)

#%% ====================== step 2: identify restarutants ========================== 

df_business = pd.read_json(business_file, lines=True)
print(df_business['city'].value_counts(dropna=False).sort_values(ascending=False).head(10))

print(df_business.columns.tolist)
print(df_business.shape)  #(150_346,14)
#%% 
""" 
<bound method IndexOpsMixin.tolist of Index(['business_id', 'name', 'address', 'city', 'state', 'postal_code',
       'latitude', 'longitude', 'stars', 'review_count', 'is_open',
       'attributes', 'categories', 'hours'],
      dtype='object')>
""" 

"""
restaurant_keywords = [
    "Restaurant", "Restaurants", "Food", "Cafe", "Cafes", "Coffee", "Tea",
    "Breakfast", "Brunch", "Bakeries", "Sandwiches", "Pizza", "Burgers",
    "BBQ", "Tex-Mex", "Seafood", "Steakhouses", "Fast Food", "Food Trucks",
    "Delis", "Diners", "Asian", "Chinese", "Japanese", "Korean", "Indian",
    "Mediterranean", "Middle Eastern", "Italian", "Mexican", "Thai",
    "Vietnamese", "Tapas", "Sushi"
]
""" 
#%% 
restaurant_keywords = [
    "Restaurant", "Restaurants", "Food", "Cafe", "Cafes", "Coffee", "Tea",
    "Breakfast", "Brunch", "Bakeries", "Sandwiches", "Pizza", "Burgers",
    "BBQ", "Tex-Mex", "Seafood", "Steakhouses", "Fast Food", "Food Trucks",
    "Delis", "Diners", "Tapas", "Sushi", "Bars", "Pubs",  
]

pattern = "|".join(restaurant_keywords)

restaurants = df_business[
    df_business['categories'].str.contains(pattern, case=False, na=False)
]
print(restaurants.shape) #67645,14) 

#%% 
print(restaurants['city'].value_counts(dropna=False)) 

col = ['name','stars','categories','review_count','is_open']
print(restaurants[col].head(10))

# print(restaurants['business_id'].duplicated().sum()) #0 dups 


#%% 
"""================= step 3: merge business and review files ======================"""

df_review = pd.read_json(review_file, lines=True)
print(df_review.columns.tolist())

df_review['year']=pd.to_datetime(df_review["date"]).dt.year
print(df_review['year'].value_counts(dropna=False).sort_index())
print(df_review.head())

#%% 
print(df_review.columns.tolist())
print(restaurants.columns.tolist())

"""
['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 
 'text', 'date', 'year']
['business_id', 'name', 'address', 'city', 'state', 'postal_code',
  'latitude', 'longitude', 'stars', 'review_count', 'is_open', 
  'attributes', 'categories', 'hours']
""" 

#%%
df_review = df_review.rename(columns={"stars":"stars_review"})
restaurants = restaurants.rename(columns={"stars":"stars_business"}) 

#%% 
df_review_business = df_review.merge(
    restaurants, on='business_id', how='inner')

print(df_review_business.shape)

print(df_review_business.columns.tolist())

#(6_990_280, 10), (5_245_324, 23)

#%% 
print(df_review_business['year'].value_counts().sort_index())


#%%
"""======================== step 4: append user info ========================="""
df_user = pd.read_json(user_file, lines=True)
print(df_user.columns.tolist())
#print(df_user['user_id'].duplicated().sum()) #0 dups
print(df_user[['name','useful','funny','cool','review_count']].head())

#%% 
#['user_id', 'name', 'review_count', 'yelping_since', 'useful', 'funny', 'cool', 'elite', 'friends', 
# 'fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
# 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 
# 'compliment_funny', 'compliment_writer', 'compliment_photos']

#%% 
df_user=df_user.rename(columns={"name":"name_user",
                                "review_count": "review_count_user",
                                "useful":"useful_user",
                                "funny":"funny_user",
                                "cool":"cool_user"})
 
df_review_business_user = (
    df_review_business.merge(
        df_user, 
        on='user_id',
        how='left',
        indicator=True
    )
)

print(df_review_business_user.shape)  #5_245_324, 45 
print(df_review_business_user.head()) 


#%%
print(df_review_business_user.columns.tolist())
#['review_id', 'user_id', 'business_id', 'stars_review', 'useful', 'funny', 'cool', 'text', 'date', 'year', 
# 'name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars_business', 
# 'review_count', 'is_open', 'attributes', 'categories', 'hours', 
# 'name_user', 'review_count_user', 'yelping_since', 'useful_user', 'funny_user', 'cool_user', 
# 'elite', 'friends', 'fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 
# 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 
# 'compliment_writer', 'compliment_photos', '_merge']

#%%
print(df_review_business_user[['review_id','date','name','review_count','name_user','review_count_user','yelping_since','elite','friends','fans']].head())
   

#%%  add a random number for future random sampling

import numpy as np
np.random.seed(42)   # set seed for reproducibility
df_review_business_user['rand'] = np.random.rand(len(df_review_business_user))


#%% 

df_review_business_user.to_parquet(
    "01_df_review_business_user.parquet", 
    index=False,
    compression='snappy'
) 

# %%
