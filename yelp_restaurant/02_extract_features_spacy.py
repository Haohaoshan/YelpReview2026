
#%% 
from scipy.__config__ import show
import spacy
print(spacy.info())

import en_core_web_sm
print(en_core_web_sm.__file__)

nlp = spacy.load("en_core_web_sm", disable=["ner"])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import re
import pandas as pd
import textstat
from collections import Counter

import time 

#%%
print(textstat.__version__) 

#%%
# run in shell: pip show vaderSentiment    




#%% 
# ---------------------------
# 0. Prepare input dataframe
# ---------------------------
review_all = pd.read_parquet("01_df_review_business_user.parquet")

review_5pct = review_all[['review_id','text','rand']] \
    .loc[review_all['rand'] < 0.05] \
    .copy()

review_5pct = review_5pct.drop(columns=["rand"])
print(review_5pct.shape)  # 261k reviews 
print(review_5pct.head())


#%%
review_tiny = review_all[['review_id','text','rand']] \
    .loc[review_all['rand'] < 0.0001] \
    .copy()
print(review_tiny.shape)
print(review_tiny.head())

#%% 
# -----------------------------
# 1. Preprocessing
# -----------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# -----------------------------
# 2. Feature Extraction
# -----------------------------
def extract_features(review_id, doc):
    tokens = [t for t in doc if t.is_alpha]
    lemmas = [t.lemma_ for t in tokens]

    # Lexical
    token_count = len(tokens)  #include duplicate. text length, verbosity 
    unique_tokens = len(set(t.text for t in tokens))  #vocabulary richness, surface vocabulary 
    unique_lemmas = len(set(lemmas))  #vocabulary richness   
    type_lemma_ratio = unique_lemmas / token_count if token_count > 0 else 0    #lemma diversity 
    avg_word_len = sum(len(t.text) for t in tokens) / token_count if token_count > 0 else 0    #lexical diversity 

    #hapax_ratio = sum(1 for l in set(lemmas) if lemmas.count(l) == 1) / unique_lemmas if unique_lemmas > 0 else 0   #rare lemma share  
    lemma_freq = Counter(lemmas)
    hapax_ratio = sum(1 for count in lemma_freq.values() if count == 1) / len(lemma_freq) if lemma_freq else 0

    # Syntactic
    noun_count = sum(1 for t in tokens if t.pos_ == "NOUN")   
    verb_count = sum(1 for t in tokens if t.pos_ == "VERB")
    adj_count = sum(1 for t in tokens if t.pos_ == "ADJ")
    adv_count = sum(1 for t in tokens if t.pos_ == "ADV")

    #avg_dep_depth = sum(len([a for a in t.ancestors]) for t in doc) / len(doc) if len(doc) > 0 else 0
    avg_dep_depth = (
        sum(sum(1 for _ in t.ancestors) for t in tokens) / len(tokens)
        if tokens else 0
    )

    # Readability (textstat)
   
    flesch_reading_ease = textstat.flesch_reading_ease(doc.text)
    smog_index = textstat.smog_index(doc.text)
    ari = textstat.automated_readability_index(doc.text)
    coleman_liau = textstat.coleman_liau_index(doc.text)
    dale_chall = textstat.dale_chall_readability_score(doc.text)

    # Sentiment
    # sentiment_proxy = textstat.textstat.polarity_scores(doc.text)["polarity"]

    # Sentiment (VADER)
    sentiment_proxy = analyzer.polarity_scores(doc.text)["compound"]

    # Structural
    sents = list(doc.sents) 
    sent_count = len(sents)
    avg_sent_len = token_count / sent_count if sent_count > 0 else 0
    char_count = len(doc.text)
    punctuation_count = sum(1 for t in doc if t.is_punct)
    burstiness = (
        pd.Series([len([t for t in s if t.is_alpha]) for s in sents]).std()
        if sent_count > 1 else 0
    )

    # Semantic
    concreteness_proxy = noun_count / token_count if token_count > 0 else 0
    action_ratio = verb_count / token_count if token_count > 0 else 0

    return {
        "review_id": review_id,
        "token_count": token_count,
        "unique_tokens": unique_tokens,
        "unique_lemmas": unique_lemmas,
        "type_lemma_ratio": type_lemma_ratio,
        "avg_word_len": avg_word_len,
        "hapax_ratio": hapax_ratio,
        "noun_count": noun_count,
        "verb_count": verb_count,
        "adj_count": adj_count,
        "adv_count": adv_count,
        "avg_dep_depth": avg_dep_depth,
        "flesch_reading_ease": flesch_reading_ease,
        "smog_index": smog_index,
        "ari": ari,
        "coleman_liau": coleman_liau,
        "dale_chall": dale_chall,
        "sentiment_proxy": sentiment_proxy,
        "sent_count": sent_count,
        "avg_sent_len": avg_sent_len,
        "char_count": char_count,
        "punctuation_count": punctuation_count,
        "burstiness": burstiness,
        "concreteness_proxy": concreteness_proxy,
        "action_ratio": action_ratio
    }

# -----------------------------
# 3. Apply to review_smpl
# -----------------------------
def process_reviews(df):
    feature_rows = []
    text_iter = (preprocess_text(text) for text in df["text"])

    for review_id, doc in zip(df["review_id"], nlp.pipe(text_iter, batch_size=200)):
        feats = extract_features(review_id, doc)
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)

#%% 
# -----------------------------
# 4. Run pipeline
# -----------------------------
start = time.time() 
features_df = process_reviews(review_5pct)
#features_df = process_reviews(review_tiny)
features_df.head()
print(features_df.shape)   

end = time.time()
print("time spent:", round((end-start)/60, 1), "minutes")

#260k reviews, took 31 mins. 

features_df.to_parquet(
    "02_features_df_5pct.parquet", 
    index=False,
    compression='snappy'
)  

#%% 
print(features_df.columns)
print(features_df.head())



#%%
lexical_features =['token_count', 'unique_tokens', 'unique_lemmas', 'type_lemma_ratio', 'avg_word_len', 'hapax_ratio']
syntactic_features=['noun_count', 'verb_count', 'adj_count', 'adv_count', 'avg_dep_depth']
readability_features=['flesch_reading_ease', 'smog_index', 'ari', 'coleman_liau', 'dale_chall']
sentiment_features=['sentiment_proxy'] 
structural_features=['sent_count', 'avg_sent_len', 'char_count', 'punctuation_count','burstiness']
semantic_features=['concreteness_proxy', 'action_ratio'] 

feature_definitions = {
    "lexical": {
        "token_count": "Number of alphabetic tokens in the text.",
        "unique_tokens": "Count of distinct surface word forms.",
        "unique_lemmas": "Count of distinct lemmas (vocabulary size).",
        "type_lemma_ratio": "Unique lemmas divided by total tokens; lemma-based diversity.",
        "avg_word_len": "Average number of characters per word.",
        "hapax_ratio": "Fraction of lemmas appearing exactly once."
    },

    "syntactic": {
        "noun_count": "Number of nouns; concreteness indicator.",
        "verb_count": "Number of verbs; action orientation.",
        "adj_count": "Number of adjectives; descriptiveness.",
        "adv_count": "Number of adverbs; modifier intensity.",
        "avg_dep_depth": "Average syntactic dependency chain length; complexity."
    },

    "readability": {
        "flesch_reading_ease": "Standard readability score; higher = easier.",
        "smog_index": "Readability index based on polysyllabic words.",
        "ari": "Automated Readability Index; character-based difficulty.",
        "coleman_liau": "Readability score using characters per word and words per sentence.",
        "dale_chall": "Readability score based on proportion of 'hard' words."
    },

    "sentiment": {
        "sentiment_proxy": "Heuristic polarity-based sentiment score."
    },

    "structural": {
        "sent_count": "Number of sentences in the text.",
        "avg_sent_len": "Average number of tokens per sentence.",
        "char_count": "Total number of characters.",
        "punctuation_count": "Number of punctuation marks.",
        "burstiness": "Variance of sentence lengths; rhythm/pacing."
    },

    "semantic": {
        "concreteness_proxy": "Nouns divided by tokens; concreteness indicator.",
        "action_ratio": "Verbs divided by tokens; action orientation."
    }
}

print(feature_definitions["lexical"]["token_count"]) 

# %%
