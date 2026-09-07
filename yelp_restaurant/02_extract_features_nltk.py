
#%% 
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure required NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

#%% 
# ---------------------------------------------------------
# 1. TEXT PREPROCESSING
# ---------------------------------------------------------

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


# ---------------------------------------------------------
# 2. SENTIMENT FEATURES
# ---------------------------------------------------------

def sentiment_features(text):
    # TextBlob sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # VADER sentiment
    vader_scores = sia.polarity_scores(text)
    vader_pos = vader_scores['pos']
    vader_neg = vader_scores['neg']
    vader_neu = vader_scores['neu']
    vader_compound = vader_scores['compound']

    # Extreme sentiment words
    extreme_words = ["amazing", "terrible", "awful", "excellent", "horrible",
                     "best", "worst", "perfect", "disgusting"]
    extreme_count = sum(text.lower().count(w) for w in extreme_words)

    return {
        "sent_polarity": polarity,
        "sent_subjectivity": subjectivity,
        "vader_pos": vader_pos,
        "vader_neg": vader_neg,
        "vader_neu": vader_neu,
        "vader_compound": vader_compound,
        "extreme_sentiment_words": extreme_count
    }


def enhanced_sentiment_features(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    vader = sia.polarity_scores(text)

    # Sentence-level sentiment variance
    sentences = sent_tokenize(text)
    sentence_polarities = [TextBlob(s).sentiment.polarity for s in sentences]
    sent_var = np.var(sentence_polarities) if len(sentence_polarities) > 1 else 0

    # Sentiment contrast
    sentiment_contrast = vader['pos'] - vader['neg']

    # Negation handling
    negations = ["not", "never", "no", "n't"]
    negation_count = sum(text.lower().count(n) for n in negations)

    return {
        "sent_polarity": polarity,
        "sent_subjectivity": subjectivity,
        "vader_pos": vader['pos'],
        "vader_neg": vader['neg'],
        "vader_neu": vader['neu'],
        "vader_compound": vader['compound'],
        "sentiment_variance": sent_var,
        "sentiment_contrast": sentiment_contrast,
        "negation_count": negation_count
    }



# ---------------------------------------------------------
# 3. LEXICAL FEATURES
# ---------------------------------------------------------

def lexical_features(tokens):
    total_words = len(tokens)
    unique_words = len(set(tokens))
    lexical_diversity = unique_words / total_words if total_words > 0 else 0

    return {
        "word_count": total_words,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity
    }

def enhanced_lexical_features(tokens):
    total = len(tokens)
    unique = len(set(tokens))

    hapax = sum(1 for t in set(tokens) if tokens.count(t) == 1)
    avg_word_len = np.mean([len(t) for t in tokens]) if total > 0 else 0
    long_word_ratio = sum(1 for t in tokens if len(t) >= 7) / total if total > 0 else 0

    function_words = set(stopwords.words('english'))
    func_ratio = sum(1 for t in tokens if t in function_words) / total if total > 0 else 0

    return {
        "word_count": total,
        "unique_words": unique,
        "lexical_diversity": unique / total if total > 0 else 0,
        "hapax_legomena": hapax,
        "avg_word_length": avg_word_len,
        "long_word_ratio": long_word_ratio,
        "function_word_ratio": func_ratio
    }


# ---------------------------------------------------------
# 4. SYNTACTIC FEATURES (POS TAGGING)
# ---------------------------------------------------------

nltk.download('averaged_perceptron_tagger')

def syntactic_features(tokens):
    if len(tokens) == 0:
        return {
            "noun_prop": 0,
            "verb_prop": 0,
            "adj_prop": 0,
            "adv_prop": 0,
            "pron_prop": 0
        }

    pos_tags = nltk.pos_tag(tokens)
    total = len(pos_tags)

    noun_prop = sum(1 for w, t in pos_tags if t.startswith("NN")) / total
    verb_prop = sum(1 for w, t in pos_tags if t.startswith("VB")) / total
    adj_prop  = sum(1 for w, t in pos_tags if t.startswith("JJ")) / total
    adv_prop  = sum(1 for w, t in pos_tags if t.startswith("RB")) / total
    pron_prop = sum(1 for w, t in pos_tags if t in ["PRP", "PRP$", "WP", "WP$"]) / total

    return {
        "noun_prop": noun_prop,
        "verb_prop": verb_prop,
        "adj_prop": adj_prop,
        "adv_prop": adv_prop,
        "pron_prop": pron_prop
    }

def enhanced_syntactic_features(tokens, raw_text):
    if len(tokens) == 0:
        return { "noun_prop":0, "verb_prop":0, "adj_prop":0, "adv_prop":0,
                 "pron_prop":0, "sentence_length_var":0, "clause_density":0,
                 "passive_voice":0, "pos_entropy":0 }

    pos_tags = nltk.pos_tag(tokens)
    total = len(pos_tags)

    noun_prop = sum(1 for _, t in pos_tags if t.startswith("NN")) / total
    verb_prop = sum(1 for _, t in pos_tags if t.startswith("VB")) / total
    adj_prop  = sum(1 for _, t in pos_tags if t.startswith("JJ")) / total
    adv_prop  = sum(1 for _, t in pos_tags if t.startswith("RB")) / total
    pron_prop = sum(1 for _, t in pos_tags if t in ["PRP","PRP$","WP","WP$"]) / total

    # Sentence length variance
    sentences = sent_tokenize(raw_text)
    sent_lengths = [len(word_tokenize(s)) for s in sentences]
    sentence_length_var = np.var(sent_lengths) if len(sent_lengths) > 1 else 0

    # Clause density (approx)
    clause_markers = ["and", "but", "because", "which", "that"]
    clause_density = sum(raw_text.lower().count(c) for c in clause_markers)

    # Passive voice detection
    passive_voice = sum(1 for w, t in pos_tags if t == "VBN")  # past participle

    # POS entropy
    from collections import Counter
    tag_counts = Counter([t for _, t in pos_tags])
    probs = np.array(list(tag_counts.values())) / total
    pos_entropy = -np.sum(probs * np.log(probs))

    return {
        "noun_prop": noun_prop,
        "verb_prop": verb_prop,
        "adj_prop": adj_prop,
        "adv_prop": adv_prop,
        "pron_prop": pron_prop,
        "sentence_length_var": sentence_length_var,
        "clause_density": clause_density,
        "passive_voice": passive_voice,
        "pos_entropy": pos_entropy
    }

# ---------------------------------------------------------
# 5. READABILITY FEATURES
# ---------------------------------------------------------

def readability_features(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    num_sentences = len(sentences)
    num_words = len(words)

    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    syllable_count = sum(count_syllables(w) for w in words)
    syllables_per_word = syllable_count / num_words if num_words > 0 else 0

    flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * syllables_per_word

    punctuation_density = len(re.findall(r'[.!?,;:]', text)) / num_words if num_words > 0 else 0

    return {
        "avg_sentence_length": avg_sentence_length,
        "syllables_per_word": syllables_per_word,
        "flesch_reading_ease": flesch_score,
        "punctuation_density": punctuation_density
    }


def enhanced_readability_features(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    num_sent = len(sentences)
    num_words = len(words)

    avg_sentence_length = num_words / num_sent if num_sent > 0 else 0
    syllable_count = sum(count_syllables(w) for w in words)
    syllables_per_word = syllable_count / num_words if num_words > 0 else 0

    flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * syllables_per_word

    # Complex words (3+ syllables)
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    complex_word_ratio = complex_words / num_words if num_words > 0 else 0

    # Gunning Fog
    fog = 0.4 * (avg_sentence_length + 100 * complex_word_ratio)

    # SMOG
    smog = 1.043 * np.sqrt(complex_words * (30 / num_sent)) + 3.129 if num_sent > 0 else 0

    return {
        "avg_sentence_length": avg_sentence_length,
        "syllables_per_word": syllables_per_word,
        "flesch_reading_ease": flesch,
        "complex_word_ratio": complex_word_ratio,
        "gunning_fog": fog,
        "smog_index": smog
    }


def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_char_was_vowel = False

    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False

    if word.endswith("e"):
        count = max(1, count - 1)

    return max(1, count)


# ---------------------------------------------------------
# 6. MAIN FEATURE EXTRACTION PIPELINE
# ---------------------------------------------------------

def extract_features(df, text_col="text"):
    processed = df[text_col].apply(preprocess_text)
    tokens = processed.apply(lambda x: x.split())

    feature_rows = []

    for raw_text, tok_list in zip(df[text_col], tokens):
        feats = {}
        feats.update(sentiment_features(raw_text))
        feats.update(lexical_features(tok_list))
        feats.update(syntactic_features(tok_list))
        feats.update(readability_features(raw_text))
        feature_rows.append(feats)

    feature_df = pd.DataFrame(feature_rows)
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)





#%% 
"""=================================== test feature derivation functions on a small sample of reviews ============================= """
review_all = pd.read_parquet("01_df_review_business_user.parquet")

review_smpl = review_all[['review_id','user_id','business_id','text','rand']] \
    .loc[review_all['rand'] < 0.001] \
    .copy()
print(review_smpl.shape)

print(review_smpl.shape)  # 5336 reviews 
# %%
features_smpl = extract_features(review_smpl, text_col="text") 

# %%
