---
implicit_figures: false
header-includes:
  - \usepackage[margin=1.1in]{geometry}
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{placeins}
  - \usepackage[table]{xcolor}
  - \definecolor{amber}{RGB}{255,191,0}
  - \definecolor{green}{RGB}{0,120,0}
  - \definecolor{blue}{RGB}{30,90,180}
---


\begin{titlepage}
\centering

{\Huge What Linguistic Features Differentiate High-Engagement and Low-Engagement Online Reviews? Evidence from Yelp Restaurant Reviews\par}

\vspace{1.5cm}

{\Large
Alexander Shan \\
Texas Academy of Mathematics and Science \\
\vspace{0.5cm}
Angela Yin \\
Reedy High School
}

\vspace{1.5cm}

{\large August 2026}

\vfill

{\small
Correspondence: \\
Alexander Shan - alsney2019@gmail.com \\
Angela Yin - angelasuyuyin@gmail.com 
}

\vspace{0.5cm}

\textit{Both authors contributed equally to this work.}

\end{titlepage}

\fontsize{13pt}{16pt}\selectfont


# Abstract 
Online reviews influence how people choose restaurants, hotels, and other services, yet the quality and usefulness of these reviews vary widely. This study investigates whether linguistic features can predict which Yelp restaurant reviews receive engagement from other users. Using a sample from the Yelp Open Dataset, we extracted features related to sentiment, vocabulary richness, readability, syntactic structure, and stylistic formatting. Logistic Regression and XGBoost models were trained to classify reviews as high-engagement or low-engagement based on crowd-sourced votes (useful, funny, cool). Both models achieved meaningful predictive performance, indicating that writing style contains reliable signals of perceived helpfulness. SHAP analysis revealed that lexical diversity, review length, syntactic complexity, and punctuation usage were consistently influential across models. A secondary analysis using hotel reviews showed similar patterns, supporting cross-domain generalizability. These results suggest that detailed, readable, and emotionally balanced writing is more likely to resonate with readers and provide value on online review platforms. 


# 1. Introduction

Online reviews shape everyday decisions, from choosing a restaurant to booking a hotel. Platforms like Yelp host millions of reviews, but only a fraction receive engagement from other users. Some reviews are validated through votes such as “useful,” “funny,” or “cool,” while many receive none. Understanding what makes certain reviews more helpful is important for consumers and platforms that rely on user-generated content.
Most prior work focuses on deception detection, but the Yelp Open Dataset does not include labels indicating whether a review is deceptive. Instead of attempting to infer deception without ground-truth labels, this study examines whether writing style can predict which reviews readers find helpful.
This project asks a central question:  
  
*Can linguistic features-such as vocabulary richness, readability, sentiment, and syntactic structure-predict whether a Yelp review receives engagement from other users?*    
  
To explore this question, we analyze Yelp restaurant reviews using a structured set of linguistic features and two machine-learning models. Engagement votes serve as a clean, review-level proxy for perceived helpfulness. The goal is not only to build predictive models but also to identify the linguistic patterns that characterize high-engagement reviews and to test whether these patterns generalize to hotel reviews. Together, these analyses show how linguistic signals can reveal meaningful differences in how readers perceive and respond to online content.


# 2. Related Work

Research on online review quality spans three major areas:

```{=latex}
\begin{itemize}
\item \textbf{Deception detection —}  
Early work (Jindal \& Liu, 2008; Mihalcea \& Strapparava, 2009; Ott et al., 2011) examined linguistic cues associated with fake reviews, later incorporating syntactic stylometry (Feng et al., 2012) and behavioral metadata (Mukherjee et al., 2013). However, most deception studies on Yelp rely on filtered reviews, which are not included in the Yelp Open Dataset and therefore cannot be used for verified deception labeling.

\item \textbf{Helpfulness prediction —}  
Research on Amazon, TripAdvisor, and StackExchange shows that crowd‑sourced helpfulness votes correlate with linguistic properties such as sentiment, specificity, readability, and emotional tone. These studies treat reader engagement as a proxy for perceived usefulness.

\item \textbf{Linguistic and stylistic analysis —}  
A broader line of work investigates stylistic markers including part‑of‑speech patterns, syntactic complexity, vocabulary richness, and other structural indicators of writing quality.

\end{itemize}
```  

This study aligns most closely with helpfulness‑prediction research but adapts it to Yelp by using engagement votes (useful, funny, cool) as a crowd‑sourced reliability signal. Rather than attempting deception detection without ground‑truth labels, the analysis focuses on how linguistic features relate to perceived reliability within the publicly available Yelp dataset.
 
# 3. Dataset 
## 3.1 Overview 
The Yelp Open Dataset is distributed across several structured JSON files, each capturing a different aspect of the platform's ecosystem. The **business** file contains 
location data, attributes, and category metadata for each business. The **review** file provides full review text along with user and business identifiers. The **user** file includes user‑level metadata such as friend mappings and account attributes. The **checkin** file records check‑in activity for businesses, while the **tip** file contains short user‑written suggestions that complement full reviews. Although the dataset also includes a **photo** file, which stores image captions and classifications (e.g., “food,” “menu,” “inside”), these auxiliary files are not used for this study. Only the business and review files are used, as they provide all necessary information for domain filtering, linguistic feature extraction, and engagement labeling.

Each review includes full text, a star rating, a timestamp and engagement metrics (useful, funny, cool) contributed by other users. These engagement signals serve as non-linguistic, crowd-sourced indicators of perceived helpfulness.

To reduce domain-driven variation, this study focuses exclusively on restaurant reviews. The data-processing pipeline proceeds as follows:

\begin{enumerate}
\item Identify restaurant businesses by filtering the business file using category keywords.
\item Merge the business and review files to retain only reviews associated with restaurant businesses.
\item Apply a 5\% random sample to the restaurant reviews to create a tractable subset for NLP feature extraction.
\item Compute linguistic features on the sampled reviews, including lexical, structural, syntactic, and stylistic indicators.
\item Construct engagement labels (High, Low, Intermediate) to finalize the modeling dataset.
\end{enumerate}

Applying this pipeline yields 5,245,324 restaurant reviews that meet the filtering criteria. From this set, a 5\% random sample is drawn, producing 261,882 reviews for feature engineering and subsequent analysis. This sampled subset is large enough to support robust descriptive comparisons and predictive modeling. After linguistic feature extraction, the reviews are categorized into high‑, low‑, and intermediate‑engagement groups, forming the final labeled dataset used throughout the study. Together, these steps produce a domain‑consistent, computationally manageable dataset that is fully annotated with both linguistic features and engagement labels.


## 3.2 Sampling

Although the Yelp Open Dataset contains millions of restaurant reviews, analyzing the full corpus is computationally unnecessary for the goals of this study. A 5\% random sample of the filtered restaurant‑review set (261,882 reviews) is used to balance computational efficiency with statistical power. Random sampling preserves variation in review length, diversity in writing style, distribution of star ratings, and the natural frequency of engagement signals, ensuring that linguistic patterns observed in the sample generalize to the broader population of Yelp restaurant reviews.

## 3.3 Label Construction

The Yelp Open Dataset does not provide an explicit engagement label, so we construct one using the three available engagement fields: useful, funny, and cool. Reviews are assigned to High-, Low-, or Intermediate-engagement categories based on the following criteria:  

```{=latex}
\begin{itemize}  
\item \textbf{High-engagement}: useful >= 2 OR funny >= 1 OR cool >= 1
\item \textbf{Low-engagement}: useful = 0 AND funny = 0 AND cool = 0  
\item \textbf{Intermediate}: all remaining cases  
\end{itemize}
```     

These categories capture how other users responded to a review and serve as a proxy for perceived helpfulness. Intermediate reviews are excluded from modeling because their ambiguous engagement levels weaken the contrast between the high‑ and low‑engagement groups.


## 3.4 Train–Test Split for Modeling
  
After constructing the labeled restaurant‑review dataset, an 80/20 train–test split is applied to support predictive modeling. The training set is used to fit all models, while the held‑out test set provides an unbiased estimate of predictive performance and supports SHAP‑based interpretability analysis.


# 4. Methodology 

This study investigates whether linguistic features can distinguish high-engagement from low-engagement Yelp reviews. The methodology integrates standardized text preprocessing, structured linguistic feature engineering, and interpretable machine-learning models. All methodological design choices are optimized for the restaurant review domain, which remains the primary focus of the analysis. Hotel reviews are incorporated later as a secondary validation domain to test generalizability, but they do not alter the core methodological pipeline. 


## 4.1 Preprocessing

A standardized preprocessing pipeline is applied to all review text to ensure consistency and improve the quality of linguistic feature extraction. Each review is lowercased to eliminate case-based variation. Non-alphabetic characters, extraneous symbols, and excessive punctuation are removed to reduce noise. The text is then tokenized into individual word units, enabling both word-level and sentence-level analysis. Stop words-common function words such as “the,” “and,” and “is”-are removed because they contribute minimally to stylistic differentiation. Lemmatization reduces words to their base forms (e.g., “running” → “run”), ensuring that grammatical variants are treated consistently. Together, these steps produce a clean, normalized representation of each review suitable for linguistic analysis and predictive modeling.

To summarize, all review text was standardized through the following steps to produce a clean representation suitable for linguistic analysis:

    
**Table 4.1: Preprocessing Steps**
  
| Steps             | Description / Example                                        |
|------------------------------------|---------------------------------|
| lowercasing       |  Run $\longrightarrow$ run   |
| removing non-alphabetic characters |  $, !   |
| tokenization      |  "the service was slow!" $\longrightarrow$ [the, service, was, slow, !]  |
| stop-word removal   | the, and, is   |
| lemmatization      | running $\longrightarrow$ run    |

## 4.2 Linguistic Feature Engineering

Linguistic features are extracted to capture multiple dimensions of writing style, emotional tone, semantic content, and textual complexity. The engineered feature set is organized into six categories: lexical, syntactic, readability, sentiment, structural, and semantic. Together, these categories provide a comprehensive representation of each review’s linguistic characteristics and support both descriptive analysis and predictive modeling.   

Feature extraction is implemented using standard Python NLP libraries, including spaCy (via nlp.pipe) for tokenization, lemmatization, part-of-speech tagging, and dependency parsing; textstat for computing established readability metrics; and VADER’s SentimentIntensityAnalyzer (via analyzer.polarity_scores) for deriving polarity-based sentiment scores.

The following tables detail the individual features within each category, providing a structured overview of the linguistic signals extracted from the review text. 

**Table 4.2: Lexical Features**
  
| Feature           | Description                                        |
|-------------------|----------------------------------------------------|
| token_count      | Number of alphabetic tokens in the text |
| unique_tokens    | Count of distinct surface word forms |
| unique_lemmas    | Count of distinct lemmas (vocabulary size) |
| type_lemma_ratio | Unique lemmas divided by total tokens; lemma-based diversity |
| avg_word_len     | Average number of characters per word |
| hapax_ratio      | Fraction of lemmas appearing exactly once |

**Table 4.3: Syntactic Features**  
  
| Feature        | Description                                    |
|-----------------|------------------------------------------------|
| noun_count    | Number of nouns; concreteness indicator |
| verb_count    | Number of verbs; action orientation |
| adj_count     | Number of adjectives; descriptiveness |
| adv_count     | Number of adverbs; modifier intensity |
| avg_dep_depth | Average syntactic dependency chain length; complexity |

**Table 4.4: Readability Features**  
  
| Feature             | Description |
|-----------------------|-----------------------------------------------------------------|
| flesch_reading_ease | Standard readability score; higher = easier |
| smog_index          | Readability index based on polysyllabic words |
| ari                 | Automated Readability Index; character-based difficulty |
| coleman_liau        | Readability score using characters per word and words per sentence |
| dale_chall          | Readability score based on proportion of “hard” words |

**Table 4.5: Sentiment Features**  
  
| Feature               | Description |
|-----------------------|----------------------------------------------------------------|
| sentiment_proxy | Heuristic polarity-based sentiment score |

**Table 4.6: Structural Features**  
  
| Feature           | Description |
|--------------------|------------------------------------------------------|
| sent_count        | Number of sentences in the text |
| avg_sent_len      | Average number of tokens per sentence |
| char_count        | Total number of characters |
| punctuation_count | Number of punctuation marks |
| burstiness        | Variance of sentence lengths; rhythm/pacing |

**Table 4.7: Semantic Features**  
  
| Feature           | Description |
|-------------------|----------------------------------------------------|
| concreteness_proxy | Nouns divided by tokens; concreteness indicator |
| action_ratio       | Verbs divided by tokens; action orientation |

Some features in the tables may look similar—for example, counting nouns versus calculating the proportion of nouns—but they actually measure different things. The raw counts in Table 4.3 (like noun_count) tell us about the structure of a sentence: how many nouns appear, regardless of length. When we divide those counts by the total number of tokens, as in Table 4.7 (like concreteness_proxy), the feature starts to reflect the meaning of the sentence instead. A sentence with a high noun ratio is usually more concrete and object‑focused, while a low ratio suggests more actions or descriptions. In other words, the same part‑of‑speech information can describe either how a sentence is built (syntactic) or what the sentence is about (semantic), depending on whether it is normalized.

These features serve as the inputs to the models described in Section 4.3.

## 4.3 Models 
To evaluate whether linguistic features can predict crowd-sourced engagement, two supervised learning models are employed:

```{=latex}
\begin{itemize}  
\item \textbf{Logistic Regression} serves as a linear, interpretable baseline model. It provides direct insight into how individual linguistic features influence the probability that a review receives engagement. Coefficients indicate whether specific features increase or decrease the likelihood of high engagement.
\item \textbf{XGBoost} is used to capture nonlinear relationships and interactions among linguistic features. Its gradient-boosted decision trees are effective for structured feature sets and often outperform linear models when interactions or thresholding behavior are present. XGBoost therefore offers complementary evidence regarding which linguistic patterns matter most for engagement.
\end{itemize}
```     

**Model performance** is assessed using: accuracy, precision, recall, F1-score, and area under the ROC curve (AUC). Together, these metrics provide a comprehensive view of predictive effectiveness and enable comparison between linear and nonlinear modeling approaches.  

## 4.4 Model Interpretation 

Interpretability is central to this study. Beyond evaluating predictive accuracy, the goal is to understand *which linguistic features most strongly influence engagement* and *whether these effects are consistent across modeling approaches*. Because Logistic Regression and XGBoost differ fundamentally in how they represent relationships-linear coefficients versus nonlinear tree-based interactions-interpretation relies on model-appropriate **feature importance** measures and a rank-based comparison framework.
 

```{=latex}
\begin{itemize}  

\item \textbf{Logistic Regression}    
Feature importance is derived from the magnitude and direction of model coefficients. Positive coefficients indicate features associated with higher engagement likelihood, while negative coefficients indicate features associated with lower engagement. To ensure comparability, importance is assessed using the \textbf{absolute value} of coefficients, reflecting each feature’s overall influence regardless of direction.

\item \textbf{XGBoost}  
Feature importance is evaluated using \textbf{SHAP}-based mean absolute contribution values. SHAP values quantify how much each feature contributes to predictions across all samples, capturing nonlinear effects, thresholding behavior, and interactions that Logistic Regression cannot represent.

\end{itemize}
```     

Because Logistic Regression coefficients and SHAP values operate on different scales, raw magnitudes are not directly comparable. Instead, importance is compared using **rank order**, providing a model-agnostic measure of relative influence.

# 5. Results 

## 5.1 Descriptive Lingustic Patterns

We summarize the distribution of engagement labels by categorizing each review into low‑engagement (0), intermediate (1), and high‑engagement (2). Intermediate reviews are reported for completeness but excluded from all modeling and interpretability analysis. All descriptive statistics in later sections therefore reflect only the high‑ and low‑engagement subsets.


**Table 5.1: Sample Overview**   
    
|Engagement Category |Count   | Percent| Included for modeling|
|----------------|---------------|------------|------------|
|low-engagement	 | 134,436  | 51.1% | 59.1% |
|high-engagement | 93,007   | 35.7% | 40.9% |
|intermediate    | 34,439   | 13.2% |excluded|
|**Total**       | 261,882  | 100.0%| 100.0%| 

**Table 5.2: Training and Test Sample Statistics**  
  
|Sample       |Engagement Category |Count   | Percent|
|-------------|----------------|------------|--------|
|Training     | low-engagement | 107,549    | 59.1% | 
|             | high-engagement| 74,405     | 40.9% |
|Test         | low-engagement | 26,887     | 59.1% |  
|             | high-engagement| 18,602     | 40.9% |


With the dataset defined, we examine raw linguistic differences between high‑ and low‑engagement reviews. Clear, systematic contrasts emerge across lexical, structural, syntactic, and stylistic dimensions.

```{=latex}
\begin{itemize}  

\item \textbf{High‑engagement reviews are substantially longer and more information‑dense}.  
They contain 311 more characters, 57 more tokens, 31 more unique tokens, and 29 more unique lemmas on average—each exceeding 50\% relative to the corresponding low‑engagement measures. These differences indicate that high‑engagement reviews provide richer content and more specific detail. Structural indicators reinforce this pattern: high‑engagement reviews include 3.66 additional sentences (57.61\%), exhibit longer average sentence length (11.70\%), and show higher burstiness (18.77\%), suggesting more developed narrative flow and greater topical elaboration.

\item \textbf{Syntactic features show strong divergence}.  
High‑engagement reviews use more nouns (+11.9), verbs (+7.3), adjectives (+4.9), and adverbs (+4.0)—each exceeding 50\% relative to the corresponding low‑engagement measures—reflecting denser descriptive content. Syntactic complexity increases as well: average dependency depth rises from 2.16 to 2.35 (8.85\%), indicating deeper syntactic structures and more elaborated phrasing.

\item \textbf{Stylistic markers point to greater expressive intensity}.  
High‑engagement reviews contain 8 additional punctuation marks, suggesting more emphasis, expressiveness, or rhetorical variation. Sentiment differences are modest but directionally consistent: the sentiment proxy increases from 0.68 to 0.70 (2.89\%), indicating a slightly more balanced or positive tone.

\item \textbf{Readability differences are small but directionally consistent}.  
High‑engagement reviews show slightly higher ARI, SMOG, and Dale–Chall scores, indicating marginally more complex writing. These shifts are minor compared to the pronounced structural, lexical, and syntactic contrasts observed above.

\end{itemize}
```  


**Table 5.3: Mean Differences between Engagement Groups**   

|Category |Feature	      |low engagement mean|high engagement mean|difference| pct diff|
|---------|----------------|--------|--------|-------|-------|
|Structural|char_count	    |412.07	| 723.79 |	\textcolor{blue} {311.73} | \textcolor{blue} {75.65\%}|
|Lexical|token_count	  |75.49	|132.87| \textcolor{blue} {57.38}| \textcolor{blue} {76.02\%}|
|Lexical|unique_tokens	|53.39	|84.73|	\textcolor{blue} {31.34}| \textcolor{blue} {58.71\%}|
|Lexical|unique_lemmas	|50.64	|79.60|	\textcolor{blue} {28.96}| \textcolor{blue} {57.18\%}|
|Syntactic|noun_count	    |15.76	|27.67|	\textcolor{blue} {11.90}| \textcolor{blue} {75.51\%}|
|Structural|punctuation_count|	10.30	|18.45|	\textcolor{blue} {8.16}| \textcolor{blue} {79.22\%}|
|Syntactic|verb_count	|9.05	|16.36	|\textcolor{blue} {7.31}|\textcolor{blue} {80.70\%}|
|Syntactic|adj_count	|7.93	|12.85	|\textcolor{blue} {4.92}|\textcolor{blue} {62.02\%}|
|Syntactic|adv_count	|5.68	|9.66	|\textcolor{blue} {3.99}|\textcolor{blue} {70.27\%}|
|Structural|sent_count	|6.36	|10.02|	\textcolor{blue} {3.66}|\textcolor{blue} {57.61\%}|
|Structural|avg_sent_len	|11.91	|13.30	|1.39|11.70%|
|Structural|burstiness	|5.99	|7.11	|1.12|18.77%|
|Readability|ari	|6.07	|6.42	|0.35 |5.81%|
|Readability|smog_index	|8.73	|8.95	|0.22|2.56%|
|Syntactic|avg_dep_depth	|2.16	|2.35	|0.19|8.85%|
|Lexical|type_lemma_ratio |	0.76	|0.68|	-0.07|-9.82%|
|Lexical|avg_word_len	|4.32	|4.25 |	-0.07|-1.58%|
|Lexical|hapax_ratio	|0.81	|0.77	|-0.04|-5.07%|
|Readability|dale_chall	|8.01	|8.05	|0.04|0.47%|
|Readability|coleman_liau	|6.73	|6.69	|-0.04|-0.55%|
|Sentiment|sentiment_proxy	|0.68	|0.70	|0.02|2.89%|
|Semantic|action_ratio	|0.11	|0.12 |	0.01|5.91%|
|Semantic|concreteness_proxy	|0.22	|0.21	|-0.01|-2.58%|
|Readability|flesch_reading_ease |74.96	|74.96|	0.00|0.00%|  
*All summary statistics in Table 5.3 are computed on the full dataset (training + test), whereas the tables in subsequent sections report results using the held‑out test set only.*     

## 5.2 Model Performance 

To evaluate how well each model predicts whether a Yelp restaurant review receives engagement, we report five standard classification metrics. Each captures a different aspect of predictive behavior:

```{=latex}
\begin{itemize}

\item \textbf{Accuracy -- } proportion of all predictions the model gets correct; reflects overall correctness.

\item \textbf{Precision --} among reviews predicted as high‑engagement, the fraction that truly are; measures false‑positive control.

\item \textbf{Recall -- } among all truly high‑engagement reviews, the fraction the model successfully identifies; measures false‑negative control.

\item \textbf{F1 Score -- } harmonic mean of precision and recall; balances the trade‑off between catching more positives and being correct when doing so.

\item \textbf{AUC -- } measures how well the model ranks reviews by engagement likelihood; higher values indicate better separation between classes.

\end{itemize}
```


With 41% high‑engagement reviews, a majority‑class baseline still yields an AUC of 0.5, so the models’ AUC values around 0.705–0.706 represent substantial discriminative power from text alone. 


**Table 5.4: Logistic Regressiion and XGB Model Performance**   
  
|Model                 |	Accuracy |	Precision |	Recall	|F1 Score |	**AUC** |
|----------------------|-----------|----------|---------|---------|--------|
|XGBoost	|\textcolor{blue} {0.67080}	  |0.63279	|\textcolor{blue} {0.46457}	|\textcolor{blue}{0.53579}	| \textcolor{blue}{0.70648}  |
|Logistic Regression	|0.67054	| \textcolor{blue}{0.64299}	|0.43694	|0.52031	|0.70505|  


To contextualize these results, we summarize the main performance patterns observed across the two models:


```{=latex}
\begin{itemize}

\item \textbf{Overall performance is nearly identical.}  
Both models achieve comparable accuracy and AUC, indicating that they capture the same broad linguistic signal underlying Yelp engagement.

\item \textbf{Logistic Regression is more selective.}  
It produces higher precision but lower recall, meaning it identifies fewer high‑engagement reviews but is more accurate when it does so. This reflects a conservative decision boundary that limits false positives.

\item \textbf{XGBoost is more inclusive.}  
It achieves higher recall but lower precision, suggesting it detects more potentially engaging reviews while tolerating additional false positives. This pattern aligns with XGBoost`s ability to model nonlinear interactions among linguistic features.

\item \textbf{F1 scores highlight the trade‑off.}  
XGBoost`s slightly higher F1 score indicates a better balance between precision and recall, giving it a modest advantage in overall classification performance.

\item \textbf{Summary comparison.}  
Although both models perform similarly, XGBoost’s flexibility yields small gains in identifying reviews likely to receive engagement, while Logistic Regression offers more conservative, precision‑oriented predictions.

\end{itemize}
```



## 5.3 Feature Importance 

To identify the most influential predictors among the 24 linguistic features used across both models, we classify the top one‑third of ranked features in each model as top features. As shown in Table 5.3, features that appear in the top tier for both models are highlighted in blue, while those that are top‑tier in only one model are highlighted in green. Across both models, four linguistic features consistently emerge as top‑tier predictors of engagement: 


```{=latex}
\begin{itemize}

\item unique lemmas (lexical richness)

\item punctuation count (stylistic intensity)

\item character count (verbosity)

\item avg\_dep\_depth (syntactic complexity) 

\end{itemize}
``` 

Importantly, three of these features—unique lemmas, punctuation count, and character count—also showed clear mean differences between high‑ and low‑engagement reviews, indicating that their model‑based importance aligns with observable patterns in the data. Although avg_dep_depth exhibited a more modest mean difference, its consistent top‑tier ranking in both Logistic Regression and XGBoost suggests that syntactic complexity contributes additional predictive signal beyond what is captured by surface‑level lexical and stylistic features. Together, these results show that the features elevated by both models correspond to stable linguistic cues that meaningfully shape how readers evaluate user‑generated content.


**Table 5.5: Cross-Model Feature Importance Rank Comparison** 

| Feature                   | Coeff |Abs Coeff| LR Rank |SHAP Importance|SHAP Rank|
|---------------------------|-------|---------|---------|---------------|---------|
|**unique_lemmas**|	1.47|	1.47|\textcolor{blue} {1}|	0.34|\textcolor{blue} {1}|
|**punctuation_count**|	0.30|	0.30|\textcolor{blue} {3}|	0.17|\textcolor{blue} {2}|
|unique_tokens|	-0.07|	0.07|	10|	0.09|	\textcolor{green} {3}|
|avg_word_len	|-0.02|	0.02|	18|	0.08|	\textcolor{green} {4}|
|**char_count**|	-1.11|	1.11| \textcolor{blue} {2}	|0.07|\textcolor{blue} {5}|
|**avg_dep_depth**|	0.10|	0.10|	\textcolor{blue} {6}	|0.06|\textcolor{blue} {6}|
|type_lemma_ratio	|-0.06|	0.06|	12|	0.06| \textcolor{green} {7}|
|flesch_reading_ease	|0.04|	0.04|	16|	0.06|\textcolor{green} {8}|
|dale_chall|	0.07|	0.07|	9	|0.05|	9|
|sentiment_proxy|	-0.03|	0.03|	17|	0.05|	10|
|avg_sent_len	|-0.04|	0.04|	15|	0.05|	11|
|hapax_ratio|	-0.06|	0.06|	11|	0.04|	12|
|coleman_liau|	-0.01|	0.01|	21|	0.03|	13|
|ari	|0.02|	0.02|	19|	0.03|	14|
|noun_count	|0.04|	0.04|	13|	0.02|	15|
|sent_count|	0.08|	0.08|	\textcolor{green} {7}|	0.02|	16|
|action_ratio	|0.08	|0.08	|\textcolor{green} {8}	|0.02	|17|
|verb_count|	-0.29|	0.29|	\textcolor{green} {4}|	0.02|	18|
|concreteness_proxy|	0.01|	0.01|	23|	0.02|	19|
|burstiness	|-0.02|	0.02|	20|	0.02|	20|
|token_count|	0.24|	0.24|	\textcolor{green} {5}	|0.02|	21|
|adj_count	  |-0.04|	0.04|	14|	0.02|22|
|adv_count	  |-0.01|	0.01|	24|	0.01|23|
|smog_index|	-0.01|	0.01|	22	|0.01|	24|  


While the rank comparison establishes that both models prioritize similar linguistic signals, it does not reveal how these features relate to one another across models or where the two approaches diverge. To visualize cross‑model agreement directly, Figure 5.1 plots each feature’s Logistic Regression rank against its XGBoost SHAP rank.

```{=latex}
\FloatBarrier
\begin{figure}[H]
\centering
{\large \textbf{Figure 5.1: Cross-Model Feature Rank Comparison}}\

\includegraphics[width=1.05\textwidth]{fig_lr_shap_rank_distance_colored.png}
\label{fig:lr-shap-rank}
\end{figure}
```  

The scatter plot shows a clear concentration of features along the 45‑degree diagonal, indicating that the two models assign broadly similar importance ranks to most predictors. This diagonal clustering is consistent with the moderate Spearman correlation (0.49) between the two ranking systems and reinforces the conclusion that both models rely on a shared core of linguistic signals.

The color‑coded points in the figure further clarify the structure of agreement and disagreement:

```{=latex}
\begin{itemize}
\item \textcolor{green}{\textbf{Green}} points features ranked in the top eight by both models---cluster tightly near the diagonal and represent the strongest areas of cross‑model alignment.
\item \textcolor{blue}{\textbf{Blue}} points show small deviations from the diagonal, reflecting modest rank differences and generally consistent behavior across models.
\item \textcolor{amber}{\textbf{Orange}} points sit farther from the diagonal, indicating moderate disagreement in how the models prioritize these features.
\item \textcolor{red}{\textbf{Red}} points notably token count, verb count, and avg\_word\_len---fall well outside the diagonal corridor and mark the largest rank discrepancies.
\end{itemize}
```

These off-diagonal features reflect systematic differences in how the two models respond to certain linguistic patterns rather than conflicting interpretations. XGBoost gives higher importance to features with nonlinear or interaction-driven effects (e.g., readability metrics such as flesch_reading_ease and coleman_liau), whereas Logistic Regression emphasizes features with strong linear associations to engagement (e.g., token_count, verb_count, sentence_count, action_ratio). Overall, the scatter plot shows a dense diagonal core with only a few outliers, indicating that the two models agree on most feature rankings and diverge meaningfully on only a small subset.

Taken together, the rank comparison and cross-model visualization establish which linguistic features matter most and where the models differ. To understand how these shared top-tier predictors influence predictions, the next subsection examines SHAP dependence plots for unique lemmas, punctuation count, character count, and avg_dep_depth. These plots reveal each feature’s contribution pattern across its range, providing a clearer view of how both models translate the same linguistic signals into engagement predictions.



## 5.4 Feature-Effect Patterns for Shared Top‑Tier Features

To examine how individual linguistic features shape model predictions, we compare XGBoost’s SHAP‑dependence plots with the Logistic Regression $X\beta$ contribution plots. SHAP values capture nonlinear, context‑dependent effects, while $X\beta$ values represent strictly linear log‑odds contributions. Because these arise from different modeling frameworks, their y‑axes are not directly comparable, so we focus on the direction and overall trend of each feature’s effect rather than its absolute magnitude. Although the visual styles differ, both plots show each feature’s marginal impact on engagement and highlight how the two models learn different feature–outcome relationships under their respective assumptions. We then apply this comparison to the four features both models rank most highly.


```{=latex}
\FloatBarrier
\begin{figure}[H]
\centering
{\large \textbf{Figure 5.2:  XGB SHAP Dependence Plot}}\

\includegraphics[width=1.05\textwidth]{05_xgb_top_features_shap_dependence.png}
\label{fig:xgb-shap-rank}
\end{figure}
```  


```{=latex}
\FloatBarrier
\begin{figure}[H]
\centering
{\large \textbf{Figure 5.3:  Logistic Regression XBETA Plot}}\

\includegraphics[width=1.05\textwidth]{05_lr_top_features_xbeta.png}
\label{fig:lr-xbeta}
\end{figure}
```  

The two figures reveal that three of the shared top‑tier features exhibit broadly similar directional patterns across models, while one shows a clear divergence.

```{=latex}
\begin{itemize}
\item For \textbf{unique lemmas} and \textbf{punctuation count}, both models show positive contributions, with XGBoost capturing gentle curvature and Logistic Regression applying constant linear increases. 
\item For \textbf{character count}, XGBoost displays an initial rise, followed by a decline and eventual plateau, while Logistic Regression applies a steady negative effect. 
\item For \textbf{avg\_dep\_depth}, both models indicate a positive relationship, with XGBoost showing mild saturation and Logistic Regression maintaining a linear rise.
\end{itemize}
```

With these feature‑level patterns established for restaurant reviews, we next examine whether the same signals hold in a different domain by applying the same analysis to Yelp hotel reviews. 

## 5.5 Cross-Domain Check using Yelp Hotel Reviews

\textcolor{red}{placeholder -  not finished yet} 

To evaluate whether the linguistic signals identified in restaurant reviews generalize to other domains, we applied the same feature set and modeling pipeline to Yelp hotel reviews. Using both Logistic Regression and XGBoost, we extracted each model’s top-ranked predictors and examined the overlap between the two models as well as their consistency with the restaurant results.

The hotel models highlight the same core linguistic features—lexical richness, punctuation use, review length, and syntactic complexity—as their most influential predictors. Logistic Regression and XGBoost again show strong agreement on these top-tier features, and the shared hotel feature set closely matches the shared feature set obtained from restaurant reviews. This cross-domain consistency indicates that the linguistic signals identified in this study are stable across different types of consumer reviews and are not specific to the restaurant domain.

Hotel reviews are more factual and uniform, leading to stronger cross-model agreement.  
Restaurant reviews show greater stylistic variability, producing more nonlinear behavior and wider divergence between models. Hotel reviews are more consistent, producing tighter agreement. These results show that while stylistic variability differs across domains, core linguistic signals-verbosity, lexical richness, descriptive intensity, and readability-remain influential predictors of engagement.

# 6. Conclusion and Discussion

## 6.1 Interpretation and Implications 

This study shows that writing style meaningfully shapes how readers engage with Yelp reviews. Linguistic features such as lexical richness, punctuation use, review length, and syntactic complexity consistently predict engagement, and both Logistic Regression and XGBoost identify these same core signals as the strongest contributors. The models differ in how they express these effects—Logistic Regression applies uniform linear relationships, while XGBoost captures mild nonlinearities and saturation—but the underlying patterns are stable across modeling approaches. These results suggest that detailed, varied, and clearly structured writing tends to resonate more with readers, and that linguistic cues offer a reliable lens for understanding perceived usefulness in user-generated content. The cross-domain check with hotel reviews further indicates that these signals generalize beyond a single review category.


## 6.2 Limitations 

Engagement is an imperfect proxy for reliability. Some high-quality reviews may receive few or no votes due to limited visibility, timing, or platform dynamics. The primary focus on restaurant reviews may constrain generalizability, though the hotel results provide evidence of broader applicability. Additionally, linguistic features capture only part of what drives engagement; factors such as reviewer reputation, business characteristics, and platform-level exposure are not modeled here. Despite these limitations, the findings demonstrate that linguistic structure and writing style offer a meaningful and interpretable foundation for understanding how readers evaluate online reviews.  

## 6.3 Future Work 

Future work could incorporate reviewer‑ and business‑level metadata to separate linguistic effects from visibility dynamics, extend the analysis to additional domains, and examine temporal patterns in engagement. Combining linguistic features with modern text embeddings may also capture deeper semantic cues while preserving interpretability. These directions would provide a more complete view of how writing style and contextual factors jointly shape engagement in online review platforms.



# References

[1] Jindal, N., & Liu, B. (2008). Opinion spam and analysis. *Proceedings of the ACM International Conference on Web Search and Data Mining (WSDM)*.

[2] Mihalcea, R., & Strapparava, C. (2009). The lie detector: Explorations in the automatic recognition of deceptive language. *Proceedings of ACL-IJCNLP (Joint Conference of the Association for Computational Linguistics and the International Joint Conference on Natural Language Processing)*.

[3] Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. *Proceedings of the Association for Computational Linguistics (ACL)*.

[4] Feng, S., Banerjee, R., & Choi, Y. (2012). Syntactic stylometry for deception detection. *Proceedings of the Association for Computational Linguistics (ACL)*.

[5] Mukherjee, A., Venkataraman, V., Liu, B., & Glance, N. (2013). What Yelp fake review filter might be doing? *Proceedings of the AAAI International Conference on Web and Social Media (ICWSM)*.

[6] Li, H., Wang, X., Chen, H., & Xu, G. (2022). Online spam review detection: A survey of literature. *Journal article*.

[7] Gupta, R., Jindal, V., & Kashyap, I. (2024). Recent state-of-the-art of fake review detection: A comprehensive review. *The Knowledge Engineering Review (Cambridge University Press)*.


