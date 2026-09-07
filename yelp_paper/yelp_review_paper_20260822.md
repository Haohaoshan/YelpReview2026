---
implicit_figures: false
header-includes:
  - \usepackage[margin=1.1in]{geometry}
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{placeins}
  - \usepackage[table]{xcolor}
  - \definecolor{amber}{RGB}{255,191,0}
  - \definecolor{greenpoint}{RGB}{0,120,0}
  - \definecolor{bluepoint}{RGB}{30,90,180}
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

\textit{Both authors contributed equally to all stages of the research, from conceptual design and dataset construction to linguistic analysis, modeling, and manuscript preparation.}

\end{titlepage}

\fontsize{13pt}{16pt}\selectfont


# Abstract 

Online reviews play a central role in how people choose restaurants and other services, yet only a small fraction of Yelp reviews attract engagement from readers. This study examines whether specific linguistic features can help explain that gap. Using a sample of restaurant reviews from the Yelp Open Dataset, we extracted features capturing sentiment, lexical diversity, readability, syntactic structure, and formatting choices. Logistic Regression and XGBoost models were trained to distinguish high‑engagement reviews from those that receive no votes. Both models showed that writing style carries measurable predictive signal. SHAP analysis highlighted review length, lexical diversity, syntactic complexity, and punctuation patterns as consistent contributors across models. Taken together, the results suggest that reviews that are detailed, readable, and emotionally balanced are more likely to resonate with readers and provide value on online review platforms. 


# 1. Introduction

Online reviews shape everyday decisions, from choosing a restaurant to booking a hotel. Platforms like Yelp host millions of reviews, yet only some of them attract engagement from other users through votes such as “useful,” “funny,” or “cool.” Understanding why certain reviews draw attention while others do not is important for both readers and platforms that rely on user‑generated content.

Much of the existing research on online reviews focuses on detecting deception, but the Yelp Open Dataset does not include labels that identify whether a review is fake. Rather than trying to infer deception without ground‑truth data, this study looks at a different question: whether aspects of writing style can help explain which reviews readers respond to and find helpful.

This project asks a central question:

*Can linguistic features—such as vocabulary richness, readability, sentiment, and syntactic structure—predict whether a Yelp review receives engagement from other users?*

To investigate this question, we analyze Yelp restaurant reviews using a structured set of linguistic features and two machine‑learning models. Engagement votes serve as a practical, review‑level indicator of perceived helpfulness. The goal is not simply to build predictive models, but to identify the linguistic patterns that distinguish high‑engagement reviews from those that receive no votes. These analyses show how writing style can shape how readers perceive and respond to online content. Understanding why certain reviews draw attention while others do not matters for both writers and platforms. Writers benefit from knowing what kinds of reviews are more likely to be noticed, and platforms can use linguistic patterns to identify helpful reviews even when votes are missing. Because many reviews receive no engagement at all, writing‑style cues give platforms an additional signal for deciding which reviews should appear near the top.

In this study, the machine‑learning models are used as tools to highlight which linguistic features matter most. The focus is on writing style and its relationship to engagement, not on comparing model performance.


# 2. Related Work

Research on online review quality spans three major areas:

```{=latex}
\begin{itemize}
\item \textbf{Deception detection —}  
Early work (Jindal \& Liu, 2008; Mihalcea \& Strapparava, 2009; Ott et al., 2011) examined linguistic cues associated with fake reviews, later incorporating syntactic stylometry (Feng et al., 2012) and behavioral metadata (Mukherjee et al., 2013). However, most deception studies on Yelp rely on filtered reviews, which are not included in the Yelp Open Dataset and therefore cannot be used for verified deception labeling.

\item \textbf{Helpfulness prediction —}  
Research on Amazon (Kim et al., 2006; Ghose \& Ipeirotis, 2011), TripAdvisor (Liu et al., 2008), and StackExchange (Danescu‑Niculescu‑Mizil et al., 2013) shows that crowd‑sourced helpfulness votes correlate with linguistic properties such as sentiment, specificity, readability, and emotional tone. These studies treat reader engagement as a proxy for perceived usefulness.

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

Because engagement votes accumulate over time, newer reviews may appear low‑engagement simply because they have had less opportunity to receive votes. Recognizing this potential bias, we later conduct a temporal segmentation analysis to examine whether the linguistic patterns identified in this study remain stable across older and newer reviews. The results show that the influential features are consistent for older reviews, suggesting that the main patterns are not driven solely by differences in exposure time. 


## 3.2 Sampling

Although the Yelp Open Dataset contains millions of restaurant reviews, analyzing the full corpus is computationally unnecessary for the goals of this study. A 5\% random sample of the filtered restaurant‑review set (261,882 reviews) is used to balance computational efficiency with statistical power. Random sampling preserves variation in review length, diversity in writing style, distribution of star ratings, and the natural frequency of engagement signals, ensuring that linguistic patterns observed in the sample generalize to the broader population of Yelp restaurant reviews. The 5% sampling rate was selected to make the feature‑engineering stage computationally manageable. Generating linguistic features for millions of reviews requires substantial processing time, especially when using NLP tools such as spaCy and textstat on a standard home computer. Several sampling percentages were tested, and 5% provided a practical balance: the resulting dataset of over 260,000 reviews is large enough to preserve linguistic diversity and support reliable analysis, while keeping preprocessing and feature extraction within a reasonable runtime of a few hours.  


## 3.3 Label Construction

The Yelp Open Dataset does not provide an explicit engagement label, so we construct one using the three available engagement fields: useful, funny, and cool. Reviews are assigned to High-, Low-, or Intermediate-engagement categories based on the following criteria:  

```{=latex}
\begin{itemize}  
\item \textbf{High-engagement}: useful >= 2 OR funny >= 1 OR cool >= 1
\item \textbf{Low-engagement}: useful = 0 AND funny = 0 AND cool = 0  
\item \textbf{Intermediate}: all remaining cases  
\end{itemize}
```     

These categories capture how other users responded to a review and serve as a proxy for perceived helpfulness. The thresholds were chosen to create a clear separation between reviews that received meaningful attention and those that received none. Although “funny” and “cool” votes are not direct measures of helpfulness, they still indicate user interaction and therefore contribute to the broader concept of engagement used in this study. Intermediate reviews were excluded because their ambiguous engagement levels would weaken the contrast between the high‑ and low‑engagement groups and reduce interpretability in subsequent analyses.


## 3.4 Train–Test Split for Modeling
  
After constructing the labeled restaurant‑review dataset, an 80/20 train–test split is applied to support predictive modeling. The training set is used to fit all models, while the held‑out test set provides an unbiased estimate of predictive performance and supports SHAP‑based interpretability analysis.


# 4. Methodology 

This study investigates whether linguistic features can distinguish high-engagement from low-engagement Yelp reviews. The methodology integrates standardized text preprocessing, structured linguistic feature engineering, and interpretable machine-learning models, all applied within the restaurant review domain to maintain linguistic consistency and avoid cross‑domain confounds.


## 4.1 Preprocessing

A minimal preprocessing pipeline is applied to all review text to ensure consistency while preserving the full linguistic structure required for feature extraction. Each review is lowercased, stripped of leading and trailing whitespace, and normalized so that repeated whitespace is collapsed into a single space. No punctuation or stop words are removed, and no additional text transformations are applied. The normalized text is then passed directly into spaCy, and all linguistic features—lexical, syntactic, readability, structural, sentiment, and semantic—are computed from the resulting document.


## 4.2 Linguistic Feature Engineering

Linguistic features are extracted to capture multiple dimensions of writing style, emotional tone, and textual complexity. All features are computed from the same spaCy document produced by the preprocessing step in Section 4.1. Feature-specific filtering is applied within the extraction process to isolate the relevant linguistic signals. For example, lexical features use alphabetic tokens and their lemmas, while syntactic, readability, structural, sentiment, and semantic features rely on the full document, including punctuation and stop words.

Feature extraction is implemented using spaCy for tokenization, lemmatization, part-of-speech tagging, and dependency parsing; textstat for readability metrics; and VADER for sentiment scoring. The resulting feature set spans six interpretable categories—lexical, syntactic, readability, sentiment, structural, and semantic—providing a comprehensive representation of each review’s linguistic characteristics.
  
  
**Table 4.1: Lexical Features**
  
| Feature           | Description                                        |
|-------------------|----------------------------------------------------|
| token_count      | Number of alphabetic tokens in the text |
| unique_tokens    | Count of distinct surface word forms |
| unique_lemmas    | Count of distinct lemmas (vocabulary size) |
| type_lemma_ratio | Unique lemmas divided by total tokens; lemma-based diversity |
| avg_word_len     | Average number of characters per word |
| hapax_ratio      | Fraction of lemmas appearing exactly once |

**Table 4.2: Syntactic Features**  
  
| Feature        | Description                                    |
|-----------------|------------------------------------------------|
| noun_count    | Number of nouns; concreteness indicator |
| verb_count    | Number of verbs; action orientation |
| adj_count     | Number of adjectives; descriptiveness |
| adv_count     | Number of adverbs; modifier intensity |
| avg_dep_depth | Average syntactic dependency chain length; complexity |

**Table 4.3: Readability Features**  
  
| Feature             | Description |
|-----------------------|-----------------------------------------------------------------|
| flesch_reading_ease | Standard readability score; higher = easier |
| smog_index          | Readability index based on polysyllabic words |
| ari                 | Automated Readability Index; character-based difficulty |
| coleman_liau        | Readability score using characters per word and words per sentence |
| dale_chall          | Readability score based on proportion of “hard” words |

Readability metrics are included to capture surface‑level aspects of writing complexity that may influence how readers process a review. These features complement the lexical and syntactic measures by providing an additional perspective on how easily a review can be read. Including readability scores ensures that the feature set covers multiple dimensions of writing style, allowing later analyses to determine which aspects are most relevant to engagement.


**Table 4.4: Sentiment Features**  
  
| Feature               | Description |
|-----------------------|----------------------------------------------------------------|
| sentiment_proxy | Heuristic polarity-based sentiment score |

**Table 4.5: Structural Features**  
  
| Feature           | Description |
|--------------------|------------------------------------------------------|
| sent_count        | Number of sentences in the text |
| avg_sent_len      | Average number of tokens per sentence |
| char_count        | Total number of characters |
| punctuation_count | Number of punctuation marks |
| burstiness        | Variance of sentence lengths; rhythm/pacing |

**Table 4.6: Semantic Features**  
  
| Feature           | Description |
|-------------------|----------------------------------------------------|
| concreteness_proxy | Nouns divided by tokens; concreteness indicator |
| action_ratio       | Verbs divided by tokens; action orientation |


Several features in the tables may appear similar—for example, counting nouns versus computing the proportion of nouns—but they capture different aspects of language. Raw counts describe structural properties of a review, while normalized ratios reflect meaning‑oriented characteristics such as concreteness or action orientation. The feature set is intentionally broad and interpretable, prioritizing transparent connections to writing style over more opaque representations like n‑grams or neural embeddings. Some features naturally overlap across categories—for instance, lexical diversity relates to readability, and syntactic depth correlates with sentence length—but each category highlights a distinct dimension of linguistic expression. This comprehensive, interpretable feature set provides the foundation for the modeling analyses introduced in Section 4.3.


## 4.3 Models 

To evaluate whether linguistic features can predict crowd‑sourced engagement, two supervised learning models are employed. Although both models use the same feature set, they differ in how the input features are handled within each model pipeline due to fundamental differences in their learning mechanisms.

The modeling approach balances interpretability with predictive flexibility. Logistic regression serves as a transparent baseline that provides easily interpretable coefficients and establishes a reference point for how individual linguistic features relate to engagement. XGBoost is used as the primary model because it can capture nonlinear relationships and interactions among features that linear models cannot represent, while still offering structured, feature‑level interpretability through SHAP values. More complex neural architectures were considered, but they provide limited insight into which specific linguistic characteristics drive engagement. Because the goal of this study is to understand how writing style influences reader responses, the analysis focuses on models that offer both strong predictive performance and clear interpretability.

Model training is performed on the subset of reviews labeled as either high‑engagement or low‑engagement. Intermediate reviews are excluded prior to modeling, as described in Section~5.1. After this filtering step, the remaining dataset is split into training and test sets using an 80/20 ratio. Although this reduces the size of the modeling dataset relative to the original 5\% sample (261{,}882 reviews), the resulting high‑ and low‑engagement subsets remain sufficiently large for reliable model estimation.

Hyperparameters for XGBoost are selected manually rather than through automated tuning. The chosen configuration—300 trees, maximum depth of 6, learning rate of 0.05, and moderate subsampling and column sampling—reflects a balanced setting that provides enough model capacity to capture nonlinear relationships while avoiding excessive complexity. These values are intentionally conservative to reduce overfitting and maintain stable SHAP‑based interpretability. Because the goal of this study is to understand how linguistic features relate to engagement rather than to maximize predictive performance, a manually curated configuration is appropriate and avoids the opacity introduced by aggressive hyperparameter optimization. Logistic regression uses standard regularization settings because it serves as a baseline model.

```{=latex}
\begin{itemize}
\item \textbf{Logistic Regression} serves as a linear, interpretable baseline model. Because logistic regression optimizes a linear decision boundary using gradient-based methods, feature scaling is required to ensure stable coefficient estimation and meaningful regularization. All linguistic features are standardized using a \texttt{StandardScaler} before model training. This preprocessing step allows the model’s coefficients to reflect the relative influence of each feature on the probability that a review receives engagement.

\item \textbf{XGBoost} is used to capture nonlinear relationships and interactions among linguistic features. As a tree-based ensemble method, XGBoost determines splits based on the \emph{ordering} of feature values rather than their absolute magnitudes. Consequently, feature scaling is not applied: decision trees are invariant to monotonic transformations, and standardization does not improve performance. Using raw features preserves sparsity patterns and avoids unnecessary transformations. XGBoost therefore provides complementary evidence regarding which linguistic patterns matter most for engagement.
\end{itemize}
```     

**Model performance** is assessed using: accuracy, precision, recall, F1-score, and area under the ROC curve (AUC). Together, these metrics provide a comprehensive view of predictive effectiveness and enable comparison between linear and nonlinear modeling approaches.  

## 4.4 Model Interpretation 

Interpretability is central to this study. Beyond evaluating predictive accuracy, the goal is to understand *which linguistic features most strongly influence engagement* and *whether these effects are consistent across modeling approaches*. Because Logistic Regression and XGBoost differ fundamentally in how they represent relationships—linear coefficients versus nonlinear tree‑based interactions—interpretation relies on model‑appropriate feature **importance measures** and a rank‑based comparison framework. These differences also interact with preprocessing: Logistic Regression operates on standardized features, while XGBoost operates on raw, unscaled features, which affects how importance values should be interpreted.



```{=latex}
\begin{itemize}

\item \textbf{Logistic Regression}  
Feature importance is derived from the magnitude and direction of model coefficients. Because the model is trained on standardized features, each coefficient reflects the change in engagement likelihood associated with a one–standard‑deviation increase in the corresponding linguistic feature. Positive coefficients indicate features associated with higher engagement likelihood, while negative coefficients indicate features associated with lower engagement. To ensure comparability across features, importance is assessed using the \textbf{absolute value} of coefficients, reflecting each feature’s overall influence regardless of direction. 

\item \textbf{XGBoost}  
Feature importance is evaluated using \textbf{SHAP}-based mean absolute contribution values. SHAP values quantify how much each raw (unscaled) feature contributes to predictions across all samples, capturing nonlinear effects, thresholding behavior, and interactions that Logistic Regression cannot represent. Because XGBoost is tree‑based and invariant to monotonic transformations, operating on unscaled features preserves natural feature structure and does not affect interpretability. SHAP values are computed using the TreeSHAP algorithm, which provides exact, efficient feature attributions for tree‑based models.

\end{itemize}
```     

Because Logistic Regression coefficients and XGBoost SHAP values operate on different scales—and arise from fundamentally different model classes—raw magnitudes are not directly comparable. Instead, importance is compared using **rank order**, providing a model‑agnostic measure of relative influence across both linear and nonlinear modeling approaches.



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

High‑engagement reviews are noticeably longer and contain more tokens, which suggests that length may play an important role in how readers respond to a review. However, length alone may not fully explain the differences between the two groups. Later sections use model‑based analyses to examine whether additional linguistic features—such as lexical diversity, syntactic structure, and punctuation—contribute to engagement beyond simple differences in review length.


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

## 5.3 Top Features Identified by XGBoost  

To identify the most influential predictors among the 24 linguistic features, we classify the top one‑third of each model’s ranked features as its top‑tier set. This criterion provides a consistent basis for determining which predictors each model considers most important. Table 5.5 summarizes these rankings: rank values that fall within the top tier for either model are highlighted in blue, along with their mean percentage differences from Section 5.1.

XGBoost elevates eight features into its top‑tier group. These features fall naturally into several interpretable linguistic categories: 

```{=latex}
\begin{itemize}

\item \textbf{Content richness and lexical sophistication:}  
unique lemmas (lexical richness), unique tokens (lexical diversity), type\_lemma\_ratio (lexical sophistication)

\item \textbf{Stylistic intensity:}  
punctuation count (stylistic emphasis)

\item \textbf{Review length:}  
character count (verbosity)

\item \textbf{Complexity and readability:}  
avg\_dep\_depth (syntactic complexity), avg\_word\_len (morphological complexity), flesch\_reading\_ease (readability)

\end{itemize}
``` 

This grouping highlights the dimensions XGBoost finds most predictive: reviews that are lexically rich, stylistically expressive, sufficiently detailed, and moderately complex and readable tend to receive more engagement. Several of these features, such as character count and punctuation count, also show clear mean differences between high‑ and low‑engagement reviews, indicating that their importance aligns with observable linguistic variation. Others, such as avg_word_len and type_lemma_ratio, rise in importance because XGBoost captures nonlinear interactions that amplify their predictive value.



**Table 5.5: Cross-Model Feature Importance Rank Comparison** 

Category          | Feature                           |Abs Pct Diff| LR Coeff    | LR Rank |SHAP Rank  |
|-----------------|-----------------------------------|------------|-------------|---------|-----------|
|Lexical |**unique_lemmas**|57.18%|	1.47|\textcolor{blue} {1}|\textcolor{blue} {1}|
|Lexical |**unique_tokens**|58.71%|	-0.07 |	10|	\textcolor{blue} {3}|
|Lexical |**avg_word_len**|1.58%|-0.02| 18|		\textcolor{blue} {4}|
|Lexical |**type_lemma_ratio**|9.82%	|-0.06|	12| \textcolor{blue} {7}|
|Lexical |hapax_ratio|5.07%|	-0.06|	11|	12|
|Lexical |token_count | 76.02%|	0.24|	\textcolor{blue} {5}	|	21|
|Syntactic |**avg_dep_depth**|8.85%|	0.10|	\textcolor{blue} {6}	|\textcolor{blue} {6}|
|Syntactic |noun_count	|75.51%|0.04|		13|	15|
|Syntactic |verb_count| 80.70%|	-0.29|\textcolor{blue} {4}|	18|
|Syntactic |adj_count	|62.02%  |-0.04|14|22|
|Syntactic |adv_count	|70.27%|-0.01|24|23|
|Readability |**flesch_reading_ease**|0.003%|0.04|	16|\textcolor{blue} {8}|
|Readability |dale_chall|0.47%|	0.07|	9	|	9|
|Readability |coleman_liau|0.55%|	-0.01|	21|	13|
|Readability |ari |5.81%	|0.02|	19|	14|
|Readability |smog_index|2.56%|	-0.01|	22	|	24|  
|Sentiment |sentiment_proxy|2.89%|	-0.03|	17|		10|
|Structural |**punctuation_count**| 79.22%|	0.30|\textcolor{blue} {3}|\textcolor{blue} {2}|
|Structural |**char_count**|75.65%|	-1.11| \textcolor{blue} {2}	|\textcolor{blue} {5}|
|Structural |avg_sent_len |	11.70%|-0.04|		15|	11|
|Structural |sent_count|57.61%|	0.08|		\textcolor{blue} {7}|	16|
|Structural |burstiness	|18.77%|-0.02|	20|	20|
|Semantic |action_ratio	|5.91%|0.08|\textcolor{blue} {8}		|17|
|Semantic |concreteness_proxy|2.58%|	0.01|	23|	19|



## 5.4 Cross-Model Rank Agreement

To visualize the cross-model rank agreement, Figure 5.1 plots each feature’s Logistic Regression rank against its XGBoost SHAP rank. The resulting scatter plot shows a fair concentration of points along the 45‑degree diagonal, indicating areas of agreement, while the spread of off‑diagonal points highlights where the two models diverge in their feature rankings.

The color‑coded points in the figure further clarify the structure of agreement and disagreement:

```{=latex}
\begin{itemize}
\item \textcolor{greenpoint}{\textbf{Green}} points features ranked in the top eight by both models---cluster tightly near the diagonal and represent the strongest areas of cross‑model alignment.
\item \textcolor{bluepoint}{\textbf{Blue}} points show small deviations from the diagonal, reflecting modest rank differences and generally consistent behavior across models.
\item \textcolor{amber}{\textbf{Orange}} points sit farther from the diagonal, indicating moderate disagreement in how the models prioritize these features.
\item \textcolor{red}{\textbf{Red}} points notably token count, verb count, and avg\_word\_len---fall well outside the diagonal corridor and mark the largest rank discrepancies.
\end{itemize}
```

```{=latex}
\FloatBarrier
\begin{figure}[H]
\centering
{\large \textbf{Figure 5.1: Cross-Model Feature Rank Comparison}}\

\includegraphics[width=1.05\textwidth]{fig_lr_shap_rank_distance_colored.png}
\label{fig:lr-shap-rank}
\end{figure}
```  


## 5.5 Feature Effect for Shared Top Features

To understand how the shared top features influence predictions, we compare the marginal effects implied by both models for the four features they jointly prioritize: unique lemmas, punctuation count, character count, and avg_dep_depth. XGBoost’s SHAP‑dependence plots and Logistic Regression’s $X\beta$ contribution plots provide complementary views of these effects. SHAP values capture nonlinear, context‑dependent effects, while $X\beta$ values represent strictly linear log‑odds contributions. Because these arise from different modeling frameworks, their y‑axes are not directly comparable, so we focus on the direction and overall trend of each feature’s effect rather than its absolute magnitude. 

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
Across the four shared features, three show broadly similar directional patterns, while one exhibits a clear difference:

```{=latex}

\begin{itemize}

\item For \textbf{unique lemmas} and \textbf{punctuation count}, both models show positive contributions, with XGBoost capturing gentle curvature and Logistic Regression applying constant linear increases. 
\item For \textbf{character count}, XGBoost displays an initial rise, followed by a slight decline and a plateau. Logistic Regression, in contrast, applies a steady negative effect across the entire range, implying that shorter reviews always receive more engagement—a counter‑intuitive pattern because very short reviews rarely provide enough information to be useful. The XGBoost pattern is more reasonable: it indicates that reviews must reach a basic level of detail before becoming helpful, after which additional length yields diminishing returns.
\item For \textbf{avg\_dep\_depth}, this feature reflects the syntactic complexity of a review by measuring how deeply words sit within the grammatical structure of each sentence, distinguishing simple, flat constructions from more layered ones. XGBoost shows a clear upward trend in the middle of the range, while the low and high ends remain relatively flat and noisy. Logistic Regression, by contrast, produces a steady linear rise that fails to capture the tail behavior at either end. The XGBoost pattern is more plausible: syntactic complexity seems to matter only once a review reaches moderate structural depth, after which additional layering provides limited benefit or even become harmful.

\end{itemize}
```

Overall, these patterns show that XGBoost captures the underlying feature–outcome relationships more faithfully. For features with largely monotonic behavior—unique lemmas and punctuation count —both models produce similar directional signals. In contrast, features with more complex or threshold‑based patterns, particularly character count, highlight the limitations of the linear specification, which tends to oversimplify and can misrepresent the true marginal effect.

## 5.6 Feature Rank by Controlling Review Year 

Because Yelp reviews span 2005–2022 and the platform does not specify when *useful*, *funny*, and *cool* votes were last aggregated, we treat engagement counts as a 2022 snapshot rather than contemporaneous measurements. Older reviews have had many years to accumulate validation, while newer reviews—especially those from 2019–2022—may not yet reflect their eventual engagement levels. Ideally, engagement counts would be standardized to a fixed window (e.g., two years after each review is written), but such information is unavailable. To control for review age within the constraints of the dataset, we segment the reviews into three balanced groups: Group 1 ($\le$ 2015), Group 2 (2016–2018), and Group 3 (2019–2022), containing 79,100, 80,999, and 67,344 reviews respectively.

We retrain the model within each segment and compare segment‑specific SHAP feature ranks to the overall ranking. Figure 5.4 summarizes these comparisons:

* Group 1 ($\le$ 2015) — Points lie close to the diagonal, showing strong alignment with the overall ranking.

* Group 2 (2016–2018) —  Similar tight clustering, indicating stable feature importance across mid‑period reviews.

* Group 3 (2019–2022) — Wider spread and greater deviation from the diagonal, likely due to: limited time for recent reviews to accumulate votes, newer or more volatile businesses producing noisier engagement, and platform‑level shifts (e.g., changing voting behavior, pandemic‑era effects).

Overall, the linguistic predictors of engagement appear stable over time, with recent‑period deviations likely reflecting data‑collection dynamics rather than definitive shifts in underlying feature importance.
```{=latex}
\begin{figure}[H]
\centering
{\large \textbf{Figure 5.4:  Overall and Segmented Feature Ranks}}\vspace{-0.3em}
\includegraphics[width=0.9\textwidth]{feature_rank_overall_vs_groups_2x2_overlay.png}
\label{fig:overall-segmented-rank}
\end{figure}
```  

# 6. Conclusion and Discussion

## 6.1 Interpretation and Implications 

This study examines which linguistic features relate most strongly to engagement in Yelp restaurant reviews. Because the feature set was intentionally designed to be meaningful and interpretable—capturing lexical diversity, readability, syntactic structure, and formatting—we can directly assess how specific aspects of writing style relate to perceived helpfulness. The two modeling approaches highlight the same core signals, with Logistic Regression imposing strictly monotonic relationships and XGBoost capturing mild nonlinearities, thresholds, and plateau effects.

For features that behave roughly monotonically—unique lemmas, punctuation count, syntactic depth—both models show similar directional patterns, suggesting straightforward and interpretable effects. For features with more complex structure, particularly character count, the linear model compresses the marginal effect, whereas XGBoost adapts to nonlinearities and provides a more faithful representation of how linguistic attributes shape engagement.

The temporal segmentation analysis reinforces this stability. Reviews written before 2019 (Groups 1 and 2) show feature‑importance rankings that closely match the overall pattern, indicating that once reviews have had time to accumulate votes, the relationship between linguistic cues and perceived helpfulness is highly consistent. More recent reviews (Group 3) show greater variability, likely tied to limited exposure rather than changes in how readers respond to writing style. This highlights a practical data‑collection issue: because engagement accumulates over time, analyses based on a single snapshot can introduce bias, especially for newer reviews. Future work should account for this exposure‑time effect, for example by using a fixed observation window or excluding reviews that have not yet had time to gather votes.

The results point to a clear pattern: reviews that are detailed, varied, and structurally coherent tend to resonate more strongly with readers. The stability of the influential features across analyses suggests that writing style offers a reliable basis for understanding perceived helpfulness in user‑generated content. These findings also have practical implications. Reviewers can use these signals to write clearer and more informative reviews, and platforms can draw on linguistic cues to identify potentially helpful content when engagement signals are sparse.

## 6.2 Limitations 

While the findings are robust, several limitations shape their interpretation:

* \textbf{Engagement as an imperfect proxy for perceived helpfulness}  
Engagement counts reflect not only writing quality but also visibility, timing, and platform dynamics. Well‑written reviews may receive few votes simply because they were not widely seen.

* \textbf{Temporal incompleteness in recent reviews}  
As shown in Section 5, reviews from 2019–2022 have had less time to accumulate votes, introducing noise and making their engagement levels less comparable to older reviews.

* \textbf{Restricted domain (restaurant reviews only)}  
Focusing on restaurants reduces domain‑driven linguistic variation but limits generalizability to other Yelp categories or platforms.

* \textbf{Linguistic features do not fully control for contextual confounding}  
Although the research question is linguistic, reviewer‑level and business‑level factors may still confound the relationship between writing style and engagement. These factors were not explicitly controlled for in the present analysis.

These limitations motivate several directions for future work that preserve the linguistic focus while strengthening interpretability and generalizability.


## 6.3 Future Work 

Future work can extend this study by addressing the limitations above while maintaining a clear linguistic focus:

* \textbf{Addressing engagement as an imperfect proxy}  
Incorporate additional control variables—such as reviewer reputation, business attributes, or exposure‑related signals—to better isolate linguistic effects without shifting the research question away from writing style.

* \textbf{Improving temporal comparability of engagement counts}  
The variability observed in recent reviews suggests the need for fixed engagement‑measurement windows (e.g., two years after publication). A longitudinal design would reduce temporal confounding and allow cleaner comparisons of linguistic effects across review age.

* \textbf{Expanding segmentation to test stability of linguistic effects}  
Building on the three‑group temporal segmentation in Section 5, future work could explore finer temporal bins, domain‑specific segments, or business‑type segments to assess whether linguistic feature importance remains stable across different contextual slices of the dataset.

* \textbf{Enhancing linguistic feature coverage while staying within the linguistic domain} 
Additional linguistic indicators—such as discourse structure, coherence markers, rhetorical devices, or transformer‑based linguistic embeddings—could capture higher‑order writing patterns beyond the lexical and syntactic features analyzed here.

* \textbf{Assessing generalizability across domains while preserving linguistic focus} 
Applying the same methodology to other Yelp categories or platforms would clarify whether linguistic predictors of engagement behave similarly across domains, provided domain differences are explicitly modeled rather than assumed away.

Together, these extensions directly address the constraints of the present study and deepen understanding of how readers evaluate user‑generated content while keeping the research question centered on linguistic structure.


# References

[1] Jindal, N., & Liu, B. (2008). Opinion spam and analysis. *Proceedings of the ACM International Conference on Web Search and Data Mining (WSDM)*.

[2] Mihalcea, R., & Strapparava, C. (2009). The lie detector: Explorations in the automatic recognition of deceptive language. *Proceedings of ACL‑IJCNLP*.

[3] Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. *Proceedings of the Association for Computational Linguistics (ACL)*.

[4] Feng, S., Banerjee, R., & Choi, Y. (2012). Syntactic stylometry for deception detection. *Proceedings of the Association for Computational Linguistics (ACL)*.

[5] Mukherjee, A., Venkataraman, V., Liu, B., & Glance, N. (2013). What Yelp fake review filter might be doing? *Proceedings of the AAAI International Conference on Web and Social Media (ICWSM)*.

[6] Kim, S., Pantel, P., Chklovski, T., & Pennacchiotti, M. (2006). Automatically assessing review helpfulness. *Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

[7] Ghose, A., & Ipeirotis, P. G. (2011). Estimating the helpfulness and economic impact of product reviews. *IEEE Transactions on Knowledge and Data Engineering*, 23(10), 1498–1512.

[8] Liu, Y., Huang, X., An, A., & Yu, X. (2008). Modeling review helpfulness on TripAdvisor. *Proceedings of the ACM Conference on Recommender Systems (RecSys)*.

[9] Danescu‑Niculescu‑Mizil, C., West, R., Jurafsky, D., Leskovec, J., & Potts, C. (2013). A computational analysis of helpfulness in community question‑answering forums. *Proceedings of the International Conference on World Wide Web (WWW)*.

