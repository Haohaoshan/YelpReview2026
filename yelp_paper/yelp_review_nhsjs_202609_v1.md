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

<!--
{\Large
Alexander Shan \\
\vspace{0.1cm}
Texas Academy of Mathematics and Science \\
\vspace{0.5cm}
Angela Yin \\
\vspace{0.1cm}
Reedy High School
}
--> 

\vspace{1.5cm}


\vfill

<!--
{\small
Correspondence: \\
Alexander Shan - alsney2019@gmail.com \\
Angela Yin - angelasuyuyin@gmail.com 
}
--> 
\vspace{0.5cm}

\end{titlepage}

\fontsize{12pt}{12pt}\selectfont


# Abstract 

Online reviews influence how people choose restaurants and other services, and engagement levels on Yelp vary substantially across reviews. This study analyzes a large sample of Yelp restaurant reviews and extracts linguistic features covering sentiment, lexical diversity, readability, syntactic structure, and formatting. Logistic Regression and XGBoost models are trained to distinguish reviews that received engagement votes from those that received none, with both models reaching test AUC values near 0.71. SHAP analysis identifies review length, vocabulary size, syntactic depth, and punctuation patterns as the most influential features. A temporal comparison shows that reviews from two older periods exhibit a similar feature‑importance pattern, indicating that the associations are stable after controlling for exposure‑time differences. Because engagement also reflects visibility and platform dynamics, the results represent associations rather than causal effects, but they point to consistent linguistic characteristics linked to higher reader interaction in this dataset.


# 1. Introduction and Background 

Online reviews shape everyday decisions, from choosing a restaurant to booking a hotel. Platforms like Yelp host millions of reviews, and engagement levels vary widely across them, reflected in votes such as “useful,” “funny,” or “cool.” Understanding why certain reviews draw attention while others do not is important for both readers and platforms that rely on user‑generated content.

Research on online review quality spans several major areas:

```{=latex}

\begin{itemize}

\item \textbf{Deception detection —}  
Early work examined linguistic cues associated with fake reviews $^{1,2,3}$, later incorporating syntactic stylometry$^{4}$ and behavioral metadata$^{5}$. Although neural and transformer‑based text models have become standard tools for document classification, recent advances in review analysis have focused more on helpfulness and engagement prediction than on deception. Because Yelp’s Open Dataset does not include filtered reviews or ground‑truth deception labels, these methods cannot be directly applied here.

\item \textbf{Helpfulness prediction —}  
Foundational studies on Amazon$^{6, 7}$ show that crowd‑sourced helpfulness votes correlate with linguistic properties. More recent work introduces attention‑based models that incorporate customer‑expectation signals$^{8}$, neighbor‑aware relational frameworks$^{9}$, and and surveys that summarize modern deep‑learning approaches to helpfulness prediction$^{10}$.

\item \textbf{Linguistic and stylistic analysis —}  
A broader line of work investigates stylistic markers including part‑of‑speech patterns, syntactic complexity, vocabulary richness, and other structural indicators of writing quality. Contemporary neural models also incorporate these features: the helpfulness‑prediction approaches discussed above $^{8,9}$ explicitly leverage linguistic cues within attention‑based and relational architectures. Survey work $^{10}$ further highlights that interpretable linguistic features remain influential even as review‑quality modeling shifts toward deep learning.

\end{itemize}
``` 


Because Yelp does not provide deception labels, this study focuses instead on whether aspects of writing style help explain which reviews readers respond to and find helpful. Our research question is: 

*Can linguistic - features such as vocabulary richness, readability, sentiment, and syntactic structure - predict whether a Yelp review receives engagement from other users?*

To investigate this question, we analyze Yelp restaurant reviews using a structured set of linguistic features and two machine‑learning models. Engagement votes serve as a practical, review‑level indicator of perceived helpfulness. The goal is not simply to build predictive models, but to identify the linguistic patterns that distinguish high‑engagement reviews from those that receive no votes. These analyses identify which aspects of writing style are associated with higher engagement in this dataset. Understanding these associations is useful for both writers, who often want their reviews to be noticed, and platforms, which need additional signals when engagement votes are sparse. Because many reviews receive no votes, linguistic features offer platforms another way to identify reviews that may be helpful when direct engagement signals are missing.

In this study, the machine‑learning models are used primarily to highlight which linguistic features matter most. The focus is on writing style and its relationship to engagement, not on comparing model performance.


# 2. Dataset 
## 2.1 Overview 
The Yelp Open Dataset provides structured JSON files containing business metadata and full review text. This study only the **business** and **review** files, which supply the information needed for domain filtering, linguistic feature extraction and engagement labeling. Each review includes full text, a star rating, a timestamp and engagement metrics (useful, funny, cool) contributed by other users. These engagement signals serve as non-linguistic, crowd-sourced indicators of perceived helpfulness.

To reduce domain-driven variation, the analysis focuses exclusively on restaurant reviews. The data-processing pipeline proceeds as follows:


```{=latex}
\begin{itemize}  
\item Identify restaurant businesses by filtering the business file using category keywords.
\item Merge the business and review files to retain only reviews associated with restaurant businesses.
\item Apply a 5\% random sample to the restaurant reviews to create a tractable subset for NLP feature extraction.
\item Compute linguistic features on the sampled reviews, including lexical, structural, syntactic, and stylistic indicators.
\item Construct engagement labels (High, Low, Intermediate) to finalize the modeling dataset.
\end{itemize}
```   

Applying this pipeline yields 5,245,324 restaurant reviews that meet the filtering criteria. From this set, a 5\% random sample is drawn, producing 261,882 reviews for feature engineering and subsequent analysis. This sampled subset is large enough to support robust descriptive comparisons and predictive modeling. After linguistic feature extraction, the reviews are categorized into high‑, low‑, and intermediate‑engagement groups, forming the final labeled dataset used throughout the study. Together, these steps produce a domain‑consistent, computationally manageable dataset that is fully annotated with both linguistic features and engagement labels.

Because engagement votes accumulate over time, newer reviews may appear low‑engagement simply because they have had less opportunity to receive votes. To account for this exposure‑time bias, we conduct a temporal segmentation analysis to examine whether the linguistic patterns identified in our study remain stable across different temporal periods. The results show that the influential features persist among older reviews and partially persist in the most recent reviews, indicating that the main drivers of high engagement are largely the identified linguistic features rather than differences in exposure time.

## 2.2 Sampling

Although the Yelp Open Dataset contains millions of restaurant reviews, analyzing the full corpus is computationally unnecessary for the goals of this study. A 5\% random sample of the filtered restaurant‑review set (261,882 reviews) is used to balance computational efficiency with statistical power. Random sampling preserves variation in review length, diversity in writing style, distribution of star ratings, and the natural frequency of engagement signals, ensuring that linguistic patterns observed in the sample generalize to the broader population of Yelp restaurant reviews. The 5% sampling rate was selected to make the feature‑engineering stage computationally manageable. Generating linguistic features for millions of reviews requires substantial processing time, especially when using NLP tools such as spaCy and textstat on a standard home computer. Several sampling percentages were tested, and 5% provided a practical balance: the resulting dataset of over 260,000 reviews is large enough to preserve linguistic diversity and support reliable analysis, while keeping preprocessing and feature extraction within a reasonable runtime of a few hours.  


## 2.3 Label Construction

The Yelp Open Dataset does not provide an explicit engagement label, so we construct one using the three available engagement fields: useful, funny, and cool. Reviews are assigned to High‑, Low‑, or Intermediate‑engagement categories based on the following criteria:

```{=latex}
\begin{itemize}  
\item \textbf{High-engagement}: useful >= 2 OR funny >= 1 OR cool >= 1
\item \textbf{Low-engagement}: useful = 0 AND funny = 0 AND cool = 0  
\item \textbf{Intermediate}: all remaining cases  
\end{itemize}
```     

These categories capture how other users responded to a review and serve as a proxy for perceived helpfulness. The thresholds are chosen to separate reviews that received clear, non‑trivial engagement from those that received none. A cutoff of two “useful” votes reduces noise from single‑vote cases, which often reflect minimal or incidental interaction. In contrast, “funny” and “cool” votes occur less frequently overall, so a threshold of one is sufficient to indicate meaningful attention. The asymmetric thresholds therefore reflect the different usage patterns of the three engagement signals and produce a more stable distinction between high‑ and low‑engagement reviews. Intermediate cases are excluded because their engagement levels are ambiguous and do not provide a clear contrast for modeling.

## 2.4 Train–Test Split for Modeling
  
After constructing the labeled restaurant‑review dataset, an 80/20 train–test split is applied to support predictive modeling. The training set is used to fit all models, while the held‑out test set provides an unbiased estimate of predictive performance and supports SHAP‑based interpretability analysis.


# 3. Methodology 

This study investigates whether linguistic features can distinguish high-engagement from low-engagement Yelp reviews. The methodology integrates standardized text preprocessing, structured linguistic feature engineering, and interpretable machine-learning models, all applied within the restaurant review domain to maintain linguistic consistency and avoid cross‑domain confounds.


## 3.1 Preprocessing

A minimal preprocessing pipeline is applied to all review text to ensure consistency while preserving the full linguistic structure required for feature extraction. Each review is lowercased, stripped of leading and trailing whitespace, and normalized so that repeated whitespace is collapsed into a single space. No punctuation or stop words are removed, and no additional text transformations are applied. The normalized text is then passed directly into spaCy, and all linguistic features—lexical, syntactic, readability, structural, sentiment, and semantic—are computed from the resulting document.


## 3.2 Linguistic Feature Engineering

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


Several features in the tables may appear similar—for example, counting nouns versus computing the proportion of nouns—but they capture different aspects of language. Raw counts describe structural properties of a review, while normalized ratios reflect meaning‑oriented characteristics such as concreteness or action orientation. The feature set was designed to capture observable aspects of writing style—such as vocabulary variety, syntactic depth, readability, and formatting—using measures that can be directly interpreted in terms of linguistic behavior. We avoid representations that would be difficult to explain in this context, such as high‑dimensional embeddings. These features form the input to the models described in Section 4.3.


## 3.3 Models 

This study uses two supervised learning models—Logistic Regression and XGBoost—to evaluate whether linguistic features contain predictive information about engagement. Although both models use the same feature set, they differ in how they process input features and represent relationships among them.

Logistic Regression provides a linear baseline with feature‑level coefficients. Because the model optimizes a linear decision boundary, all linguistic features are standardized using a StandardScaler prior to training. Standardization ensures stable coefficient estimation and allows the coefficients to reflect the relative influence of each feature on the predicted probability that a review receives engagement.
Several structural and lexical features — such as token count, character count, sentence count, and unique lemma count — are naturally correlated because they all increase with review length. Logistic Regression therefore distributes the underlying length signal across multiple coefficients, which limits the extent to which individual count‑based coefficients can be interpreted as independent effects. In this study, Logistic Regression serves primarily as a linear baseline and a directional check on feature influence, while XGBoost and SHAP provide the main basis for fine‑grained attribution.

XGBoost is used to capture nonlinear relationships and interactions among linguistic features. As a tree‑based ensemble method, it determines splits based on the ordering of feature values rather than their absolute magnitudes, so feature scaling is unnecessary. Operating on raw features preserves natural variation and avoids superfluous preprocessing. XGBoost also supports SHAP‑based interpretation, which provides structured, sample‑level attributions for how individual features influence predictions. SHAP, introduced by Lundberg \& Lee $^{11}$, offers a unified, model‑agnostic framework for computing Shapley‑value explanations and enables transparent interpretation of tree‑based models through the efficient TreeSHAP algorithm.

Model training is performed on reviews labeled as either high‑engagement or low‑engagement; intermediate reviews are excluded as described in Section 5.1. After filtering, the dataset is split into training and test sets using an 80/20 ratio. Although this reduces the size of the modeling dataset relative to the original 5% sample (261,882 reviews), the resulting subsets remain sufficiently large for reliable estimation.

Hyperparameters for XGBoost are selected manually rather than through automated tuning. The chosen configuration—300 trees, maximum depth of 6, learning rate of 0.05, and moderate subsampling and column sampling—was identified through iterative experimentation and provides stable performance without excessive model complexity. Extensive hyperparameter optimization was not pursued because the primary goal is to interpret how linguistic features relate to engagement rather than to maximize predictive accuracy. Logistic Regression uses standard regularization settings and serves as a baseline model.

Model performance is assessed using accuracy, precision, recall, F1‑score, and area under the ROC curve (AUC), which provide a comprehensive view of predictive effectiveness and enable comparison between linear and nonlinear modeling approaches.


## 3.4 Model Interpretation 

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



# 4. Results 

## 4.1 Descriptive Linguistic Patterns

We summarize the distribution of engagement labels by categorizing each review into low‑engagement (0), intermediate (1), and high‑engagement (2). Intermediate reviews are reported for completeness but excluded from all modeling and interpretability analysis. All descriptive statistics in later sections therefore reflect only the high‑ and low‑engagement subsets.


**Table 4.1: Sample Overview**   
    
|Engagement Category |Count   | Percent| Included for modeling|
|----------------|---------------|------------|------------|
|low-engagement	 | 134,436  | 51.3% | 59.1% |
|high-engagement | 93,007   | 35.5% | 40.9% |
|intermediate    | 34,439   | 13.2% |excluded|
|**Total**       | 261,882  | 100.0%| 100.0%| 

**Table 4.2: Training and Test Sample Statistics**  
  
|Sample       |Engagement Category |Count   | Percent|
|-------------|----------------|------------|--------|
|Training     | low-engagement | 107,549    | 59.1% | 
|             | high-engagement| 74,405     | 40.9% |
|Test         | low-engagement | 26,887     | 59.1% |  
|             | high-engagement| 18,602     | 40.9% |


With the dataset defined, we examine raw linguistic differences between high‑ and low‑engagement reviews. Clear, systematic contrasts emerge across lexical, structural, syntactic, and stylistic dimensions.

```{=latex}
\begin{itemize}  

\item \textbf{High‑engagement reviews are substantially longer and content-rich}.  
They contain 311 more characters, 57 more tokens, 31 more unique tokens, and 29 more unique lemmas on average—each exceeding 50\% relative to the corresponding low‑engagement measures. These differences indicate that high‑engagement reviews provide more specific and elaborated content. Structural indicators reinforce this pattern: high‑engagement reviews include 3.66 additional sentences (57.61\%), exhibit longer average sentence length (11.70\%), and show higher burstiness (18.77\%), suggesting more developed narrative flow and greater topical elaboration.

\item \textbf{Syntactic features show strong divergence}.  
High‑engagement reviews use more nouns (+11.9), verbs (+7.3), adjectives (+4.9), and adverbs (+4.0)—each exceeding 50\% relative to the corresponding low‑engagement measures—reflecting denser descriptive content. Syntactic complexity increases as well: average dependency depth rises from 2.16 to 2.35 (8.85\%), indicating deeper syntactic structures and more elaborated phrasing.

\item \textbf{Stylistic markers point to greater expressive intensity}.  
High‑engagement reviews contain 8 additional punctuation marks, suggesting more emphasis, expressiveness, or rhetorical variation. Sentiment differences are modest but directionally consistent: the sentiment proxy increases from 0.68 to 0.70 (2.89\%), indicating a slightly more balanced or positive tone.

\item \textbf{Readability differences are small but directionally consistent}.  
High‑engagement reviews show slightly higher ARI, SMOG, and Dale–Chall scores, indicating marginally more complex writing. These shifts are minor compared to the pronounced structural, lexical, and syntactic contrasts observed above.

\end{itemize}
```  

High‑engagement reviews are noticeably longer and contain more tokens, which suggests that length may play an important role in how readers respond to a review.
However, length alone may not fully explain the differences between the two groups.
Although length is a meaningful linguistic characteristic, many structural and lexical features scale with review size, so part of their influence may reflect underlying length differences rather than independent linguistic effects.  



**Table 4.3: Mean Differences between Engagement Groups**   

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


## 4.2 Model Performance 

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


**Table 4.4: Logistic Regressiion and XGB Model Performance**   
  
|Model                 |	Accuracy |	Precision |	Recall	|F1 Score |	**AUC** |
|----------------------|-----------|----------|---------|---------|--------|
|XGBoost	|\textcolor{blue} {0.67080}	  |0.63279	|\textcolor{blue} {0.46457}	|\textcolor{blue}{0.53579}	| \textcolor{blue}{0.70648}  |
|Logistic Regression	|0.67054	| \textcolor{blue}{0.64299}	|0.43694	|0.52031	|0.70505|  


To contextualize these results, we summarize the main performance patterns observed across the two models:


```{=latex}
\begin{itemize}

\item \textbf{Overall performance} - Both models achieve similar accuracy and AUC, indicating they capture the same underlying signals. 

\item \textbf{Logistic Regression} - Higher precision and lower recall, reflecting a more conservative boundary that flags fewer higher-engagement reviews but with greater accuracy.

\item \textbf{XGBoost} - Higher recall and lower precision, identifying more potentially engaging reviews while tolerating additional false positives due to its nonlinear model capacity.  

\item \textbf{F1 score} - XGBoost`s slightly higher F1 score shows a better precision-recall balance.

\item \textbf{Summary} - Performance is broadly similar, but XGBoost offers modest gains in detecting high-engagement reviews, whereas Logistic Regression provides more selective, precision-oriented predictions.  

\end{itemize}
```

## 4.3 Top Features Identified by XGBoost  

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


**Table 4.5: Cross-Model Feature Importance Rank Comparison** 

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

## 4.4 Cross-Model Rank Agreement

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

## 4.5 Feature Effect for Shared Top Features

To examine how the shared top features influence predictions, we compare the marginal effects implied by both models for the four features they jointly prioritize: unique lemmas, punctuation count, character count, and avg_dep_depth. XGBoost’s SHAP‑dependence plots and Logistic Regression’s $X\beta$ contribution plots provide complementary views of these effects. SHAP values capture nonlinear, context‑dependent effects, while $X\beta$ values represent strictly linear log‑odds contributions. Because these arise from different modeling frameworks, their y‑axes are not directly comparable, so we focus on the direction and overall trend of each feature’s effect rather than its absolute magnitude. 

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
\item For \textbf{character count}, the XGBoost SHAP dependence plot shows that the model assigns larger positive contributions to shorter reviews as length increases, after which the marginal effect levels off. This pattern reflects how the model uses character count in combination with correlated features such as token count, unique lemmas, and sentence count, rather than a causal threshold in review helpfulness.
\item For \textbf{avg\_dep\_depth}, the SHAP plot shows a similar leveling pattern: increases in dependency depth correspond to higher SHAP values up to a moderate range, after which the marginal contribution stabilizes. This behavior reflects the fitted model’s conditional associations and should not be interpreted as evidence that additional syntactic complexity becomes detrimental.

\end{itemize}
```
These plots illustrate how XGBoost models nonlinear associations among correlated linguistic features, even though its overall predictive performance is similar to the logistic model.

## 4.6 Feature Rank Stability Across Review Years 

Because Yelp reviews span 2005–2022 and the platform does not specify when *useful*, *funny*, and *cool* votes were last aggregated, we treat engagement counts as a 2022 snapshot rather than contemporaneous measurements. Older reviews have had many years to accumulate validation, while newer reviews—especially those from 2019–2022—may not yet reflect their eventual engagement levels. Ideally, engagement counts would be standardized to a fixed window (e.g., two years after each review is written), but such information is unavailable. To control for review age within the constraints of the dataset, we segment the reviews into three balanced groups: Group 1 ($\le$ 2015), Group 2 (2016–2018), and Group 3 (2019–2022), containing 79,100, 80,999, and 67,344 reviews respectively.

The model is retrained within each segment, and segment‑specific SHAP feature ranks are compared to the overall ranking. Figure 5.4 summarizes these comparisons:

* Group 1 ($\le$ 2015) — Points lie close to the diagonal, showing strong alignment with the overall ranking.

* Group 2 (2016–2018) —  Similar tight clustering, indicating stable feature importance across mid‑period reviews.

* Group 3 (2019–2022) — Wider spread and greater deviation from the diagonal, likely due to: limited time for recent reviews to accumulate votes, newer or more volatile businesses producing noisier engagement, and platform‑level shifts (e.g., changing voting behavior, pandemic‑era effects).

Across the three groups, the two earlier groups align closely with the full‑dataset patterns. Group 3 shows greater variability, and the available data does not allow us to determine its source; possibilities include changes in review composition, incomplete engagement accumulation, or other temporal factors. Even with this additional noise, the pattern still shows partial consistency between Group 3 and the overall ranking. The strong alignment in Groups 1 and 2, together with the partial alignment in Group 3 despite its additional noise, indicates that these linguistic associations form robust patterns in the data.
```{=latex}
\begin{figure}[H]
\centering
{\large \textbf{Figure 5.4:  Overall and Segmented Feature Ranks}}\vspace{-0.3em}
\includegraphics[width=0.9\textwidth]{feature_rank_overall_vs_groups_2x2_overlay.png}
\label{fig:overall-segmented-rank}
\end{figure}
```  

# 5. Conclusion and Discussion

## 5.1 Interpretation and Implications 

This study examines which linguistic features are most strongly associated with engagement in Yelp restaurant reviews. Because the feature set was designed to capture interpretable aspects of writing—lexical diversity, readability, syntactic structure, and formatting—the models allow us to assess how these linguistic indicators relate to engagement within the constraints of the dataset. Logistic Regression provides linear, monotonic contributions, while XGBoost captures nonlinear associations and interactions among correlated features.

For features with roughly monotonic behavior—unique lemmas, punctuation count, and syntactic depth—the two models produce similar directional patterns. For features with more complex structure, such as character count, the linear model compresses the marginal effect, whereas XGBoost reflects nonlinear associations and plateaus. These differences stem from the representational capacities of the models rather than differences in the underlying linguistic signals.

The temporal segmentation analysis shows that the two earlier groups reproduce the feature‑ranking patterns observed in the full dataset. Group 3 displays greater variability, likely reflecting incomplete engagement accumulation or shifts in review composition. Even with this noise, the segment‑specific rankings retain partial alignment with the overall pattern, indicating that the main linguistic associations persist across the temporal subsets examined.

Overall, the results show that vocabulary diversity, punctuation use, and syntactic depth are consistently associated with higher engagement in this dataset across modeling approaches and time periods. These patterns help clarify how linguistic structure relates to engagement signals in user‑generated content and may assist platforms in identifying reviews that merit additional visibility when direct engagement signals are sparse.

## 5.2 Limitations and Future Directions  

Several factors constrain how the findings should be interpreted and point to directions for future work. These considerations help build a fuller picture of how linguistic structure relates to engagement in user‑generated content.


* \textbf{Domain restriction} -  The study focuses on restaurant reviews to maintain a consistent writing domain. This improves internal validity but limits external validity. Applying the same methods to other Yelp categories or platforms would show whether the linguistic patterns generalize.

* \textbf{Temporal incompleteness} - Newer reviews have had less time to receive votes, which reduces comparability across review age. Temporal segmentation in this study shows that key feature‑importance patterns remain stable in older periods, but fixed measurement windows or longitudinal designs would further reduce exposure‑time bias. These approaches may require platform‑level data access that is not publicly available.

* \textbf{Contextual confounding} -  Reviewer and business characteristics can influence engagement independently of language. Adding controls for reviewer activity, business popularity, or review visibility would help determine whether the observed linguistic patterns persist after accounting for these factors.

* \textbf{Linguistic coverage} —  The current feature set captures core linguistic signals but not higher‑order writing structure. Embeddings and discourse‑level features were excluded for interpretability, but future work could incorporate them in a secondary analysis to provide a more complete representation of review language.


## 5.2 Limitations and Future Directions  

Several factors constrain how the findings should be interpreted and point to directions for future work. These considerations help build a fuller picture of how linguistic structure relates to engagement in user‑generated content.


* \textbf{Domain restriction} -  The study focuses on restaurant reviews to maintain a consistent writing domain. This improves internal validity but limits external validity. Applying the same methods to other Yelp categories or platforms would show whether the linguistic patterns generalize.

* \textbf{Temporal incompleteness} - Newer reviews have had less time to receive votes, which reduces comparability across review age. Temporal segmentation in this study shows that key feature‑importance patterns remain stable in older periods, but fixed measurement windows or longitudinal designs would further reduce exposure‑time bias. These approaches may require platform‑level data access that is not publicly available.

* \textbf{Contextual confounding} -  Reviewer and business characteristics can influence engagement independently of language. Adding controls for reviewer activity, business popularity, or review visibility would help determine whether the observed linguistic patterns persist after accounting for these factors.

* \textbf{Linguistic coverage} —  The current feature set captures core linguistic signals but not higher‑order writing structure. Embeddings and discourse‑level features were excluded for interpretability, but future work could incorporate them in a secondary analysis to provide a more complete representation of review language.

* \textbf{Platform evolution} — Yelp has updated its review‑interaction system since the period covered by the Yelp Open Dataset, replacing the “useful,” “funny,” and “cool” vote buttons with a different engagement interface. Because our engagement labels rely on these legacy signals, the findings reflect engagement dynamics specific to the earlier platform design. Future work could examine whether similar linguistic associations hold under Yelp’s updated interaction mechanisms or across platforms that use different feedback models.

These limitations highlight opportunities to strengthen interpretability, improve temporal comparability, and broaden the linguistic and domain scope of engagement modeling.

# Acknowledgements

We thank our research mentor, XX at Meta, for guidance throughout this project. We also acknowledge the Computer Science Lab at the University of Texas at Dallas, where Angela conducted summer research that helped shape the direction of this study. In addition, we thank the Texas Academy of Mathematics and Science (TAMS), where Alexander’s research interests were nurtured and developed, contributing to the collaborative work presented here.


# References

1. N. Jindal, B. Liu. Opinion spam and analysis. *Proceedings of the ACM International Conference on Web Search and Data Mining (WSDM)*. pg. 219-229, 2008,
https://dl.acm.org/doi/10.1145/1341531.1341560. 


2. R. Mihalcea, C. Strapparava. The lie detector: Explorations in the automatic recognition of deceptive language. *Proceedings of ACL‑IJCNLP*. pg. 309-312, 2009, 
https://aclanthology.org/P09-2078/. 

3. M. Ott, Y. Choi, C. Cardie, J.T. Hancock. Finding deceptive opinion spam by any stretch of the imagination. *Proceedings of the Association for Computational Linguistics (ACL)*. pg. 309-319, 2011,   
https://aclanthology.org/P11-1032/. 

4. S. Feng, R. Banerjee, Y. Choi. Syntactic stylometry for deception detection. *Proceedings of the Association for Computational Linguistics (ACL)*. pg. 171-175, 2012, 
https://aclanthology.org/P12-2034/. 

5. A. Mukherjee, V. Venkataraman, B. Liu, N. Glance. What Yelp fake review filter might be doing? *Proceedings of the AAAI International Conference on Web and Social Media (ICWSM)*. pg. 409-418, 2013,   
https://doi.org/10.1609/icwsm.v7i1.14389. 

6. S. Kim, P. Pantel, T. Chklovski, M. Pennacchiotti. Automatically assessing review helpfulness. *Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)*. pg. 423-430, 2006,   
https://dl.acm.org/doi/10.5555/1610075.1610135.  

7. A. Ghose, P. G. Ipeirotis. Estimating the helpfulness and economic impact of product reviews: Mining text and reviewer characteristics. *IEEE Transactions on Knowledge and Data Engineering*. Vol. 23, No. 10, pg. 1498-1512, 2011,   
https://doi.org/10.1109/TKDE.2010.188. 

8. X. Qu, X. Li, C. Farkas, J. Rose. An attention model of customer expectation to improve review helpfulness prediction. *In Advances in Information Retrieval (ECIR 2020), Lecture Notes in Computer Science*, pg. 836-851, 2020,   
https://link.springer.com/chapter/10.1007/978-3-030-45439-5_55. 

9. J. Du, J. Rong, H. Wang, Y. Zhang. Neighbor‑aware review helpfulness prediction. *Decision Support Systems*. Vol. 148, pg. 113581, 2021,   
https://doi.org/10.1016/j.dss.2021.113581.

10. S. Saumya, P. Kumar, J. Prakash. Review helpfulness prediction on e‑commerce websites: A comprehensive survey. *Engineering Applications of Artificial Intelligence*, Vol. 126, pg. 107075, 2023,   
https://doi.org/10.1016/j.engappai.2023.107075. 

11. S. Lundberg, S. Lee. A unified approach to interpreting model predictions. *Proceedings of the 31st International Conference on Neural Information Processing Systems (NeurIPS)*. pg. 4768-4777, 2017,     
https://doi.org/10.48550/arXiv.1705.07874

