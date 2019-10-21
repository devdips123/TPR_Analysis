# TPR_Analysis
Statistical analysis of empirical data of Translational Process Research (TPR) database. Using computational linguistics techniques to analyse the behavioral data of the translators present in the TPR database and drawing inference and correlation between several measures like word production time and Word Translation Entropy. Using corpora in English and Spanish to train word2Vec models to find semantically similar words across languages and investigate the translation difficulty.

Following are some of the computational techniques used
1. Words and segment tokenization
2. Stop word removal and removal of alpha-numeric characters, junk characters etc.
3. Building Language Models (N-gram modeling). Evaluate 2-grams and 3-grams from corpora like BNC and calculating the perplexity
4. Named entity recognition and substitution. 
5. Stemming and lemmatization
6. Implementing Levenshtein's distance to find Orthographically similar words
7. Training Word2Vec models to find Semantically similar words
8. Statistical analysis and visualization on TPR database - Linear regression models are used to show correlation between different behavioural data( Word Production time, eye tracking data) and linguistic measures (Word translation entropy). P-value, R-square tests are conducted to evaluate the efficacy of the statistical inferences.

