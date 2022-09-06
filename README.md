#  Interpreting non-linear text classifiers using LRP 
---
This is a project for the [Machine Learning](https://github.com/matf-ml) course on Faculty of Mathematics, University of Belgrade.
This project is based on [^f1].

### About the project
---
When using machine learning models for classification, we often want to interpret the result - explain the decision our model has made.
Our goal is to understand (quantify) how different components of the input affected the output (based on the model).
In this project we train two models for text classification - one **SVM** model (linear) and one **CNN** model (non-linear) on the [*20 Newsgoups dataset*](http://qwone.com/~jason/20Newsgroups/).

After preprocessing the data, we use **Tf-Idf** vectorization for SVM and **CBOW (word2vec)** word embeddings for CNN.
We use pre-trained word2vec embeddings available [here](https://code.google.com/archive/p/word2vec/) (although we tried training our own).
After training the models, our goal is to assign **relevances** to words of a text document, based on how that document was classified.
In other words, we want to quantify how much each word contributed to the given classification (extract most/least relevant words).

To compute the relevances for SVM we simply map our trained weights to corresponding words.
In case of CNN, we use ***Layer-wise relevance propagation (LRP)***[^f2].

After implementing the LRP method, we compare our models in terms of classification scores and word relevances.
Although our best CNN model performs slightly worse in terms of classification scores than our SVM model,
it's interpretability is (intuitively) superior - thanks to **word2vec** and the deep layers (*convolution*, *max-pooling*).

---
## Jupyter notebooks:
- ### 01 - Data analysis
    * dataset analysis
    * initial preprocessing
    * word2vec (cbow) training (failed)
    * fetching *google word2vec* embeddings
- ### 02 - SVM
    * training SVM model
    * analysing it's performance
- ### 03 - CNN
    * training several CNN models, picking the best
    * analysing it's performance
- ### 04 - LRP
    * detailed explanaiton of the LRP method for CNN
    * implementing LRP
    * testing LRP 
- ### 05 - SVM vs CNN
    * comparing our SVM and CNN models in terms of classification scores
    * comparing our SVM and CNN models in terms of word relevances

---





Contributors: 
- [Vladan Kovacevic 1013/2021](https://github.com/vladan17059)
- [Luka Djorovic 1029/2021](https://github.com/luka19517)

# 

### Literature:
[^f1]: [Arras L, Horn F, Montavon G, Müller K-R, Samek W (2017) "What is relevant in a text document?": An interpretable machine learning approach](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181142)
[^f2]: [Montavon, Grégoire & Binder, Alexander & Lapuschkin, Sebastian & Samek, Wojciech & Müller, Klaus-Robert. (2019). Layer-Wise Relevance Propagation: An Overview] (https://iphome.hhi.de/samek/pdf/MonXAI19.pdf)

