---
title: 1. How do wordembeddings work? An Overview
tags: 
article_header:
  type: cover
  image:
---

## Introduction

Computers can analyze texts on many levels, but they are still not able to understand them. While lexical and syntactic problems have been modeled and relating tasks been automated successfully for years, semantics still pose a fundamental problem. Because *meaning* is concept not directly accessible for computer analysis at the moment, researchers and programmers widely use a concept called **distributional semantics** as an approximation. The idea behind that concept: words that are semantically related should often be found together. Distributional semantics are therefore based on statistics about word co-occurrences.

A wordembedding is a **vector space representation** of a word, based co-occurrence statistics. It represents the word as a point in an *n*-dimensional space. In that space, semantically related words are positioned closer together than those that have nothing to do with other. There are many ways to compute embeddings. The simplest ones are based on the entire text as the level of co-occurrence and are known for many years. The more sophisticated ones are based on intricate **deep-learning** models.

Representing words with semantic embeddings opens many new possibilities for text analysis. Beyond direct uses like the quantification of semantic chance, embeddings are often used as input for upstream tasks, e.g. to classify texts according to content-related features.

In recent years this technology has advanced impressively. It revolutionized the world of natural language processing and opened up new ways of analyzing the content of texts automatically. Semantic applications like *sentiments analysis* (the quantification of the emotions encoded in a text) or *hate speech detection* saw drastic improvements.

Wordembedding technologies are currently, as of 2021, still developing at a breathtaking pace. This tutorial can only provide a snapshot view. It aims to provide an introduction for interested users from a digital humanities context. It will begin in this chapter with a quick introduction to the fundamental ideas and concepts, and their history. Later on, it will demonstrate how to compute semantic representations from two different, currently widely popular embedding systems, *FastText* and *BERT*, both with their own advantages and disadvantages.


## LSA - Semantic representation through distributional semantics

**Latent Semantic Analysis** (LSA, Deerwester et al. 1988) has been used in information retrieval since 1990. It is based on a so-called *document-term matrix*, a matrix indicating how often each word occurs in each document. Theoretically, this matrix in itself can already be seen as a vector representation of the words: each word is represented by a vector as long as the number of documents, and if we located the word in this very high-dimensional space we would probably find related words in its proximity.

The problem is dimensionality. Statistics get more reliable with more measurements, in this case that means more texts and more dimensions. To keep the matrix manageable, LSA applies a mathematical transformation technique called **singular value decomposition**. This technique finds new coordinates for each word while preserving the relative positions of the words to each other and maximizes the variability of vector values in a few of the new dimensions. As a consequence, the original vector dimensions (i.e. the texts) are replaced by more abstract ones, and very few of these are important and represent most of the variability of the matrix. Why is that useful? All the less important dimensions can be ignored and the word represented with a shorter vector without sacrificing much information. Singular value decomposition is used for **dimensionality reduction**.

**Advantages and disadvantages**

- LSA allows to compute corpus-specific word embeddings from a limited number of texts quite easily. 

- The procedure is **computationally intense**, especially with many texts, but many texts should be used for good results.

- As each type in the vocabulary is represented by one vector, therefore cases of **polysemy** are a fundamental problem.
 

## Word2Vec - Big-data machine learning for semantic representations

LSA Can be described an unsupervised machine learning approach. It is unsupervised because it has no ground truth to learn from. On the contrary, in supervised machine learning, the algorithm learns from training data containing known *correct* responses. The idea behind **Word2Vec** is to use existing text as training material in supervised machine learning to **train** wordembeddings. Compared to LSA this training approach is more scalable. In LSA, more data input automatically results in a larger matrix that has to be handled by a computer's memory. In Word2Vec, training examples are added sequentially, so it is mainly a question of time if a lot of data is added. And in the age of the internet, training data is available in abundance, even more so as Word2Vec uses short text snippets rather than coherent documents as training material. This constitutes a **big data**-compatible approach to wordembeddings.

The algorithm behind Word2Vec can be described as an artificial neural network with a single hidden layer, or as a regression-based prediction model with multiple parameters. The model is trained to do two things: (1) to predict a word if the five words before and after it are known and (2) to predict the five words before and after a given word. After fitting the model to the training data (the more, the better), each word is represented by a vector of 300 dimensions.

All the compelling results produced with is it aside, Word2Vec was the first procedure offering something very compelling to many users: you can download and use readily available embeddings others have trained on millions of examples. On the downside, if you download a set of embeddings, and you need to analyze a word that is not part of the set, there is no way to find a representation for it.

**Advantages and disadvantages**

- Significant improvement in **performance** as compared to LSA.

- **Pre-trained** embeddings can be downloaded and used, available for many languages.

- If working with pre-trained embeddings, there is no way to represent **out-of-vocabulary words**.

- As each type in the vocabulary is represented by one vector, **polysemy** still is a fundamental problem.



## FastText - Representing unknown words by using sub-word units 

The problem with out-of-vocabulary words can be very severe for some applications. Especially in humanities projects, words unknown to a pre-trained language model are encountered frequently, as these can include exotic inflections of known words, outdated orthographic versions and word creations. Furthermore, literary and historical texts can represent an earlier stage of a modern language not accounted for when the model was trained. To add to complication, humanists often deal with so-called *low-resource* domains, domains of language with simply too little surviving (digitized) text material to train a specific word embedding on.

FastText (Bojanowski et al. 2016) addresses the problem of out-of-vocabulary words with a more complex approach to word representation. In Word2Vec the *word* is the principal unit of training. FastText instead works on the basis of **subword units** or **n-grams** and models each word as a combinations of subword units. As a consequence, semantic meaning encoded in sub-word morphemes can influence model training and embeddings can be computed for out-of-vocabulary words based on the n-grams composing them.

**Advantages and disadvantages**

- **Pre-trained** embeddings can be downloaded and used, available for many languages.

- No issues with **out-of-vocabulary words**.

- **Polysemy** is still a problem.


## BERT - Contextualized embeddings from complex language models

The solution for the polysemy problem is provided by **contextualized** embeddings. Rather than representing each instance of the same word with the same vector, like the **static** embeddings introduced before do, they provide an individual vector for each and every new token depending on its current context. If we want to start working with context-dependent embeddings, we do not download a set of words or sub-word units and their respective embedding vectors. What we download is a vast statistical model with an enormous number of parameters, trained on an even larger number of examples, that allows us to generate vector representations *on-the-fly* for any word in a given text.

The model behind this is an **artificial neural network** with a highly complex architecture. Currently the most popular example for that type of embedding model is **BERT** (Bidirectional Encoder Representation from Transformers; Devlin et al. 2018). It was not the first system applying language models based on **deep learning** (complex artificial neural networks with multiple hidden layers) to compute wordembeddings, but it was the first to adopt a network architecture known as **bidirectional transformers** that represents the current *state-of-the-art* in terms of downstream-task performance. Accordingly, many currently used models, even though having seen several incremental improvements, are considered belonging to the BERT family.

A BERT model is trained on two tasks. In the **masked language modeling** task words in the training texts are masked randomly, and the model is trained to predict the masked words. In the **next sentence prediction** task, the model is presented with two sequences of text and it has to predict if the second one is the correct continuation to the first one.

BERT models are often downloaded and then further trained with new, task specific training data by their users. This **fine tuning** step can be useful to adapt the model to a specific domain (e.g. lyric language) underrepresented in the original training set, or to optimize it for a specific task. The neural network architecture can be expanded with additional layers to produce alternative outputs, like classifications or sentiment scores, and the entire model can be optimized for the new task by fine tuning.

To deal with out-of-vocabulary words BERT also uses a tokenizer that looks at **sub-word units** and models unknown words as a combination of n-grams.

**Advantages and disadvantages**

- High **performance** in downstream tasks.

- Models can be **fine-tuned** for specific applications.

- BERT **can handle polysemy**.

- **Contextualized** embeddings do not represent a word in general, only one specific instance.

## What you need to run the example code

To reproduce the example code shown here a distribution of `Python 3.x` is required. Some operating systems come with a pre-installed `Python` on board. It is however not recommended to use the system's `Python` for data science tasks, as messing around with its packages can interfere with system functionalities that rely on `Python`. The way to go is to install an independent instance and, in the best case, work with virtual environments. Explaining virtual environments in detail is beyond the scope of this tutorial, but those completely new to the topic might consider one of the following solutions widely relied on by people doing data analysis:

- [Anaconda](https://www.anaconda.com/) is a `Python` that can be easily installed on Windows systems (installers for MacOS and Linux are also available), comes with all packages needed for data analysis and even brings its own capabilities for working in virtual environment. 
- On Unix systems, many people rely on [PyEnv](https://github.com/pyenv/pyenv) to create and manage `Python` systems and virtual environments.
- If you are familiar with [Docker](https://www.docker.com/) and unwilling to mess around with the intricacies of managing your `Python` environment, you can simply run your [Machine Learning Workspace](https://github.com/ml-tooling/ml-workspace) in Docker and have everything you need there. Why can you use a 'Machine Learning Workspace' to do statistics? Well, many machine learning algorithms really are just statistical procedures put to a slightly alternative use. 
- Probably the easiest solution is not to use any `Python` on your computer at all, but to work on a server that has everything installed for you. [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) for example offers that service free of charge, but you need Google account to use it productively. 

## Acknowledgements
This tutorial has been deveoped as part of project [CLARIAH-DE](https://www.clariah.de/), which was funded by the German [Federal Ministry of Education and Research](https://www.bmbf.de/bmbf/en/home/home_node.html).

## References

- Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov: Enriching Word Vectors with Subword Information. (2016).
- Scott Deerwester, Susan T. Dumais, Richard Harshman: Indexing by Latent Semantic Analysis. (1988).
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova: BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding. (2018).
- Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean: Efficient Estimation of Word Representations in Vector Space. (2013).
