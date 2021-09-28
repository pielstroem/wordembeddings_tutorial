---
title: 2. FastText - Static embeddings with sub-word information
tags: 
article_header:
  type: cover
  image:
---

Static embeddings can give as a number of 'semantic' coordinates for each word in the vocabulary. Each item in the lexicon has its specific embedding, and it can be analyzed independent of any particular context. Therefore, *FastText* embedding can be easily applied to questions like

> Which word is similar to *X*?

## Preparations

In this chapter, we will use the `gensim` library to get static *FastText* wordembeddings. **If** you have not yet installed the library, you should run

```
!pip install gensim
```

Now let us get the function we need to access the embeddings.

```
import gensim.downloader as api
```

## Loading the model

In the next step, we will load a set of embeddings for the English language from an online resource. Be careful: it means we are downloading a data set containing a vector of numbers for each and every word. It is a large file - about 1 GB in this example - and that might take a minute or two.

```
fasttext = api.load('fasttext-wiki-news-subwords-300')
```

## Entering the matrix

Now let us look what we got, what the data looks like. The `fasttext` object now contains an array of numbers for each word. The word themselves can be used as indices. To get the vector for the word *chocolate* we could just type `fasstext['chocolate']`. But right now, we do not want to see all 300 dimensions of the vector. The first five will suffice now as an example:

```
fasttext['chocolate'][:5]
```


```
array([ 4.1900e-02,  6.1037e-03, -6.1248e-02, -6.9743e-02,  3.4114e-05],
      dtype=float32)
```


## Examining semantic relations

Let us look at another example. These are first five figures in the word vector for the country Sweden:

```
fasttext['Sweden'][:5]
```


```
array([-0.051901,  0.052235, -0.018424, -0.092136,  0.016752],
      dtype=float32)
```

The `similarity` method allows us to quantify the similarity between Sweden and Denmark within this set of embeddings.

```
fasttext.similarity('Sweden', 'Denmark')
```


```
0.8104305
```

This here, in contrast, is the similarity between Sweden and Australia.

```
fasttext.similarity('Sweden', 'Australia')
```


```
0.53964645
```

It seems like this list of coordinates that is entirely based on information in texts does know that Sweden is closer to Denmark than Australia.

We can also ask for the *n* most similar items to a word in question. For example, the 5 words most similar to *Sweden* in this set of embeddings are:

```
fasttext.most_similar(positive=['Sweden'], topn=5)
```


```
[('Denmark', 0.8104305267333984),
 ('Finland', 0.8086070418357849),
 ('Norway', 0.8046689629554749),
 ('Swedens', 0.779757559299469),
 ('Stockholm', 0.7788359522819519)]
``` 

But the `most_similar` method is even more powerful. Due to the nature of vector representation, it can also be used to solve questions like 

> X relates to *king* like *woman* relates to *man*.

But instead of queens and princesses, I will use another example:

```
fasttext.most_similar(positive=['church', 'muslim'], negative=['christian'], topn=1)
```


```
[('mosque', 0.7612898349761963)]
``` 

## Vectorizing text

Accordingly, the embeddings allow us to turn a piece of text into a sequence of vector representations quite easily. Let us consider an example sentence:

```
sentence = 'In a how in the ground there lived a Hobbit' 
```

We can tokenize that sentence, i.e. turn it into a list of words, very simply.

```
sentence = sentence.split()
``` 

Of course, there are more sophisticated functions for tokenizations, but for this simple example, the one we used here will do. Now, the sentence looks like this:

```
['In', 'a', 'hole', 'in', 'the', 'ground', 'there', 'lived', 'a', 'Hobbit']
```

In the interest of keeping everything neatly together while handling multiple word vectors, we will load another library that deals with vectorized data. In this case, it will be `numpy`, but there are other option that are as good.

```
import numpy as np
```

Now, we can implement the following step. We initialize an empty list to put the word embeddings in, token by token, and finally we turn the list into an array for more elegant handling.

```
embedded_sentence = []
for word in sentence:
  embedded_sentence.append(fasttext[word])
embedded_sentence = np.array(embedded_sentence)
```

And the result is an array of word vectors.

```
embedded_sentence
```


```
array([[-0.0039593,  0.039242 ,  0.049717 , ...,  0.037519 ,  0.063416 ,
         0.041059 ],
       [-0.0079206, -0.095293 ,  0.031266 , ..., -0.0084443, -0.066989 ,
         0.025416 ],
       [ 0.091771 , -0.04083  , -0.14974  , ..., -0.059553 ,  0.016671 ,
         0.17887  ],
       ...,
       [-0.014669 , -0.054525 ,  0.048457 , ...,  0.034834 , -0.081458 ,
        -0.0019505],
       [-0.0079206, -0.095293 ,  0.031266 , ..., -0.0084443, -0.066989 ,
         0.025416 ],
       [-0.13092  ,  0.0076898, -0.11942  , ...,  0.056652 , -0.017332 ,
        -0.054481 ]], dtype=float32)
``` 
