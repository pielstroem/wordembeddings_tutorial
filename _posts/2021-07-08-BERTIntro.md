---
title: 3. BERT - Contextualized embeddings from language models
tags: 
article_header: 
  type: cover
  image:
---



## Preparations

In this chapter, we will use the `flair` library to get contextualized *BERT* wordembeddings. **If** you have not yet installed the library, you should run

```
!pip install flair
```

Now we can load functionalities for using BERT embeddings and for sentence tokenization.

```
from flair.embeddings import BertEmbeddings
from flair.data import Sentence
```

## Loading the model

At this point, we have only imported the ability to work with BERT models. Now, we want the language model itself. Such a model is quite large. So be careful with the next line of code, it will download such a language model and that **can take several minutes**.

```
bert = BertEmbeddings('bert-base-uncased')
```

## Embedding some text

Finally, we will filter some text through the language model. The `Sentence()` function turns a chunk of text into the type of object the language model can work with.

```
sentence = Sentence('In a hole there lived a hobbit')
```

And the `embed()` method adds the emebeddings to that object.

```
bert.embed(sentence)
```


## Looking into the embeddings

Each item/word in the `sentence` object now has a `text` attribute. For example, the `text` attribute of the 6th item, i.e. the 6th word in the sentence is:

```
sentence[6].text
```


```
'hobbit'
```

And each word is also represented by a corresponding embedding vector.

```
sentence[6].embedding
```


```
tensor([-0.1592, -1.3375, -0.0391,  ..., -0.2366, -0.0568, -0.1235])
```

In our short example here, it is also possible to look at the entire sentence. In this example only the first 5 dimensions of the embedding vectors are shown.

```
for token in sentence:
  print(f"{token.text}: {token.embedding[:5]}")
```


```
In: tensor([-0.4488,  0.4056, -0.6896, -0.2846,  0.2260])
a: tensor([-0.0880, -0.0006, -0.0651,  0.2443,  0.5529])
hole: tensor([ 0.7901,  0.5116, -0.2935,  0.0651,  0.9028])
there: tensor([-0.1365, -0.0060, -0.1858,  0.0149,  0.6088])
lived: tensor([ 0.5754,  0.0102,  0.0554,  0.0383, -0.0127])
a: tensor([-0.1586,  0.1072, -0.0145,  0.0070,  0.1593])
hobbit: tensor([-0.1592, -1.3375, -0.0391, -1.4294,  1.4285])
``` 

## An example for polysemy 

One main strength of contextualized embeddings like BERT lies in their ability to deal with polysemy. If we take the sentences

> The **head** is where the brain should be.

and

> The **head** of department decided to change the curriculum.

we see that the word *head* has two different  meanings in these different contexts. When using static embeddings like *Word2Vec* or *FastText*, there is only one fixed vector representation for the word *head* in the data set. When using BERT, the language model generates embeddings for each instance of a word depending on the current context. Therefore, if we compute BERT embeddings for the word *head* in these two sentences, we get two different vector representations.

```
text1 = Sentence('The head is where the brain should be.')
text2 = Sentence('The head of department decided to change the curriculum.')
bert.embed(text1)
bert.embed(text2)
print(f"{text1[1].text}: {text1[1].embedding[:5]}")
print(f"{text2[1].text}: {text2[1].embedding[:5]}")
```


```
head: tensor([0.3794, 0.4647, 0.5184, 0.0952, 0.4101])
head: tensor([ 0.0198, -0.0975,  0.3457, -0.1001,  0.0600])
```

