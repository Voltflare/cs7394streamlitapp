# The GLOVE Word Embedding Visualization:

GloVe is a tool for generating models for word cooccurrences that maps words into a vector space grouped by semantic similarities. (Ie: "ball" and "throw" would be much closer (as balls are very often thrown objects and thus appear in sentences commonly) than "ball" and "spaceship"). As the computer is not able to understand these similarities without a model relating them, GloVe is used to allow computers to "learn" these semantics for use in various word analysis tasks. Particularly, GloVe was used in the human trafficking prosecution space for research, of which my thesis intersects.

As explained in the https://github.com/stanfordnlp/GloVe official GitHub for GloVe: GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 

There are pre-trained word vectors also available, as the amount of time required to train a model is very large. Thus, most machine learning experiements either devote large amounts of parallelized computing power to training, or use these pre-trained word vectors. These are the most available GloVe pre-trained word vectors:

Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download) (commonly called Glove.6B.{vectorsize}.txt)
Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download) (Glove.42B.300d.txt)
Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) (Glove.840B.300d.txt)

A demo case of the GLoVe training, as well as a copy of GloVe's source code, is found at the following url. The entire package is too large for GitHub to store, so it is currently housed in a Google Drive. From the GitHub: "The (contained) demo.sh script downloads a small corpus, consisting of the first 100M characters of Wikipedia. It collects unigram counts, constructs and shuffles cooccurrence data, and trains a simple version of the GloVe model." 
https://drive.google.com/file/d/15-p7nT4E88khab8IDK43i7jCG8nXTjEm/view?usp=sharing

Several files are created and used during the training of the model:

- The /build directory: these are the files required to run GloVe and are built as needed
- The /eval directory: this directory contains files used to test the accuracy of the GloVe model. This is typically used for different use cases than simply visualizing these word vectors.
- The /src directory: the directory that contains all of the source code, that is compiled into the /build directory. The shell script that begins model training is also located in this directory.
- Several binary files: **cooccurence.bin**, **cooccurrence-shuf.bin**, and **vectors.bin**. These are binary files that are created during the training process.
  - Cooccurence binaries are part of the cooccurrence matrix- which are used to construct word-to-word cooccurence statistics from a given input file (ie: how often certain types of words appear together. This is then used to construct word vectors by the GloVe model).
  - Vectors.bin is the end result binary of the GloVe model. It is not in parseable format by most contemporary programs, and is thus fitted into a more-easily-parseable **vectors.txt**. This vectors.txt contains the trained word vectors (embeddings) for the GloVe model.


# Word Embedding Visualization

From: https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5, code slightly adapted to run on different hardware.

"This is a repo to visualize the word embedding in 2D or 3D with either Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE).

Below is the snapshot of the web app to visualize the word embedding."

<p align="center">
  <img width="700" height="350" src=https://github.com/marcellusruben/Word_Embedding_Visualization/blob/master/word_embedding_gif.gif>
</p>

**Note: The program uses a relative pathing to find the word vector file. Place it in the same (root) directory if you wish to run the app locally.

Equipped with the trained GloVe word vectors, run the following command in the project directory.

```
python train_model.py
```

This runs gensim's (a python utility library that contains implementations of GloVe and word2vec that work with Python) model method and saves the trained model in the file **glove2word2vec_model.sav**

Finally, to execute the web app, go to the working directory of the app.py and type the following command in the conda environment:
```
streamlit run app.py
```
