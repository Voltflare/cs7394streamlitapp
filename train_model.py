# -*- coding: utf-8 -*-
"""
"""

import numpy as np

import pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from mpl_toolkits.mplot3d import Axes3D 

glove_file = datapath('C:/Users/natha/Box Sync/COLLEGE STUFFS/Spring 2022/CS 7394/Projects/StreamlitApp/vectors.txt')
word2vec_glove_file = get_tmpfile("vectors.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

filename = 'glove2word2vec_model.sav'
pickle.dump(model, open(filename, 'wb'))