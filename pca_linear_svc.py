#!/usr/bin/env python
"""
A linear SVC (SVM) classifier with suggested features and singular value decomp.

The singular value decomposition uses partial least squares.
"""
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import feature as fe

"""The number of components to keep during singular value decomposition."""
KEEP_COMPONENTS = 50

svc = OneVsRestClassifier(svm.LinearSVC())
standardizer = StandardScaler()
decomposer = RandomizedPCA(n_components=KEEP_COMPONENTS)

steps = [('randomized PCA', decomposer),
         ('standardizer',   standardizer), # hurts accuracy
         ('linear SVC',     svc)]
# TODO: Add a standardizer to the pipeline.  We can do it properly, too, since
# the SVD probably gives us a dense matrix of manageable size.

estimator = Pipeline(steps)

feature_templates = [

    # n-grams of stems
    fe.stem_ngrams_factory(1,2),

    # bag of 1,2-grams, no stemming
    # fe.ngrams_factory(1,2),

    # part-of-speech n-grams
    # fe.pos_ngrams_factory(1,4),

    # part-of-speech unigrams
    fe.pos_ngrams_factory(1,1),

    # part-of-speech bigrams
    # fe.pos_ngrams_factory(2,2),

    # part-of-speech trigrams
    # fe.pos_ngrams_factory(3,3),

    # whether currency is mentioned (US-centric for now)
    # fe.has_dollar_amount, # makes no difference at all!

    # word lengths, binned
    # fe.word_length_ngrams_factory(1,1,1)
]
