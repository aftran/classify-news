#!/usr/bin/env python
"""A K-nearest-neighbours classifier with suggested features."""
from sklearn import neighbors
from sklearn.multiclass import OneVsRestClassifier
import feature as fe

# TODO: We're not giving KNN a fair go until we properly
# divide the mean out of each dimension.

estimator = OneVsRestClassifier(neighbors.KNeighborsClassifier())

feature_templates = [
    # All other features I've added have hurt.

    # stem 1- and 2-grams
    # fe.stem_ngrams_factory(1,2),

    # bag of stems
    fe.stem_ngrams_factory(1,1),

    # bag of 1,2-grams, no stemming
    # fe.ngrams_factory(1,2),

    # bag of words, no stemming
    # fe.ngrams_factory(1,1),

    # bag of bigrams, no stemming
    # fe.ngrams_factory(2,2),

    # bag of bigrams with stemming
    # fe.stem_ngrams_factory(2,2),

    # part-of-speech 1,2,3-grams
    # fe.pos_ngrams_factory(1,3)

    # part-of-speech unigrams: probably too dense
    # fe.pos_ngrams_factory(1,1),

    # part-of-speech bigrams
    # fe.pos_ngrams_factory(2,2),

    # part-of-speech trigrams
    # fe.pos_ngrams_factory(3,3), # slightly harms accuracy
]
