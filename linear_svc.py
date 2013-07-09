#!/usr/bin/env python
"""A linear SVC (SVM) classifier with suggested features."""
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
import feature as fe

"""
Feature selection: only use features with a top-PERCENTILE chi2 score.

Set to 100 to disable feature selection.
"""
PERCENTILE = 30

svc = OneVsRestClassifier(svm.LinearSVC())
selector = SelectPercentile(score_func=chi2, percentile=PERCENTILE)
steps = [('choose top features', selector),
         ('linear SVC',          svc)]
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
