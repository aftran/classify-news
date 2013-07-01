#!/usr/bin/env python
"""A logistic regression classifier with suggested features."""
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import feature as fe

estimator = OneVsRestClassifier(LogisticRegression())

feature_templates = [
    fe.stem_ngrams_factory(1,2),
]
