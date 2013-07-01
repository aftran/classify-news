#!/usr/bin/env python
"""
A linear regression classifier with a hand-crafted set of features.

This is too slow to be of any use on the 20_newsgroup corpus.
"""
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
import feature as fe

estimator = OneVsRestClassifier(LinearRegression())
# fit_prior is false because the relative sizes of each newsgroup in the
# training corpus are coincidental, not indicative of which newsgroup is most
# likely a priori.

feature_templates = [
    fe.stem_ngrams_factory(1),
    fe.stem_ngrams_factory(2)
]
