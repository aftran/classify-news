#!/usr/bin/env python
"""A naive Bayes classifier with a hand-crafted set of features."""
from sklearn.naive_bayes import MultinomialNB
import feature as fe

estimator = MultinomialNB(fit_prior=False)
# fit_prior is false because the relative sizes of each newsgroup in the
# training corpus are coincidental, not indicative of which newsgroup is most
# likely a priori.

feature_templates = [
    fe.stem_ngrams_factory(1),
    # fe.stem_ngrams_factory(2)
]
