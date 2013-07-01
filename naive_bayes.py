#!/usr/bin/env python
"""A naive Bayes classifier with suggested features."""
from sklearn.naive_bayes import MultinomialNB
import feature as fe

estimator = MultinomialNB(fit_prior=False, alpha=1)
# fit_prior is false because the relative sizes of each newsgroup in the
# training corpus are coincidental, not indicative of which newsgroup is most
# likely a priori.

feature_templates = [

    # stem 1- and 2-grams
    # fe.stem_ngrams_factory(1,2),

    # bag of stems: slow and somehow makes accuracy worse on dev set
    # fe.stem_ngrams_factory(1,1),

    # bag of words, no stemming
    fe.ngrams_factory(1,1),

    # bag of bigrams, no stemming
    # fe.ngrams_factory(2,2),

    # bag of bigrams with stemming
    # fe.stem_ngrams_factory(2,2),

    # part-of-speech 1,2,3-grams
    #fe.pos_ngrams_factory(1,3)

    # part-of-speech unigrams: probably too dense
    # fe.pos_ngrams_factory(1,1),

    # part-of-speech bigrams
    # fe.pos_ngrams_factory(2,2),

    # part-of-speech trigrams
    fe.pos_ngrams_factory(3,3),
]
