"""
Feature templates and feature template factories.

Every function in this file is a feature template, which takes a document and
outputs a sequence of tokens, or a feature template factory (iff it ends in
_factory), which returns a feature template.

Features templates expect to only be given the newsgroup message's payload, no
headers.  Features that depend on headers are not supported.
"""
import feature_support
from sklearn.feature_extraction.text import CountVectorizer

"""Bag of n-grams."""
def ngrams_factory(n):
  return CountVectorizer(ngram_range=(n,n)).build_analyzer()
