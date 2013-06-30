"""
Feature templates and feature template factories.

Every function in this file is a feature template, which takes a document and
outputs a sequence of tokens, or a feature template factory (iff it ends in
_factory), which returns a feature template.

Features templates expect to only be given the newsgroup message's payload, no
headers.  Features that depend on headers are not supported.
"""
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from feature_support import *

"""Bag of n-grams."""
def ngrams_factory(n):
  return CountVectorizer(ngram_range=(n,n)).build_analyzer()

"""Bag of n-grams after Porter-stemming."""
def stem_ngrams_factory(n):
  cv = CountVectorizer(ngram_range=(n,n), preprocessor=stem_text)
  return cv.build_analyzer()
