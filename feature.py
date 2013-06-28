"""
Every function in this file is a feature template, which takes a document and
outputs a sequence of tokens, or a feature template factory (iff it ends in
_factory), which returns a feature template.
"""
import feature_support
from sklearn.feature_extraction.text import CountVectorizer

"""Bag of n-grams."""
def ngrams_factory(n):
  return CountVectorizer(ngram_range=(n,n)).build_analyzer()
