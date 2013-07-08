"""
Feature templates and feature template factories.

Every function in this file is a feature template, which takes a document and
outputs a sequence of tokens, or a feature template factory (iff it ends in
_factory), which returns a feature template.

Features templates expect to only be given the newsgroup message's payload, no
headers.  Features that depend on headers are not supported.
"""
from sklearn.feature_extraction.text import CountVectorizer
import nltk, re
from feature_support import *

"""Bag of n-grams."""
def ngrams_factory(n1, n2):
  return CountVectorizer(ngram_range=(n1,n2)).build_analyzer()



"""Bag of n-grams after Porter-stemming."""
def stem_ngrams_factory(n1, n2):
  cv = CountVectorizer(ngram_range=(n1,n2), preprocessor=stem_text)
  return cv.build_analyzer()



"""Bag of part-of-speech n-grams."""
def pos_ngrams_factory(n1, n2):
  def pos_ngrams(doc):
    # doc = doc[0:200] # sloppy way to speed things up
    tokens = nltk.word_tokenize(doc)
    tagged = tag(tokens)
    tags_only = map(lambda (_,b): b,
                    tagged)
    safe_tags_only = map(str, tags_only)
    return list2ngrams(n1, n2, safe_tags_only)
  return pos_ngrams



"""
Bag of word length n-grams, binned.

binning_divisor controls how large each bin is.  The word lengths are divided
by binning_divisor then rounded down.  Smaller values of binning_divisor mean
sparser word length features.
"""
def word_length_ngrams_factory(n1, n2, binning_divisor):
  def word_length_ngrams(doc):
    tokens = nltk.word_tokenize(doc)
    lengths = map(lambda w: str(len(w)/binning_divisor),
                  tokens)
    length_ngrams = map(lambda x: 'LENGTH ' + x,
                        list2ngrams(n1, n2, lengths))
    return length_ngrams
  return word_length_ngrams



"""
Whether the doc has a dollar amount (US-centric).

Either $ with a number shortly afterwards, or "dollars" with a number shortly
before.  Meant to correlate with misc.forsale.
"""
def has_dollar_amount(doc):
  if re.match(r'.*[$]\s*\d+', doc) or re.match(r'.*\d+\s*dollars', doc):
    return ['HAS_DOLLAR_AMOUNT']
  else:
    return []
