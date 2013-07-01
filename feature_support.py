"""
Support objects for feature templates to use.
"""
from nltk.stem.porter import PorterStemmer
import nltk.data, nltk.tag

stem = PorterStemmer().stem

"""POS tagger for a collection of tokens; tag will might be None."""
tag = nltk.tag.UnigramTagger(nltk.corpus.brown.tagged_sents()).tag
# tag  = nltk.data.load(nltk.tag._POS_TAGGER).tag   # too slow



def stem_text(text):
  """The text with all words (found by str.split) stemmed."""
  return ' '.join(map(stem, text.split()))



def list2ngrams(n1, n2, lst):
  """All n-grams in list."""
  for nn in range(n1, n2+1):
    for ix in range(0, len(lst) - nn):
      yield ' '.join(lst[ix:ix+nn])
