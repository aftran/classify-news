"""
Support objects for feature templates to use.
"""
from nltk.stem.porter import PorterStemmer

stem = PorterStemmer().stem

def stem_text(text):
  """The text with all words (found by str.split) stemmed."""
  return ' '.join(map(stem, text.split()))
