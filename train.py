from os import path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from collections import deque
from numpy import array
import codecs, sys, email


def train_estimator(estimator, feature_templates, corpus_dir, train_paths):
  """
  Trains estimator on the files in train_paths after using feature_templates to
  project them into feature space.  The class of each document is considered to
  be the name of the subfolder it is in.

  Returns a pair: the vectorized corpus (a sparse matrix of row vectors) and
  the vectorizer used to convert documents into feature representations, which
  is needed for predicting the class of new documents.
  
  The estimator MUST be multiclass: naive Bayes or a binary classifier wrapped
  in OneVsRestClassifier is acceptable.

  estimator: an sklearn multiclass estimator
  feature_templates: a collection of functions, each taking a document string
  and returning a collection of features
  corpus_dir: the root of the corpus dir
  train_paths: files relative to corpus_dir to train the model based on
  """
  print 'Vectorizing the corpus...'
  vectors, vectorizer = vectorize_corpus(feature_templates,
                                         corpus_dir, train_paths)
  vectors = standardize(vectors)

  print 'Fitting the estimator...'
  class_labels = paths2class_labels(train_paths)
  estimator.fit(vectors, class_labels)
  return vectors, vectorizer



def vectorize_corpus(feature_templates, corpus_dir, train_paths):
  """Turn the corpus into a sparse matrix of row vectors and a vectorizer."""
  # For now, we'll store the whole corpus in RAM.  If some future corpus is too
  # big to fit in RAM, replace docs/read_corpus with an iterable that reads one
  # file at a time.
  docs = read_corpus(corpus_dir, train_paths)
  analyzer = make_analyzer(feature_templates)
  vectorizer = CountVectorizer(analyzer=analyzer, dtype=float)
  vectors = vectorizer.fit_transform(docs)
  return vectors, vectorizer



def vectorize_corpus_with_vectorizer(vectorizer, corpus_dir, data_paths):
  """Given an already-fit vectorizer, return sparse matrix of row vectors."""
  docs = read_corpus(corpus_dir, data_paths)
  vectors = vectorizer.transform(docs)
  return vectors



def standardize(vectors):
  """Do standardization, which is recomended if estimator is an SVM."""
  return preprocessing.scale(vectors, with_mean=False)
  # Ideally we'd like to use with_mean=True, but I can't get that to work with
  # a sparse matrix.



def make_analyzer(feature_templates):
  """
  Returns a closure that returns a sequence of features given a list of feature
  templates.  Each feature template takes a document and returns a list of
  features.
  """
  def analyzer(doc):
    features = deque()
    for feature_template in feature_templates:
      features.extend(feature_template(doc))
    return features
  return analyzer



def paths2class_labels(train_paths):
  """
  Given a list of relative paths within the training corpus, returns a list of
  the class labels of the documents at those paths, preserving order.
  Hard-coded to work with the 20_newsgroup corpus.
  """
  return array(map(lambda x: path.split(x)[0],
                   train_paths))



def read_corpus(corpus_dir, train_paths):
  """
  Return an iterable whose elements are the raw payload text (no headers) of
  each email pointed to by train_paths, in that order.  The train_paths are
  relative to the corpus_dir.
  """
  result = deque()
  for train_path in train_paths:
    doc_path = path.join(corpus_dir, train_path)
    with codecs.open(doc_path, 'r', 'cp850') as f: # cp850 is what worked
      result.append(email.message_from_file(f).get_payload())
  return result
