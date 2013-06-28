from os import path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from collections import deque
import codecs, sys


def train_estimator(estimator, feature_templates, corpus_dir, train_paths):
  """
  Trains estimator on the files in train_paths after using feature_templates to
  project them into feature space.  The class of each document is considered to
  be the name of the subfolder it is in.

  Returns the vectorizer used to convert documents into feature
  representations.  This is needed for predicting the class of new documents.
  
  The estimator MUST be multiclass: naive Bayes or a binary classifier wrapped
  in OneVsRestClassifier is acceptable.

  estimator: an sklearn multiclass estimator
  feature_templates: a collection of functions, each taking a document string
  and returning a collection of features
  corpusdir: the root of the corpus dir
  train_paths: files relative to corpus_dir to train the model based on
  """
  # For now, we'll store the whole corpus in RAM.  If some future corpus is too
  # big to fit in RAM, replace docs/read_corpus with an iterable that reads one
  # file at a time.
  docs = read_corpus(corpus_dir, train_paths)
  class_labels = paths2class_labels(train_paths)
  analyzer = make_analyzer(feature_templates)
  vectorizer = CountVectorizer(analyzer=analyzer, dtype=float)

  print 'Vectorizing the corpus...'
  vectors = vectorizer.fit_transform(docs)

  # Standardization, which is recommended if estimator is as SVM.
  # vectors = preprocessing.scale(vectors, with_mean=False)
  # I'm omitting standardization for now since SVMs might be infeasible anyway.
  # If we standardize later, be sure to output a function that lets us perform
  # the same transformation to each test vector.
  # Ideally we'd like to use with_mean=True, but I can't get that to work with
  # a sparse matrix.

  print 'Fitting estimator...'
  estimator.fit(vectors, class_labels)
  return vectorizer



def standardize(vectors):
  """Do standardization, which is recomended if estimator is an SVM."""
  return preprocessing.scale(vectors, with_mean=False)



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
  return map(lambda x: path.split(x)[0],
             train_paths)



def read_corpus(corpus_dir, train_paths):
  """
  An iterable whose elements are the raw text of each document pointed to by
  train_paths, in that order.  The train_paths are relative to the corpus_dir.
  """
  result = deque()
  for train_path in train_paths:
    doc_path = path.join(corpus_dir, train_path)
    with codecs.open(doc_path, 'r', 'cp850') as f: # cp850 is what worked
      result.append(f.read())
  return result



def doc2vector(docs, feature_templates):
  """
  A vector representation of a document given a list of feature templates.  The
  order of the vector components is consistent given a value of
  feature_templates.
  """
  # TODO: A bag of words feature template can't just 
  return reduce(
                map(lambda x: x(doc), feature_templates))
