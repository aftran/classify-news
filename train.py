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
  # vectorizer = CountVectorizer(analyzer=analyzer, dtype=float64) # TODO: This is the true line.
  vectorizer = CountVectorizer(dtype=float64) # TODO: This is the stub/test version of above.

  print 'Vectorizing the corpus...'
  vectors = vectorizer.fit_transform(docs)

  # Data standardization is highly recommended for SVMs, but the following line
  # changes the variances but does not recenter.  We would like to also
  # do mean removal (that is, use with_mean=True), but I'm not sure how to do
  # that with a sparse matrix.  The lack of recentering might be why SVMs take
  # a long time to train, but I'm not sure.
  vectors = preprocessing.scale(vectors, with_mean=False)

  print 'Fitting estimator...'
  estimator.fit(vectors, class_labels)
  return vectorizer



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
