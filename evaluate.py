from train import *
from sklearn.cross_validation import cross_val_score

def kfold(estimator, feature_templates, corpus_dir, data_paths):
  """
  Do k-fold cross validation.

  estimator: an sklearn multiclass estimator
  feature_templates: a collection of functions, each taking a document string
  and returning a collection of features
  corpus_dir: the root of the corpus dir
  data_paths: files to use for training and evaluating
  """
  class_labels = paths2class_labels(data_paths)
  print 'Vectorizing the corpus...'
  vectors, vectorizer = vectorize_corpus(feature_templates,
                                         corpus_dir, data_paths)
  print 'Doing cross-validation...'
  return cross_val_score(estimator, vectors, class_labels, cv=5)
