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

def dev_evaluate(estimator, feature_templates, corpus_dir, data_paths):
  """Evaluate the corpus with as 90% train, 10% test from end of corpus."""
  data_paths = list(data_paths) # can't slice a deque
  class_labels = paths2class_labels(data_paths)
  split_point = int(0.9 * len(data_paths))

  train_paths = data_paths[:split_point]
  test_paths  = data_paths[split_point:]
  test_labels = paths2class_labels(test_paths)

  train_labels = class_labels[:split_point]
  test_labels  = class_labels[split_point:]

  _, vectorizer = train_estimator(estimator, feature_templates,
                                  corpus_dir, train_paths)
  test_vectors = vectorize_corpus_with_vectorizer(vectorizer, corpus_dir,
                                                  test_paths)
  # return estimator.score(test_vectors, test_labels)
  predictions = estimator.predict(test_vectors)
  mistakes = deque()
  for path, gold, prediction in zip(test_paths, test_labels, predictions):
    if gold != prediction:
      mistakes.append((path, prediction))
  return 1-float(len(mistakes))/len(test_labels), list(mistakes)
