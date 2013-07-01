#!/usr/bin/env python
"""
Save the vectorizer and estimator associated with the given model.

Usage: save_model.py model_name corpus_dir train_paths output_dir

Outputs the model to output_dir/model_name.{vectorizer,estimator}, appropriate
for cnn.py.
"""
import sys
sys.path.append('..')
import importlib, pickle, cloud
from os import path
from sklearn.externals import joblib
from train import *

def main():
  estimator_name = sys.argv[1]
  mod = importlib.import_module(sys.argv[1])
  corpus_dir = sys.argv[2]
  with open(sys.argv[3]) as f:
    train_paths = pickle.load(f)
  out_dir = sys.argv[4]

  vectors, vectorizer = train_estimator(mod.estimator, mod.feature_templates,
                                        corpus_dir, train_paths)
  joblib.dump(mod.estimator, path.join(out_dir, estimator_name + '.estimator'), compress=9)
  with open(path.join(out_dir, estimator_name + '.vectorizer'), 'w') as f:
    cloud.serialization.cloudpickle.dump(vectorizer, f)



if __name__ == "__main__":
  main()
