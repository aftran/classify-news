#!/usr/bin/env python
"""
Print mistakes made by classifier and its accuracy.

Usage: find_mistakes.py corpus_dir pickled_paths classifier

pickled_paths is a pickled collection of paths to email files relative to
corpus_dir.  The last 10% will be used for testing, so it is important that it
be scrambled to prevent aliasing.

classifier is the name of the module containing the classifier.  This module
must have feature_templates and estimator.
"""
import sys
sys.path.append('..')
import importlib, pickle
from evaluate import *


corpus_dir = sys.argv[1]
with open(sys.argv[2]) as f:
  data_paths = pickle.load(f)
mod = importlib.import_module(sys.argv[3])

score, mistakes = dev_evaluate(mod.estimator, mod.feature_templates,
                               corpus_dir, data_paths)

# for pretty-printing
column1_width = 1 + max(map(lambda pair: len(pair[0]),
                            mistakes))

for doc_path, wrong_class in mistakes:
  text = doc_path + ' ' * (column1_width - len(doc_path)) + wrong_class
  print >> sys.stderr, text

print score
