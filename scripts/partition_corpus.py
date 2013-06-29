"""
For splitting the 20_newsgroup corpus into a test and non-test set randomly.
Outputs pickled lists.

Usage: data_subsets.py corpusdir outputdir
"""

#!/usr/bin/env python
import sys, random, pickle
from collections import deque
from os import listdir, path

def partition_corpus(corpusdir):
  """
  Randomly splits the 20_newsgroup corpus into approximately 10% test items and
  90% the rest, with an equal number of documents from each class appearing in
  the test set.

  Output: (test, rest), a pair of deques of paths to files.  Paths are relative
  to corpusdir.
  """
  test = deque()
  rest = deque()
  for newsgroup in listdir(corpusdir):
    filenames = listdir(path.join(corpusdir, newsgroup))
    relative_paths = map(lambda filename: path.join(newsgroup, filename),
                         filenames)
    random.shuffle(relative_paths)
    test.extend(relative_paths[0:10])
    rest.extend(relative_paths[10:])
  return test, rest


if __name__ == "__main__":
  test, rest = partition_corpus(sys.argv[1])
  output_dir = sys.argv[2]
  with open(path.join(output_dir, 'test'), 'w') as f:
    pickle.dump(test, f)
  with open(path.join(output_dir, 'rest'), 'w') as f:
    pickle.dump(rest, f)
