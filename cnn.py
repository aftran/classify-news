#!/usr/bin/env python
"""
Given URLs from CNN to stdin, output the most likely 20_newsgroup for each page.

Usage: cnn.py feature_templates estimator

vectorizer and estimator are an sklearn vectorizer and trained estimator.

vectorizer must have been saved with cloud.serialization.cloudpickle.dump, and
estimator must have been saved with joblib.dump.
"""
from bs4 import BeautifulSoup
from collections import deque
from sklearn.externals import joblib
import requests, sys, pickle

def classify(vectorizer, estimator, html):
  """Clean up html and classify resulting text."""
  soup = BeautifulSoup(html)
  html_paragraphs = soup.find_all('p', 'cnn_storypgraphtxt')
  parags = deque()
  for html_paragraph in html_paragraphs:
    parags.append(html_paragraph.text)
  text = ' '.join(parags)
  return estimator.predict(vectorizer.transform([text]))[0]



def main():
  try:
    with open(sys.argv[1]) as f:
      vectorizer = pickle.load(f)
    estimator  = joblib.load(sys.argv[2])
  except Exception as ex:
    print >> sys.stderr, 'Error loading model:'
    print ex
  else:
    while True:
      user_input = sys.stdin.readline()
      user_input = user_input.strip()
      if len(user_input) == 0:
        break
      try:
        request = requests.get(user_input)
      except Exception as ex:
        print >> sys.stderr, ex
      else:
        print classify(vectorizer, estimator, request.text)

if __name__ == '__main__': main()
