#!/usr/bin/env python
"""An SVM (SVC) classifier with suggested features. Quite slow."""
from sklearn import svm
import feature as fe

estimator = svm.SVC(kernel='rbf', C=1.0)
# fit_prior is false because the relative sizes of each newsgroup in the
# training corpus are coincidental, not indicative of which newsgroup is most
# likely a priori.

feature_templates = [
    fe.stem_ngrams_factory(1,1),
    # fe.stem_ngrams_factory(2,2)
]
