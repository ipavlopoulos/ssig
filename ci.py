# Copyright 2019
"""
ROC AUC with 95% confidence intervals.

Here, bootraping confidence intervals are used to complement the ROC AUC score.
Partially based on: https://cs.stanford.edu/people/ihaque/posters/EC4-AUC_Confidence_Intervals.pdf

Minimum requirements: pandas, sklearn, tqdm and numpy.

In python, run as:
>>>import ci, numpy as np
>>>y_true = np.array([0, 0, 1, 1])
>>>y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>>auc, (lower, upper) = ci.AUC(y_true, y_scores).evaluate()
"""

from sklearn.metrics import *
import numpy as np
import pandas as pd
from tqdm import tqdm

class AUC(object):

    def __init__(self,
                 gold_truth=np.array([], dtype=np.uint8),
                 predictions=np.array([], dtype=np.uint8),
                 method=roc_auc_score,
                 n_boot=1000):
        assert len(gold_truth) == len(predictions)
        self.method = method
        self.dataset = pd.DataFrame()
        self.dataset["GOLD"] = gold_truth
        self.dataset["PREDICTED"] = predictions
        self.n_boot = n_boot
        self.sample_evaluations = []
        self.sorted_sample_evaluations = pd.Series()
        self.score = method(gold_truth, predictions)

    def resample(self):
        for _ in tqdm(range(self.n_boot)):
            sample = self.dataset.sample(frac=.5, replace=True)
            self.sample_evaluations.append(self.method(sample.GOLD, sample.PREDICTED))
        self.sorted_sample_evaluations = pd.Series(sorted(self.sample_evaluations))

    def get_cis(self, quantiles=[0.025, 0.975]):
        assert self.sorted_sample_evaluations.shape[0]>0
        lower_ci, upper_ci = self.sorted_sample_evaluations.quantile(quantiles)
        return lower_ci, upper_ci

    def evaluate(self, conf=.95):
        self.resample()
        a = (1.-conf) / 2.
        q = [a, 1.-a]
        l, u = self.get_cis(q)
        return self.score, (l,u)

def example(ssize = 1000):
    """
    Assume a class-balanced binary classification problem.
    Let Uniform be the baseline and a Gaussian be your system.
    Then you would get something like the following.
    """
    system = np.clip(np.random.normal(0.5, 0.5, ssize), 0, 1) # normal close to .5
    baseline = np.random.uniform(size=ssize) # uniform
    gold = int(ssize*.5)*[1] + int(ssize*.5)*[0]
    system_score = AUC(gold, system).evaluate()
    baseline_score = AUC(gold, baseline).evaluate()
    return system_score, baseline_score
