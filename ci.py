# Copyright 2019
"""
ROC AUC with 95% confidence intervals.

Here, bootraping confidence intervals are used to complement the ROC AUC score.
Partially based on: https://cs.stanford.edu/people/ihaque/posters/EC4-AUC_Confidence_Intervals.pdf

In python, run as:
>>> import ci
>>> ci.AUC(gold, predictions).evaluate()
"""
from sklearn.metrics import *

class AUC(object):

    def __init__(self,
                 gold_truth=np.array([], dtype=np.uint8),
                 predictions=np.array([], dtype=np.uint8),
                 method=roc_auc_score,
                 resamples=2000,
                 random_state=42):
        self.method = method
        self.gold = pd.Series(gold_truth)
        self.predictions = pd.Series(predictions)
        self.resamples = resamples
        self.sample_evaluations = []
        self.random_state = random_state
        self.sorted_sample_evaluations = pd.Series()
        assert self.gold.shape[0] == self.predictions.shape[0]
        self.score = method(gold_truth, predictions)

    def resample(self):
        for _ in range(self.resamples):
            gold = self.gold.sample(frac=.5, replace=True, random_state=self.random_state).to_list()
            pred = self.predictions.sample(frac=.5, replace=True, random_state=self.random_state).to_list()
            self.sample_evaluations.append(self.method(gold, pred))
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
