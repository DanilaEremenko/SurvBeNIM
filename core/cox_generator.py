import math
from math import log2, exp

import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from core.datasets_processing import get_flchain


class CoxGenerator:
    def __init__(self, coefs: np.ndarray):
        # x_train, y_train, _, _ = get_flchain()
        # x_train = x_train[:, :3]
        # x_train = x_train / x_train.max(axis=0)
        #
        # self.model_cox = CoxPHSurvivalAnalysis()
        # self.model_cox.fit(x_train, y_train)
        # self.model_cox.coef_ = coefs
        self.coefs = coefs

    def generate_data(self, size: int, censored_part: float, l=1e-5, v=2):
        assert 0 <= censored_part < 1.0
        x = np.random.random((size, len(self.coefs)))
        event_times = (-math.log(0.5) / (l * np.exp(np.sum(self.coefs.T * x, axis=1)))) ** (1 / v)
        events = np.ones(len(event_times))
        censored_ids = np.random.choice(list(range(size)), int(size * censored_part))
        events[censored_ids] = 0
        # cut_coefs = np.random.uniform(low=0.5, high=0.9, size=len(censored_ids))
        cut_coefs = np.random.uniform(low=0.1, high=0.9, size=len(censored_ids))
        event_times[censored_ids] *= cut_coefs

        return x, Surv.from_arrays(event=events, time=event_times)


if __name__ == '__main__':
    cox_generator = CoxGenerator(coefs=np.array([0.8, 0.19, 0.01]))

    x_cox_train, x_cox_test, y_cox_train, y_cox_test = train_test_split(*cox_generator.generate_data(1000),
                                                                        train_size=0.7)

    # model = CoxPHSurvivalAnalysis()
    # model = RandomSurvivalForest(max_samples=min(500, len(x_cox_train[0])), max_depth=8)
    # model.fit(x_cox_train, y_cox_train)
