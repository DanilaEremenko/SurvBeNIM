import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator, nelson_aalen_estimator


class CoxWrapperLifeLines:
    def __init__(self):
        self.cox = CoxPHFitter()

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            full_df = X.copy()
        elif isinstance(X, np.ndarray):
            full_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        full_df['event'] = [y_el[0] for y_el in y]
        full_df['event_time'] = [y_el[1] for y_el in y]
        self.event_times_ = full_df['event_time'].sort_values().to_numpy()
        self.cox.fit(full_df, duration_col='event_time', event_col='event')

    def predict_survival_function(self, X):
        y_pred = self.cox.predict_survival_function(X)
        return np.array([StepFunction(x=self.event_times_, y=y_pred[sample_i].values)
                         for sample_i in range(len(y_pred.columns))])

    def predict(self, X):
        return self.cox.predict_median(X).to_numpy()

    def score(self, X, y):
        full_df = X.copy()
        full_df['event'] = [y_el[0] for y_el in y]
        full_df['event_time'] = [y_el[1] for y_el in y]
        return self.cox.score(full_df, scoring_method='concordance_index')


class CoxFairBaseline:
    def __init__(self, training_events, training_times: np.ndarray, baseline_estimator_f):
        if baseline_estimator_f == kaplan_meier_estimator:
            baseline_estimator = baseline_estimator_f(training_events, training_times)
            self._H0 = -np.log(baseline_estimator[1])
        elif baseline_estimator_f == nelson_aalen_estimator:
            baseline_estimator = baseline_estimator_f(training_events, training_times)
            self._H0 = baseline_estimator[1]
        else:
            raise Exception(f"Undefined baseline estimator = {baseline_estimator_f}")

        self.unique_times_ = np.sort(np.unique(training_times))
        self.event_times_ = self.unique_times_

    @property
    def H0(self):
        return self._H0

    def predict_survival_np(self, X: np.ndarray, cox_coefs):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        linear_predictor = X @ cox_coefs
        risk_score = np.exp(linear_predictor)
        H0_proper = self.H0[:, np.newaxis] @ np.ones((len(X)))[np.newaxis]
        return np.exp(-(H0_proper * risk_score).T)

    def predict_survival_from_surv_np(self, X: np.ndarray, cox_coefs):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        linear_predictor = X @ cox_coefs
        risk_score = np.exp(linear_predictor)
        H0_proper = self.H0[:, np.newaxis] @ np.ones((len(X)))[np.newaxis]
        H0_surv = np.exp(-H0_proper)
        return np.power(H0_surv, risk_score).T

    def predict_cum_hazard_np(self, X: np.ndarray, cox_coefs):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        linear_predictor = X @ cox_coefs
        linear_predictor /= linear_predictor.max() * 5
        risk_score = np.exp(linear_predictor)
        H0_proper = self.H0[:, np.newaxis] @ np.ones((len(X)))[np.newaxis]
        return (H0_proper * risk_score).T

    def predict_cum_hazard_from_surv_np(self, X: np.ndarray, cox_coefs):
        return -np.log(self.predict_survival_np(X, cox_coefs))

    def predict_survival_function(self, X, cox_coefs):
        y_pred_surv_cox = self.predict_survival_np(X=X, cox_coefs=cox_coefs)
        return np.array([StepFunction(x=self.unique_times_, y=sample) for sample in y_pred_surv_cox])

    def predict_survival_function(self, X, cox_coefs):
        y_pred_surv_cox = self.predict_survival_np(X=X, cox_coefs=cox_coefs)
        return np.array([StepFunction(x=self.unique_times_, y=sample) for sample in y_pred_surv_cox])
