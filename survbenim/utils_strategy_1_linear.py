import json
import time
from pathlib import Path
from typing import List

import sksurv
from sklearn.cluster import KMeans
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import nelson_aalen_estimator, kaplan_meier_estimator
from sksurv.util import Surv
from core.cox_wrapper import CoxFairBaseline
from survbex.explainers import SurvBexExplainer
import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from survbex.estimators import BeranModel
from survshap import SurvivalModelExplainer
from survshap import ModelSurvSHAP


def explain_exp_point(model, exp_point: np.ndarray, exp_event: bool, exp_time: float, pred_surv_fn, cox_clusters,
                      train_features: pd.DataFrame, train_events: np.ndarray, train_times: np.ndarray,
                      solver_name: str,
                      radius: float,
                      res_dir: Path):
    res_dir.mkdir(exist_ok=True, parents=True)
    explainer = SurvBexExplainer(
        training_features=train_features,
        training_events=list(train_events),
        training_times=list(train_times),
        model_output_times=np.argsort(np.unique(train_times)),
        kernel_width=radius,
        random_state=42
    )

    start_time = time.time()

    if solver_name == 'survlime':
        f_imps = explainer.explain_instance(
            data_row=exp_point,
            predict_fn=pred_surv_fn,
            num_samples=100,
            num_val_samples=100,
            type_fn='survival',
            optimizer='convex'
        )
        cox_model = CoxFairBaseline(training_events=train_events, training_times=train_times,
                                    baseline_estimator_f=nelson_aalen_estimator)
        explainer_neigh_s = cox_model.predict_survival_function(X=explainer.opt_funcion_maker.neighbours,
                                                                cox_coefs=f_imps)
        explainer_neigh_s = np.array([s.y for s in explainer_neigh_s])
        explainer_dp_s = cox_model.predict_survival_function(X=exp_point, cox_coefs=f_imps)
        explainer_dp_s = np.array([s.y for s in explainer_dp_s])
        bbox_neigh_s = explainer.opt_funcion_maker.bbox_neigh_s

    elif solver_name == 'survshap':
        surv_shap = SurvivalModelExplainer(
            model=model,  # bbox,
            data=train_features,  # x
            y=Surv.from_arrays(event=train_events, time=train_times),  # y
            # data=pd.DataFrame(exp_point, columns=train_features.keys()),  # x
            # y=Surv.from_arrays(event=[exp_event], time=[exp_time]),
            predict_survival_function=lambda model, X: pred_surv_fn(X)
        )

        exp_survshap = ModelSurvSHAP(random_state=42, max_shap_value_inputs=int(1e3))
        exp_survshap.fit(
            surv_shap,
            new_observations=pd.DataFrame(exp_point, columns=train_features.keys()),
            timestamps=np.array([exp_time])
        )
        f_imps = np.array(
            [
                imp[1]
                for pt_exp in exp_survshap.individual_explanations
                for imp in pt_exp.simplified_result.values
            ]
        )
        explainer_neigh_s = None
        explainer_dp_s = None
        bbox_neigh_s = None

    elif solver_name == 'survshap_km':
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(train_features.to_numpy())
        train_cl_ids = kmeans.predict(train_features.to_numpy())
        exp_cl_id = kmeans.predict(exp_point)

        surv_shap = SurvivalModelExplainer(
            model=model,  # bbox,
            data=train_features.iloc[train_cl_ids == exp_cl_id],  # x
            y=Surv.from_arrays(event=train_events, time=train_times),  # y
            # data=pd.DataFrame(exp_point, columns=train_features.keys()),  # x
            # y=Surv.from_arrays(event=[exp_event], time=[exp_time]),
            predict_survival_function=lambda model, X: pred_surv_fn(X)
        )

        exp_survshap = ModelSurvSHAP(random_state=42, max_shap_value_inputs=int(1e3))
        exp_survshap.fit(
            surv_shap,
            new_observations=pd.DataFrame(exp_point, columns=train_features.keys()),
            timestamps=np.array([exp_time])
        )
        f_imps = np.array(
            [
                imp[1]
                for pt_exp in exp_survshap.individual_explanations
                for imp in pt_exp.simplified_result.values
            ]
        )
        explainer_neigh_s = None
        explainer_dp_s = None
        bbox_neigh_s = None

    elif solver_name == 'survbex':
        f_imps = explainer.explain_instance(
            data_row=exp_point,
            predict_fn=pred_surv_fn,
            num_samples=100,
            num_val_samples=100,
            type_fn='survival',
            optimizer='gradient',
            grid_info_file=f"{res_dir}/optimization.csv",
            max_iter=200
        )

        beran_model = BeranModel(
            kernel_name='triangle',
            kernel_width=0.1 * explainer.opt_funcion_maker.bbox_neigh_s.std(axis=0).std()
        )
        beran_model.fit(X=train_features.to_numpy(), y_events=train_events, y_event_times=train_times, b=f_imps)
        explainer_neigh_s = beran_model.predict_survival_torch_optimized(explainer.opt_funcion_maker.neighbours)
        explainer_dp_s = beran_model.predict_survival_torch_optimized(exp_point)
        bbox_neigh_s = explainer.opt_funcion_maker.bbox_neigh_s

    else:
        raise Exception(f'Undefined solver name = {solver_name}')

    bbox_dp_s = pred_surv_fn(exp_point)
    bbox_dp_s = np.array([s.y for s in bbox_dp_s])

    full_time = time.time() - start_time

    with open(f'{res_dir}/res.json', 'w+') as fp:
        json.dump(
            obj=dict(
                full_time=full_time,
                importances=f_imps.tolist(),
                bbox_neigh_s=bbox_neigh_s.tolist() if bbox_neigh_s is not None else None,
                bbox_dp_s=bbox_dp_s.tolist() if bbox_dp_s is not None else None,
                explainer_neigh_s=explainer_neigh_s.tolist() if explainer_neigh_s is not None else None,
                explainer_dp_s=explainer_dp_s.tolist() if explainer_dp_s is not None else None
            ),
            fp=fp
        )


def run_main_st_1_linears(
        exp_dir: Path,
        bbox_name: str,
        solver_name: str,
        test_ids: List[int],
        radius: float
):
    res_dir = exp_dir.joinpath(f"bbox={bbox_name},radius={radius}").joinpath(f'{solver_name}_st_1')

    with open(f"{exp_dir}/config.json") as fp:
        config = json.load(fp)

    with open(f"{exp_dir}/dataset.json") as fp:
        ds_json = json.load(fp)

        train_features = pd.DataFrame(ds_json['train_features'])
        train_events = np.array(ds_json['train_events'])
        train_times = np.array(ds_json['train_times'])
        train_importances = np.array(ds_json['train_importances'])
        train_pairs = Surv.from_arrays(event=train_events, time=train_times)
        all_train = [train_features, train_pairs]

        test_features = pd.DataFrame(ds_json['test_features'])
        test_events = np.array(ds_json['test_events'])
        test_times = np.array(ds_json['test_times'])
        test_importances = np.array(ds_json['test_importances'])
        test_pairs = Surv.from_arrays(event=test_events, time=test_times)
        all_test = [test_features, test_pairs]

        train_cl_size = len(train_features) // len(config['cox_coefs_all'])
        test_cl_size = len(test_features) // len(config['cox_coefs_all'])

        cox_clusters = [
            [
                [
                    train_features[cl_i * train_cl_size:(cl_i + 1) * train_cl_size],
                    train_pairs[cl_i * train_cl_size:(cl_i + 1) * train_cl_size]
                ],
                [
                    train_features[cl_i * test_cl_size:(cl_i + 1) * test_cl_size],
                    test_pairs[cl_i * test_cl_size:(cl_i + 1) * test_cl_size]
                ]
            ]
            for cl_i, _ in enumerate(config['cox_coefs_all'])
        ]

    if bbox_name == 'rf':
        model = RandomSurvivalForest(n_estimators=100, max_samples=min(500, len(all_train[0])), max_depth=8,
                                     random_state=42)
        model.fit(all_train[0], all_train[1])
        pred_surv_fn = model.predict_survival_function
        pred_hazard_fn = model.predict_cumulative_hazard_function
        pred_risk_fn = model.predict
    elif bbox_name == 'beran':
        assert len(config['cox_coefs_all']) == 1
        model = BeranModel(kernel_width=250, kernel_name='gaussian')
        model.fit(X=all_train[0].to_numpy(), b=config['cox_coefs_all'][0],
                  y_events=train_events, y_event_times=train_times)
        pred_surv_fn = lambda X: model.predict_survival_torch_optimized(X)
        pred_hazard_fn = lambda X: -np.log(model.predict_survival_torch_optimized(X))
        pred_risk_fn = lambda X: np.sum(pred_hazard_fn(X), axis=1)

    elif 'cox' in bbox_name:
        model = CoxPHSurvivalAnalysis(alpha=1)

        model.fit(all_train[0], all_train[1])
        pred_surv_fn = model.predict_survival_function
        pred_hazard_fn = model.predict_cumulative_hazard_function
        pred_risk_fn = model.predict

        if bbox_name in ['cox_na', 'cox_km']:
            if bbox_name == 'cox_na':
                cox_fair_baseline = CoxFairBaseline(
                    training_events=train_events,
                    training_times=train_times,
                    baseline_estimator_f=nelson_aalen_estimator
                )
            elif bbox_name == 'cox_km':
                cox_fair_baseline = CoxFairBaseline(
                    training_events=train_events,
                    training_times=train_times,
                    baseline_estimator_f=kaplan_meier_estimator
                )
            else:
                raise Exception(f'Undefined cox model = {bbox_name}')

            model.coef_ /= np.abs(model.coef_).sum()
            pred_surv_fn = lambda X: cox_fair_baseline.predict_survival_function(X, cox_coefs=model.coef_)
            pred_hazard_fn = lambda X: cox_fair_baseline.predict_cum_hazard_from_surv_np(X, cox_coefs=model.coef_)
            pred_risk_fn = lambda X: np.dot(X, model.coef_)
        elif bbox_name != 'cox':
            raise Exception(f'Undefined cox model = {bbox_name}')
    else:
        raise Exception(f"Undefined bbox = {bbox_name}")

    cindex_train = concordance_index_censored(
        event_indicator=train_events, event_time=train_times, estimate=pred_risk_fn(train_features))[0]
    print(f'cindex train = {cindex_train}')
    cindex_test = concordance_index_censored(
        event_indicator=test_events, event_time=test_times, estimate=pred_risk_fn(test_features))[0]
    print(f'cindex test = {cindex_test}')

    for pt_i, exp_point, exp_event, exp_time in zip(
            test_ids,
            test_features.to_numpy()[test_ids],
            test_events[test_ids],
            test_times[test_ids]
    ):
        explain_exp_point(
            model=model,
            cox_clusters=cox_clusters, pred_surv_fn=pred_surv_fn,
            exp_point=exp_point[np.newaxis], exp_event=exp_event, exp_time=exp_time,
            train_features=train_features, train_times=train_times, train_events=train_events,
            radius=radius, solver_name=solver_name,
            res_dir=res_dir.joinpath(f"pt={pt_i}")
        )
