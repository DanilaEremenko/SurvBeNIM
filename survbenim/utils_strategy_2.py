import json
import logging
from pathlib import Path
from typing import Dict
import torch
from sklearn.metrics import pairwise_distances
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import nelson_aalen_estimator, kaplan_meier_estimator
from sksurv.util import Surv
from core.cox_wrapper import CoxFairBaseline
from core.drawing import draw_points_tsne, draw_surv_yo, draw_surv_yo_w_ax
from metrics import mse_torch
from nam_esimators import BNAMImp1, BNAMImp2, BaselineNAM, BaselineImportancesMLP
from survlimepy.utils.neighbours_generator import NeighboursGenerator
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sksurv.ensemble import RandomSurvivalForest
from matplotlib import pyplot as plt
from survbex.estimators import BeranModel
from utils_drawing import draw_shape_functions


class SamplesWeighter:
    def __init__(self, training_features, kernel_width):
        self.sd_features = np.std(
            training_features, axis=0, dtype=training_features.dtype
        )
        self.sd_sq = np.square(self.sd_features)
        self.kernel_width = kernel_width

    def weighted_euclidean_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff = x - y
        diff_sq = np.square(diff)
        div = np.divide(diff_sq, self.sd_sq)
        distance = np.sqrt(np.sum(div))
        return distance

    def kernel_fn(self, d):
        num = -(d ** 2)
        den = 2 * (self.kernel_width ** 2)
        weight = np.exp(num / den)
        return weight

    def get_weights(self, neighbours: np.ndarray, data_point: np.ndarray):
        distances = pairwise_distances(
            neighbours.detach().cpu().numpy(), data_point, metric=self.weighted_euclidean_distance
        ).ravel()
        weights = self.kernel_fn(distances)
        w = np.reshape(weights, newshape=(len(neighbours), 1))
        return w


def fit_grid_node(
        nam_claz, nam_args: dict,
        train_features: pd.DataFrame, train_events: np.ndarray, train_times: np.ndarray, bbox_train_s: torch.Tensor,
        val_features: np.ndarray, val_events: torch.Tensor, val_times: torch.Tensor, bbox_val_s: torch.Tensor,
        target_features: np.ndarray, target_points: torch.Tensor, bbox_target_s: torch.Tensor,
        criterion, optimizer: str, optimizer_args: dict, mode: str,
        v_mode: str, last_layer: str, max_epoch: int, save_dir: Path, fit=True
):
    if criterion in ['mse', 'mse_torch']:
        criterion = mse_torch
    else:
        raise Exception("Undefined criterion")

    logger = logging.getLogger('grad torch bnam optimizer')
    logger.setLevel(level=logging.DEBUG)

    train_features = torch.tensor(train_features.to_numpy(), dtype=torch.float32)
    train_y_events = torch.tensor(train_events)
    train_y_ets = torch.tensor(train_times, dtype=torch.float32)

    target_features = torch.tensor(target_features, dtype=torch.float32)

    val_features = torch.tensor(val_features, dtype=torch.float32)
    val_events = torch.tensor(val_events)
    val_times = torch.tensor(val_times, dtype=torch.float32)

    bnam_model = nam_claz(**nam_args, last_layer=last_layer)
    bnam_model.fit(
        X=train_features,
        y_event_times=train_y_ets.detach().cpu().numpy(),
        y_events=train_y_events.detach().cpu().numpy()
    )

    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=bnam_model.nam.parameters(), **optimizer_args)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(params=bnam_model.nam.parameters(), **optimizer_args)
    else:
        raise Exception(f"Unexpected optimizer = {optimizer}")

    # neighbours normalization
    # for fi in range(neighbours.shape[1]):
    #     neighbours[:, fi] -= neighbours[:, fi].min()
    #     neighbours[:, fi] /= neighbours[:, fi].max()

    t_deltas = torch.tensor(bnam_model.event_times_[1:] - bnam_model.event_times_[:-1],
                            dtype=torch.float32)
    t_deltas_target = torch.ones((len(target_features), 1)) @ t_deltas[np.newaxis]
    t_deltas_train = torch.ones((len(train_features), 1)) @ t_deltas[np.newaxis]
    t_deltas_val = torch.ones((len(val_features), 1)) @ t_deltas[np.newaxis]

    samples_weighter = SamplesWeighter(training_features=train_features.detach().cpu().numpy(), kernel_width=0.4)

    neigh_per_pt = len(bbox_target_s) // len(target_points)

    w = torch.tensor(
        np.vstack(
            [
                samples_weighter.get_weights(
                    neighbours=target_features[(pt_i) * neigh_per_pt:(pt_i + 1) * neigh_per_pt],
                    data_point=point[np.newaxis]
                )
                for pt_i, point in enumerate(target_points)
            ],
        ),
        dtype=torch.float32
    )

    history = []

    if mode in ['surv']:
        bbox_target_f = bbox_target_s
        bbox_train_f = bbox_train_s
        bbox_val_f = bbox_val_s
        pred_fn = lambda xps: torch.nan_to_num(
            bnam_model.predict_survival(xps),
            nan=1e-5
        )
        v_target = torch.ones((len(bbox_target_f), len(bnam_model.event_times_)))
        v_train = torch.ones((len(bbox_train_f), len(bnam_model.event_times_)))
        v_val = torch.ones((len(bbox_val_f), len(bnam_model.event_times_)))
    else:
        raise Exception(f"Undefined mode = {mode}")

    batch_size = 4
    best_cindex = 0
    best_file = str(save_dir.joinpath('best_model.pt'))
    for i in range(max_epoch):
        if not fit:
            history = []
            break
        for batch_start in range(0, len(target_features), batch_size):
            if i == 0:
                break
            optimizer.zero_grad()
            loss_target = criterion(
                y_true=bbox_target_f[batch_start:batch_start + batch_size],
                y_pred=pred_fn(xps=target_features[batch_start:batch_start + batch_size]),
                t_deltas=t_deltas_target[batch_start:batch_start + batch_size],
                v=v_target[batch_start:batch_start + batch_size],
                sample_weight=w[batch_start:batch_start + batch_size],
                b=torch.ones(train_features.shape[1])
            )
            loss_target.backward()
            optimizer.step()
        bnam_train_f = pred_fn(xps=train_features)
        bnam_val_f = pred_fn(xps=val_features)
        bnam_target_f = pred_fn(xps=target_features)

        cindex_background = concordance_index_censored(
            event_indicator=train_y_events.detach().cpu().numpy(), event_time=train_y_ets.detach().cpu().numpy(),
            estimate=1 / np.clip(bnam_train_f.mean(axis=-1).detach().cpu().numpy(), a_min=1e-5, a_max=1.)
        )[0]
        cindex_test = concordance_index_censored(
            event_indicator=val_events.detach().cpu().numpy(), event_time=val_times.detach().cpu().numpy(),
            estimate=1 / np.clip(bnam_val_f.mean(axis=-1).detach().cpu().numpy(), a_min=1e-5, a_max=1.)
        )[0]
        cindex_target = concordance_index_censored(
            event_indicator=np.ones(len(bbox_target_f), dtype=np.bool_),
            event_time=bbox_target_f.mean(axis=-1).detach().cpu().numpy(),
            estimate=1 / np.clip(bnam_target_f.mean(axis=-1).detach().cpu().numpy(), a_min=1e-5, a_max=1.)
        )[0]
        loss_target = criterion(y_true=bbox_target_f, y_pred=bnam_target_f, t_deltas=t_deltas_target,
                                v=v_target, sample_weight=torch.ones(len(bbox_target_f)),
                                b=torch.ones(train_features.shape[1]))
        loss_train = criterion(y_true=bbox_train_s, y_pred=bnam_train_f, t_deltas=t_deltas_train,
                               v=v_train, sample_weight=torch.ones(len(bbox_train_f)),
                               b=torch.ones(train_features.shape[1]))
        loss_val = criterion(y_true=bbox_val_s, y_pred=bnam_val_f, t_deltas=t_deltas_val,
                             v=v_val, sample_weight=torch.ones(len(bbox_val_f)),
                             b=torch.ones(val_features.shape[1]))

        curr_cindex = (cindex_background + cindex_test) / 2

        if curr_cindex > best_cindex:
            best_cindex = curr_cindex
            print('saving new best state dict..')
            torch.save(obj=bnam_model.nam.state_dict(), f=best_file)

        history.append(
            dict(
                target_loss=float(loss_target),
                train_loss=float(loss_train),
                val_loss=float(loss_val),
                target_cindex=cindex_target,
                val_cindex=cindex_test,
                train_cindex=cindex_background
            )
        )
        logger.debug(f'{len(history)}:target loss          = {loss_target:.4f}')
        logger.debug(f'{len(history)}:train loss           = {loss_train:.4f}')
        logger.debug(f'{len(history)}:target cindex        = {cindex_target:.4f}')
        logger.debug(f'{len(history)}:val cindex           = {cindex_test:.4f}')
        logger.debug(f'{len(history)}:train cindex         = {cindex_background:.4f}')

    bnam_model.nam.load_state_dict(state_dict=torch.load(best_file))
    bnam_model.nam.eval()

    return bnam_model, history


def get_neighbours_around_point(train_features: np.ndarray, exp_point: np.ndarray):
    n_generator = NeighboursGenerator(training_features=train_features, data_row=exp_point,
                                      sigma=0.4, random_state=np.random.RandomState(seed=42))
    return n_generator.generate_neighbours(num_samples=100)


def explain_points_set(
        exp_points: np.ndarray, exp_events: bool, exp_times: float, pred_surv_fn,
        train_features: pd.DataFrame, train_events: np.ndarray, train_times: np.ndarray,
        torch_bnam_grid: ParameterGrid, nam_claz,
        categories_dict: Dict[str, Dict[float, str]],
        res_dir: Path
):
    res_dir.mkdir(exist_ok=True, parents=True)
    all_neighbours = np.vstack(
        [
            get_neighbours_around_point(train_features=train_features.to_numpy(), exp_point=exp_point)
            for exp_point in exp_points
        ]
    )
    device = 'cpu'
    torch.set_default_device(device=device)
    draw_points_tsne(
        pt_groups=[
            train_features,
            all_neighbours,
            exp_points
        ],
        names=[
            'train background',
            'neighbours',
            'exp points'
        ],
        colors=[None] * 3,
        path=f'{res_dir}/tsne_exp_point.png'
    )

    bnam_grid_results = []

    bbox_neigh_s = pred_surv_fn(all_neighbours)
    bbox_neigh_s = torch.tensor([step.y for step in bbox_neigh_s], dtype=torch.float32)

    bbox_train_s = pred_surv_fn(train_features)
    bbox_train_s = torch.tensor([step.y for step in bbox_train_s], dtype=torch.float32)

    bbox_val_s = pred_surv_fn(exp_points)
    bbox_val_s = torch.tensor([step.y for step in bbox_val_s], dtype=torch.float32)

    bbox_dp_s = pred_surv_fn(exp_points)
    bbox_dp_s = torch.tensor([step.y for step in bbox_dp_s], dtype=torch.float32)

    def path_from_grid_args(grid_args: dict):
        optimizer_args = ','.join([f"{key}={val}" for key, val in grid_args['optimizer_args'].items()])
        nam_args = ','.join([f"{key}={val}" for key, val in grid_args['nam_args'].items()])
        grid_args = ','.join([f"{key}={val}" for key, val in grid_args.items()
                              if key not in ['optimizer_args', 'nam_args']])
        return ','.join([grid_args, nam_args, optimizer_args])

    for grid_i, grid_args in enumerate(torch_bnam_grid):
        grid_dir = res_dir.joinpath(path_from_grid_args(grid_args))
        grid_dir.mkdir(exist_ok=True, parents=True)
        with open(grid_dir.joinpath('grid_args.json'), 'w+') as fp:
            json.dump(fp=fp, obj=grid_args)

        print(f"torch optimization {grid_i + 1}/{len(torch_bnam_grid)}")
        print(f'grid_args = {grid_args}')
        bnam_model, history = fit_grid_node(
            nam_claz=nam_claz,
            train_features=train_features, train_events=train_events, train_times=train_times,
            val_features=exp_points, val_events=exp_events, val_times=exp_times,
            target_features=all_neighbours, target_points=exp_points,
            bbox_target_s=bbox_neigh_s, bbox_val_s=bbox_val_s, bbox_train_s=bbox_train_s,
            **grid_args, save_dir=grid_dir, fit=False
        )
        bnam_grid_results.append(
            dict(
                model=bnam_model,
                history=history,
                grid_args=grid_args
            )
        )
        if len(history) != 0:
            with open(f'{grid_dir}/history.json', 'w+') as fp:
                json.dump(
                    obj=dict(
                        history=history
                    ),
                    fp=fp
                )

        bnam_dp_s = bnam_model.predict_survival(exp_points).detach().cpu().numpy()
        bnam_neigh_s = bnam_model.predict_survival(all_neighbours).detach().cpu().numpy()

        with open(f'{grid_dir}/pred_points.json', 'w+') as fp:
            json.dump(
                obj=dict(
                    bbox_neigh_s=bbox_neigh_s.tolist(),
                    bbox_dp_s=bbox_dp_s.tolist(),
                    explainer_neigh_s=bnam_neigh_s.tolist(),
                    explainer_dp_s=bnam_dp_s.tolist()
                ),
                fp=fp
            )

    for grid_res in bnam_grid_results:
        grid_dir = res_dir.joinpath(path_from_grid_args(grid_res['grid_args']))
        grid_dir.mkdir(exist_ok=True, parents=True)
        bnam_model = grid_res['model']

        fkeys = train_features.keys()
        train_feature_w_y = train_features.copy()
        train_feature_w_y['y_time'] = train_times
        train_feature_w_y['y_pred_bbox'] = np.array([s.y for s in pred_surv_fn(train_features)]) \
            .sum(axis=-1)
        train_feature_w_y['y_pred_exp'] = bnam_model.predict_survival(train_features).detach().cpu().numpy() \
            .sum(axis=-1)

        fig, axes = plt.subplots(nrows=3, ncols=len(fkeys), figsize=(9, 3))
        for row_i, y_key in enumerate(['y_time', 'y_pred_bbox', 'y_pred_exp']):
            for col_i, fname in enumerate(fkeys):
                ax = axes[row_i, col_i]
                ordered_df = train_feature_w_y.sort_values(fname)
                ax.scatter(ordered_df[fname], ordered_df[y_key])
                ax.set_xlabel(fname)
                ax.set_ylabel(y_key)
        plt.tight_layout()
        fig.savefig(f"{grid_dir}/shape_functions_ds.png")
        fig.clf()
        if nam_claz in [BNAMImp1, BNAMImp2, BaselineNAM]:
            draw_shape_functions(
                ds=train_features.to_numpy(),
                # funcs=[lambda x: nam_nn(torch.tensor(x, dtype=torch.float32)).flatten().detach().numpy()
                #        for nam_nn in bnam_model.nam.feature_nns],
                funcs=bnam_model,
                fnames=list(train_features.keys()),
                categories_dict=categories_dict,
                shift_mean=False if nam_claz in [BNAMImp1] else True,
                derivative=False
            )
            plt.suptitle(grid_res['grid_args']['optimizer_args'])
            plt.tight_layout()
            plt.savefig(f"{grid_dir}/shape_functions.png")
            plt.clf()
        elif nam_claz in [BaselineImportancesMLP]:
            pass
        else:
            raise Exception(f"Unexpected claz = {nam_claz.__name__}")

        if len(history) != 0:
            mnames = ['loss', 'cindex']
            rename_m_dict = {'target': 'target', 'train': 'background', 'val': 'explainable points'}
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(len(mnames) * 3, 5))
            for ax, mname in zip(axes, mnames):
                for key in ['target', 'val', 'train']:
                    label_key = rename_m_dict[key]
                    ax.plot(
                        [metric_dict[f"{key}_{mname}"] for metric_dict in grid_res['history']],
                        label=f"{label_key} history"
                    )
                ax.set_xlabel('epoch')
                ax.set_ylabel(mname)
                ax.legend()
            plt.tight_layout()
            plt.savefig(f"{grid_dir}/metrics_history.png")
            plt.clf()

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 2 * 3))
        for ax, s, label in zip(axes, [bbox_dp_s[:5], bnam_dp_s[:5]], ['bbox', 'bnam']):
            draw_surv_yo_w_ax(
                time_points=bnam_model.unique_times_,
                pred_surv=np.array([StepFunction(x=bnam_model.event_times_, y=sample) for sample in s]),
                ax=ax,
                draw_args=[dict()] * len(bbox_dp_s),
                actual_et=[None] * len(bbox_dp_s)
            )
            ax.set_title(label)
        plt.tight_layout()
        plt.savefig(f"{grid_dir}/surv_functions.png")
        plt.clf()


def run_main_st_2(ds_dir: Path, bbox_name: str, nam_claz, torch_bnam_grid: ParameterGrid):
    res_dir = ds_dir.joinpath(f"bbox={bbox_name}").joinpath(nam_claz.__name__.lower() + "_st2")
    res_dir.mkdir(exist_ok=True, parents=True)
    print(res_dir)
    # with open(f"{ds_dir}/config.json") as fp:
    #     config = json.load(fp)

    with open(f"{ds_dir}/dataset.json") as fp:
        ds_json = json.load(fp)

        train_features = pd.DataFrame(ds_json['train_features'])
        train_events = np.array(ds_json['train_events'])
        train_times = np.array(ds_json['train_times'])
        # train_importances = np.array(ds_json['train_importances'])
        train_pairs = Surv.from_arrays(event=train_events, time=train_times)
        all_train = [train_features, train_pairs]

        test_features = pd.DataFrame(ds_json['test_features'])
        test_events = np.array(ds_json['test_events'])
        test_times = np.array(ds_json['test_times'])
        # test_importances = np.array(ds_json['test_importances'])
        test_pairs = Surv.from_arrays(event=test_events, time=test_times)
        all_test = [test_features, test_pairs]

        # train_cl_size = len(train_features) // len(config['cox_coefs_all'])
        # test_cl_size = len(test_features) // len(config['cox_coefs_all'])

        # cox_clusters = [
        #     [
        #         [
        #             train_features[cl_i * train_cl_size:(cl_i + 1) * train_cl_size],
        #             train_pairs[cl_i * train_cl_size:(cl_i + 1) * train_cl_size]
        #         ],
        #         [
        #             train_features[cl_i * test_cl_size:(cl_i + 1) * test_cl_size],
        #             test_pairs[cl_i * test_cl_size:(cl_i + 1) * test_cl_size]
        #         ]
        #     ]
        #     for cl_i, _ in enumerate(config['cox_coefs_all'])
        # ]
    categories_file = ds_dir.joinpath('categories.json')
    if categories_file.exists():
        with open(categories_file) as fp:
            categories_dict = json.load(fp)
            categories_dict = {
                key: dict(zip(sorted(np.unique([*all_train[0][key], *all_test[0][key]])), val))
                for key, val in categories_dict.items()
            }
    else:
        categories_dict = {}

    if bbox_name == 'rf':
        model = RandomSurvivalForest(n_estimators=100, max_samples=min(500, len(all_train[0])), max_depth=8,
                                     random_state=42)
        model.fit(all_train[0], all_train[1])
        pred_surv_fn = model.predict_survival_function
        pred_hazard_fn = model.predict_cumulative_hazard_function
        pred_risk_fn = model.predict
    elif bbox_name == 'beran':
        b_coefs = None
        model = BeranModel(kernel_width=250, kernel_name='gaussian')
        model.fit(X=all_train[0].to_numpy(), b=b_coefs,
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
    with open(res_dir.joinpath('bbox_metrics.json'), 'w+') as fp:
        json.dump(fp=fp, obj=dict(cindex_train_bbox=cindex_train, cindex_test_bbox=cindex_test))

    random_ids = np.linspace(0, len(test_events) - 1, min(10, len(test_events)), dtype=np.int_)
    explain_points_set(
        pred_surv_fn=pred_surv_fn,
        exp_points=test_features.to_numpy()[random_ids],
        exp_events=test_events[random_ids],
        exp_times=test_times[random_ids],
        train_features=train_features, train_times=train_times, train_events=train_events,
        nam_claz=nam_claz, torch_bnam_grid=torch_bnam_grid,
        categories_dict=categories_dict,
        res_dir=res_dir
    )
