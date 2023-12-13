import json
import logging
from pathlib import Path
from typing import List

import torch
from sklearn.metrics import pairwise_distances
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import nelson_aalen_estimator, kaplan_meier_estimator
from sksurv.util import Surv
from core.cox_wrapper import CoxFairBaseline
from core.drawing import draw_points_tsne, draw_surv_yo
from survbenim.metrics import mse_torch
from survlimepy.utils.neighbours_generator import NeighboursGenerator
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sksurv.ensemble import RandomSurvivalForest
from matplotlib import pyplot as plt
from survbex.estimators import BeranModel
from survbenim.nam_esimators import BaselineNAM, BNAMImp1, BaselineImportancesMLP
from survbenim.utils_drawing import draw_shape_functions


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
            neighbours, data_point, metric=self.weighted_euclidean_distance
        ).ravel()
        weights = self.kernel_fn(distances)
        w = np.reshape(weights, newshape=(len(neighbours), 1))
        return w


def draw_model_results(
        bnam_model, grid_res: dict,
        neighbours: np.ndarray, bbox_neigh_s: torch.Tensor, bbox_dp_s: torch.Tensor,
        exp_point: np.ndarray, exp_time: float, pred_surv_fn,
        train_features: pd.DataFrame, train_times: np.ndarray,
        res_dir: Path
):
    history = grid_res['history']

    bnam_dp_s = bnam_model.predict_survival(exp_point).detach().numpy()
    bnam_neigh_s = bnam_model.predict_survival(neighbours).detach().numpy()

    fkeys = train_features.keys()
    train_feature_w_y = train_features.copy()
    train_feature_w_y['y_time'] = train_times
    train_feature_w_y['y_pred_bbox'] = np.array([s.y for s in pred_surv_fn(train_features)]).sum(axis=-1)
    train_feature_w_y['y_pred_exp'] = bnam_model.predict_survival(train_features).detach().numpy().sum(axis=-1)

    fig, axes = plt.subplots(nrows=3, ncols=len(fkeys), figsize=(9, 3))
    for row_i, y_key in enumerate(['y_time', 'y_pred_bbox', 'y_pred_exp']):
        for col_i, fname in enumerate(fkeys):
            ax = axes[row_i, col_i]
            ordered_df = train_feature_w_y.sort_values(fname)
            ax.scatter(ordered_df[fname], ordered_df[y_key])
    plt.tight_layout()
    fig.savefig(f"{res_dir}/shape_functions_ds.png")
    fig.clf()

    if isinstance(bnam_model, (BaselineNAM, BNAMImp1)):
        if isinstance(bnam_model, BaselineNAM):
            shift_mean = True
            imp_func = lambda arr: arr.std()
        elif isinstance(bnam_model, BNAMImp1):
            shift_mean = False
            imp_func = lambda arr: arr.mean()
        else:
            raise Exception(f"Undefined class {bnam_model.__class__}")

        fxs, fys = draw_shape_functions(
            ds=train_features.to_numpy(),
            # funcs=[lambda x: nam_nn(torch.tensor(x, dtype=torch.float32)).flatten().detach().numpy()
            #        for nam_nn in bnam_model.nam.feature_nns],
            funcs=bnam_model,
            fnames=list(train_features.keys()),
            shift_mean=shift_mean,
            categories_dict={},
            derivative=False
        )
        plt.savefig(f"{res_dir}/shape_functions.png")
        plt.clf()

        # f_imps = []
        # for fy, fx, exp_fx in zip(fxs, fys, exp_point[0]):
        #     derr_pt_id = np.argmin(np.abs(fx - exp_fx))
        #     if derr_pt_id == len(fy) - 1:
        #         derr_pt_id -= 1
        #     elif derr_pt_id == 0:
        #         derr_pt_id += 1
        #     else:
        #         pass
        #     f_imps.append(np.abs(fy[derr_pt_id - 1] - fy[derr_pt_id + 1]))
        # f_imps = np.array(f_imps)
        neighbours_ranges = [(neighbours[:, dim].min(), neighbours[:, dim].max()) for dim in range(len(fxs))]
        f_imps = np.array([imp_func(fy[(nrange[0] < fx) & (fx < nrange[1])])
                           for nrange, fx, fy in zip(neighbours_ranges, fxs, fys)])
    elif isinstance(bnam_model, BaselineImportancesMLP):
        f_imps = bnam_model.nn(torch.tensor(exp_point, dtype=torch.float32)).detach().cpu().numpy().flatten()
    else:
        raise Exception(f"Undefined model = {bnam_model}")

    with open(f'{res_dir}/res.json', 'w+') as fp:
        json.dump(
            obj=dict(
                importances=f_imps.tolist(),
                train_history=history,
                bbox_neigh_s=bbox_neigh_s.tolist(),
                bbox_dp_s=bbox_dp_s.tolist(),
                explainer_neigh_s=bnam_neigh_s.tolist(),
                explainer_dp_s=bnam_dp_s.tolist()
            ),
            fp=fp
        )
    if len(history) != 0:
        mnames = ['loss', 'cindex']
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(len(mnames) * 3, 3))
        for ax, mname in zip(axes, mnames):
            for key in ['target', 'val', 'background']:
                ax.plot([metric_dict[f"{mname}({key})"] for metric_dict in grid_res['history']],
                        label=f"{key} history")
            ax.set_xlabel('epoch')
            ax.set_ylabel(mname)
            ax.legend()
            ax.set_title(f"{grid_res['optimizer']} {grid_res['optimizer_args']}")
        plt.tight_layout()
        plt.savefig(f"{res_dir}/metrics_history.png")
        plt.clf()

    plt.figure(figsize=(6, 4))
    draw_surv_yo(
        time_points=bnam_model.unique_times_,
        pred_surv=np.array([StepFunction(x=bnam_model.event_times_, y=sample) for sample in bbox_dp_s]),
        draw_args=[dict(label='bbox')],
        actual_et=[exp_time]
    )

    draw_surv_yo(
        time_points=bnam_model.unique_times_,
        pred_surv=np.array([StepFunction(x=bnam_model.event_times_, y=sample) for sample in bnam_dp_s]),
        draw_args=[dict(label='bnam')],
        actual_et=[exp_time]
    )
    plt.title(f"{grid_res['optimizer']} {grid_res['optimizer_args']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{res_dir}/surv_functions.png")
    plt.clf()


def fit_grid_node(
        train_features, train_events, train_times,
        bbox_train_s: torch.Tensor,
        bbox_target_s: torch.Tensor, target_features: torch.Tensor, target_point: torch.Tensor,
        bbox_val_s: torch.Tensor, val_features: torch.Tensor,
        criterion: str, optimizer: str, optimizer_args: dict, mode: str,
        v_mode: str, max_epoch: int, radius: float,
        select_criterion: str,
        bnam_model,
        res_dir: Path, fit=True
):
    if criterion in ['mse', 'mse_torch']:
        criterion = mse_torch
    else:
        raise Exception(f"Undefined criterion = {criterion}")

    res_dir.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger('grad torch bnam optimizer')
    logger.setLevel(level=logging.DEBUG)

    train_y_events = torch.tensor(train_events)
    train_y_ets = torch.tensor(train_times)

    bnam_model.fit(
        X=train_features.to_numpy(),
        y_event_times=train_y_ets,
        y_events=train_y_events
    )

    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=bnam_model.parameters(), **optimizer_args)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(params=bnam_model.parameters(), **optimizer_args)
    else:
        raise Exception(f"Unexpected optimizer = {optimizer}")

    # neighbours normalization
    # for fi in range(neighbours.shape[1]):
    #     neighbours[:, fi] -= neighbours[:, fi].min()
    #     neighbours[:, fi] /= neighbours[:, fi].max()

    t_deltas = bnam_model.event_times_[1:] - bnam_model.event_times_[:-1]
    t_deltas_train = torch.ones((len(train_features), 1)) @ t_deltas[np.newaxis]
    t_deltas_target = torch.ones((len(target_features), 1)) @ t_deltas[np.newaxis]
    t_deltas_val = torch.ones((len(val_features), 1)) @ t_deltas[np.newaxis]

    samples_weighter = SamplesWeighter(training_features=train_features.to_numpy(), kernel_width=radius)

    w_target = torch.tensor(
        samples_weighter.get_weights(
            neighbours=target_features,
            data_point=target_point
        ),
        dtype=torch.float32
    )

    w_val = torch.tensor(
        samples_weighter.get_weights(
            neighbours=val_features,
            data_point=target_point
        ),
        dtype=torch.float32
    )

    history = []

    if mode in ['surv']:
        bbox_train_f = bbox_train_s
        bbox_target_f = bbox_target_s
        bbox_val_f = bbox_val_s
        pred_fn = lambda xps: torch.nan_to_num(
            bnam_model.predict_survival(xps),
            nan=1e-5
        )
        v_train = torch.ones((len(bbox_train_f), len(bnam_model.event_times_)))
        v_target = torch.ones((len(bbox_target_f), len(bnam_model.event_times_)))
        v_val = torch.ones((len(bbox_val_f), len(bnam_model.event_times_)))
    else:
        raise Exception(f"Undefined mode = {mode}")

    best_criterions = {
        'loss(background)': 1e5,
        'loss(val)': 1e5,
        'loss(val+background)': 1e5,
        'loss(target)': 1e5,
        'loss(target+background)': 1e5,

        'cindex(background)': 0,
        'cindex(val)': 0,
        'cindex(val+background)': 0,
        'cindex(target)': 0,
        'cindex(target+background)': 0
    }

    assert select_criterion in best_criterions.keys(), f'select_criterion={select_criterion} not exists'

    best_file = str(res_dir.joinpath(f'best_file_criterion={select_criterion}.pt'))

    batch_size = 16
    for i in range(max_epoch):
        if not fit:
            break
        for batch_start in range(0, len(target_features), batch_size):
            optimizer.zero_grad()
            loss_target = criterion(
                y_true=bbox_target_f[batch_start:batch_start + batch_size],
                y_pred=pred_fn(xps=target_features[batch_start:batch_start + batch_size]),
                t_deltas=t_deltas_target[batch_start:batch_start + batch_size],
                v=v_target[batch_start:batch_start + batch_size],
                sample_weight=w_target[batch_start:batch_start + batch_size],
                b=np.ones(train_features.shape[1])
            )
            loss_target.backward()
            optimizer.step()
        bnam_train_f = pred_fn(xps=train_features)
        bnam_target_f = pred_fn(xps=target_features)
        bnam_val_f = pred_fn(xps=val_features)

        cindex_background = concordance_index_censored(
            event_indicator=train_y_events, event_time=train_y_ets,
            estimate=1 / np.clip(bnam_train_f.mean(axis=-1).detach().numpy(), a_min=1e-5, a_max=np.inf)
        )[0]
        cindex_target = concordance_index_censored(
            event_indicator=np.ones(len(bbox_target_f), dtype=np.bool_), event_time=bbox_target_f.mean(axis=-1),
            estimate=1 / np.clip(bnam_target_f.mean(axis=-1).detach().numpy(), a_min=1e-5, a_max=np.inf)
        )[0]
        cindex_val = concordance_index_censored(
            event_indicator=np.ones(len(bbox_val_f), dtype=np.bool_), event_time=bbox_val_f.mean(axis=-1),
            estimate=1 / np.clip(bnam_val_f.mean(axis=-1).detach().numpy(), a_min=1e-5, a_max=np.inf)
        )[0]

        loss_background = criterion(y_true=bbox_train_s, y_pred=bnam_train_f, t_deltas=t_deltas_train,
                                    v=v_train, sample_weight=torch.ones(len(bbox_train_f)),
                                    b=torch.ones(train_features.shape[1]))
        loss_target = criterion(y_true=bbox_target_f, y_pred=bnam_target_f, t_deltas=t_deltas_target,
                                v=v_target, sample_weight=w_target,
                                b=torch.ones(train_features.shape[1]))
        loss_val = criterion(y_true=bbox_val_f, y_pred=bnam_val_f, t_deltas=t_deltas_val,
                             v=v_val, sample_weight=w_val,
                             b=torch.ones(val_features.shape[1]))

        curr_criterions = {
            'loss(background)': float(loss_background),
            'loss(val)': float(loss_val),
            'loss(val+background)': float(loss_val + loss_background),
            'loss(target)': float(loss_target),
            'loss(target+background)': float(loss_target + loss_background),

            'cindex(background)': cindex_background,
            'cindex(val)': cindex_val,
            'cindex(val+background)': cindex_val + cindex_background,
            'cindex(target)': cindex_target,
            'cindex(target+background)': cindex_target + cindex_background
        }

        for curr_criterion_name, curr_criterion in curr_criterions.items():
            if 'cindex' in curr_criterion_name:
                continue

            best_file = str(res_dir.joinpath(f'best_file_criterion={curr_criterion_name}.pt'))
            best_criterion = best_criterions[curr_criterion_name]
            if 'cindex' in curr_criterion_name:
                condition = curr_criterion > best_criterion
            elif 'loss' in curr_criterion_name:
                condition = curr_criterion < best_criterion
            else:
                raise Exception(f"Undefined criterion name {curr_criterion_name}")

            if condition:
                best_criterions[curr_criterion_name] = curr_criterion
                print('saving new best state dict..')
                torch.save(obj=bnam_model.nam.state_dict(), f=best_file)

        history.append({**curr_criterions, 'select_criterion': best_criterions[select_criterion]})
        logger.debug(f'{len(history)}:train loss           = {loss_background:.4f}')
        logger.debug(f'{len(history)}:target loss          = {loss_target:.4f}')
        logger.debug(f'{len(history)}:val loss             = {loss_val:.4f}')

        logger.debug(f'{len(history)}:train cindex         = {cindex_background:.4f}')
        logger.debug(f'{len(history)}:target cindex        = {cindex_target:.4f}')
        logger.debug(f'{len(history)}:val cindex           = {cindex_val:.4f}')

    if fit:
        with open(res_dir.joinpath('history.json'), 'w+') as fp:
            json.dump(fp=fp, obj=history)
    else:
        with open(res_dir.joinpath('history.json'), 'r') as fp:
            history = json.load(fp=fp)

    return best_file, history


def grid_args_to_str(grid_args):
    grid_str = ''
    for key, val in grid_args.items():
        if isinstance(val, dict):
            grid_str += grid_args_to_str(val)
        else:
            grid_str += f"{key}={val},"
    return grid_str


def explain_exp_point(exp_point: np.ndarray, exp_event: bool, exp_time: float, pred_surv_fn, cox_clusters,
                      train_features: pd.DataFrame, train_events: np.ndarray, train_times: np.ndarray,
                      bnam_claz, bnam_args: dict, radius: float, param_grid: ParameterGrid,
                      res_dir: Path):
    res_dir.mkdir(exist_ok=True, parents=True)
    torch_bnam_grid = param_grid

    n_generator = NeighboursGenerator(training_features=train_features, data_row=exp_point,
                                      sigma=radius, random_state=np.random.RandomState(seed=42))
    neighbours = n_generator.generate_neighbours(num_samples=100)
    neighbours_val = n_generator.generate_neighbours(num_samples=50)

    draw_points_tsne(
        pt_groups=[
            *[cox_cluster[0][0].to_numpy() for cox_cluster in cox_clusters],
            exp_point,
            neighbours,
            neighbours_val
        ],
        names=[
            *[f'cl{i}' for i, _ in enumerate(cox_clusters)],
            'exp point',
            'neighbours',
            'neighbours_val'
        ],
        colors=[None] * (len(cox_clusters) + 3),
        path=f'{res_dir}/tsne_exp_point.png'
    )

    bnam_grid_results = []

    bbox_neigh_s = pred_surv_fn(neighbours)
    bbox_neigh_s = torch.tensor([step.y for step in bbox_neigh_s], dtype=torch.float32)

    bbox_neigh_val_s = pred_surv_fn(neighbours_val)
    bbox_neigh_val_s = torch.tensor([step.y for step in bbox_neigh_val_s], dtype=torch.float32)

    bbox_train_s = pred_surv_fn(train_features)
    bbox_train_s = torch.tensor([step.y for step in bbox_train_s], dtype=torch.float32)

    bbox_dp_s = pred_surv_fn(exp_point)
    bbox_dp_s = torch.tensor([step.y for step in bbox_dp_s], dtype=torch.float32)

    select_criterion = 'loss(val)'
    for grid_i, grid_args in enumerate(torch_bnam_grid):
        print(f"torch optimization {grid_i + 1}/{len(torch_bnam_grid)}")
        print(f'grid_args = {grid_args}')
        grid_args_str = grid_args_to_str(grid_args)
        grid_res_dir = res_dir.joinpath('grid_models').joinpath(grid_args_str)
        model_path, history = fit_grid_node(
            train_features=train_features, train_events=train_events, train_times=train_times,
            target_features=neighbours, target_point=exp_point,
            bbox_val_s=bbox_neigh_val_s, val_features=neighbours_val,
            bbox_target_s=bbox_neigh_s, bbox_train_s=bbox_train_s,
            **grid_args,
            radius=radius, bnam_model=bnam_claz(**bnam_args),
            select_criterion=select_criterion,
            res_dir=grid_res_dir,
            fit=True
        )
        bnam_grid_results.append(
            dict(
                model_path=model_path,
                history=history,
                **grid_args,
            )
        )
        grid_res = bnam_grid_results[-1]
        bnam_model = bnam_claz(**bnam_args)
        bnam_model.fit(train_features.to_numpy(), train_times, train_events)
        bnam_model.nam.load_state_dict(state_dict=torch.load(grid_res['model_path']))
        bnam_model.nam.eval()
        draw_model_results(
            bnam_model=bnam_model, grid_res=grid_res, neighbours=neighbours, bbox_neigh_s=bbox_neigh_s,
            bbox_dp_s=bbox_dp_s, exp_point=exp_point, exp_time=exp_time, pred_surv_fn=pred_surv_fn,
            train_features=train_features, train_times=train_times, res_dir=grid_res_dir
        )

    if 'loss' in select_criterion:
        best_grid_i = np.argmin(
            [min([epoch['select_criterion'] for epoch in res['history']]) for res in bnam_grid_results]
        )
    elif 'cindex' in select_criterion:
        best_grid_i = np.argmax(
            [max([epoch['select_criterion'] for epoch in res['history']]) for res in bnam_grid_results]
        )
    else:
        raise Exception("Undefined criterion select criterion")

    grid_res = bnam_grid_results[best_grid_i]

    bnam_model = bnam_claz(**bnam_args)
    bnam_model.fit(train_features.to_numpy(), train_times, train_events)
    bnam_model.nam.load_state_dict(state_dict=torch.load(grid_res['model_path']))
    bnam_model.nam.eval()

    draw_model_results(
        bnam_model=bnam_model, grid_res=grid_res, neighbours=neighbours, bbox_neigh_s=bbox_neigh_s,
        bbox_dp_s=bbox_dp_s, exp_point=exp_point, exp_time=exp_time, pred_surv_fn=pred_surv_fn,
        train_features=train_features, train_times=train_times, res_dir=res_dir
    )


def run_main_st_1_nams(
        exp_dir: Path,
        bbox_name: str,
        bnam_claz,
        bnam_args,
        test_ids: List[int],
        radius: float,
        param_grid: ParameterGrid
):
    res_dir = exp_dir.joinpath(f"bbox={bbox_name},radius={radius}").joinpath(f'{bnam_claz.__name__.lower()}_st_1')
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
            cox_clusters=cox_clusters, pred_surv_fn=pred_surv_fn,
            exp_point=exp_point[np.newaxis], exp_event=exp_event, exp_time=exp_time,
            train_features=train_features, train_times=train_times, train_events=train_events,
            bnam_claz=bnam_claz, bnam_args=bnam_args, radius=radius,
            param_grid=param_grid,
            res_dir=res_dir.joinpath(f"pt={pt_i}")
        )
