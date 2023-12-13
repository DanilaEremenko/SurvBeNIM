import json
import logging

from pathlib import Path
from typing import Dict
import torch
from sklearn.metrics import r2_score
from sksurv.functions import StepFunction
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from core.drawing import draw_points_tsne, draw_surv_yo_w_ax
from survbenim.nam_esimators import BNAMImp1, BNAMImp2, BaselineNAM, BaselineImportancesMLP
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot as plt
from survbenim.utils_drawing import draw_shape_functions


def mse_sf_and_et(y_true, sf_pred, t_deltas):
    sf_pred = sf_pred[:, 1:]
    assert t_deltas.shape == sf_pred.shape
    y_pred = (t_deltas * sf_pred).sum(axis=1)
    return torch.sqrt(((y_true - y_pred) ** 2).mean())


def fit_grid_node(
        nam_claz, nam_args: dict,
        train_features: pd.DataFrame, train_events: np.ndarray, train_times: np.ndarray,
        val_features: np.ndarray, val_events: torch.Tensor, val_times: torch.Tensor,
        criterion, optimizer: str, optimizer_args: dict, mode: str,
        last_layer: str, max_epoch: int, save_dir: Path, fit=True
):
    if criterion in ['mse', 'mse_torch']:
        criterion = mse_sf_and_et
    else:
        raise Exception("Undefined criterion")

    logging.basicConfig()
    logger = logging.getLogger('grad torch bnam optimizer')
    logger.setLevel(level=logging.DEBUG)

    train_features = torch.tensor(train_features.to_numpy(), dtype=torch.float32)
    train_y_events = torch.tensor(train_events)
    train_y_ets = torch.tensor(train_times, dtype=torch.float32)

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
    t_deltas_train = torch.ones((len(train_features), 1)) @ t_deltas[np.newaxis]
    t_deltas_val = torch.ones((len(val_features), 1)) @ t_deltas[np.newaxis]

    history = []

    if mode in ['surv']:
        pred_fn = lambda xps: torch.nan_to_num(
            bnam_model.predict_survival(xps),
            nan=1e-5
        )
    else:
        raise Exception(f"Undefined mode = {mode}")

    batch_size = 4
    best_cindex = 0
    best_f1_r2 = 0
    best_file = str(save_dir.joinpath('best_model.pt'))
    for i in range(max_epoch):
        if not fit:
            history = []
            break
        for batch_start in range(0, len(train_features), batch_size):
            if i == 0:
                break
            optimizer.zero_grad()

            loss_batch = criterion(
                y_true=train_y_ets[batch_start:batch_start + batch_size],
                sf_pred=pred_fn(xps=train_features[batch_start:batch_start + batch_size]),
                t_deltas=t_deltas_train[batch_start:batch_start + batch_size]
            )
            loss_batch.backward()
            optimizer.step()

        bnam_train_f = pred_fn(xps=train_features)
        bnam_val_f = pred_fn(xps=val_features)

        cindex_train = concordance_index_censored(
            event_indicator=train_y_events.detach().cpu().numpy(), event_time=train_y_ets.detach().cpu().numpy(),
            estimate=1 / np.clip(bnam_train_f.mean(axis=-1).detach().cpu().numpy(), a_min=1e-5, a_max=1.)
        )[0]
        cindex_val = concordance_index_censored(
            event_indicator=val_events.detach().cpu().numpy(), event_time=val_times.detach().cpu().numpy(),
            estimate=1 / np.clip(bnam_val_f.mean(axis=-1).detach().cpu().numpy(), a_min=1e-5, a_max=1.)
        )[0]

        loss_train = mse_sf_and_et(
            y_true=train_y_ets, sf_pred=bnam_train_f, t_deltas=t_deltas_train
        )
        loss_val = mse_sf_and_et(
            y_true=val_times, sf_pred=bnam_val_f, t_deltas=t_deltas_val
        )

        train_uncensored = train_y_events == 1
        r2_train = r2_score(
            y_true=train_y_ets[train_uncensored].detach().numpy(),
            y_pred=(t_deltas_train[train_uncensored] * bnam_train_f[train_uncensored][:, 1:])
            .sum(axis=1).detach().numpy()
        )
        val_uncensored = val_events == 1
        r2_val = r2_score(
            y_true=val_times[val_uncensored].detach().numpy(),
            y_pred=(t_deltas_val[val_uncensored] * bnam_val_f[val_uncensored][:, 1:])
            .sum(axis=1).detach().numpy()
        )

        curr_cindex = (cindex_train + cindex_val) / 2
        curr_f1_r2 = 2 * (r2_train * r2_val) / (r2_train + r2_val)

        if curr_f1_r2 > best_f1_r2:
            best_f1_r2 = curr_f1_r2
            print('saving new best state dict..')
            torch.save(obj=bnam_model.nam.state_dict(), f=best_file)

        history.append(
            dict(
                train_loss=float(loss_train),
                val_loss=float(loss_val),
                val_cindex=cindex_val,
                train_cindex=cindex_train,
                train_r2=r2_train,
                val_r2=r2_val,
                train_f1_r2=best_f1_r2,
                val_f1_r2=best_f1_r2,
            )
        )
        logger.debug(f'{len(history)}:train loss           = {loss_train:.4f}')
        logger.debug(f'{len(history)}:val loss             = {loss_val:.4f}')
        logger.debug(f'{len(history)}:train cindex         = {cindex_train:.4f}')
        logger.debug(f'{len(history)}:val cindex           = {cindex_val:.4f}')
        logger.debug(f'{len(history)}:train r2             = {r2_train:.4f}')
        logger.debug(f'{len(history)}:val r2               = {r2_val:.4f}')
        logger.debug(f'{len(history)}:f1 r2                = {curr_f1_r2:.4f}')

    bnam_model.nam.load_state_dict(state_dict=torch.load(best_file))
    bnam_model.nam.eval()

    return bnam_model, history


def explain_points_set(
        test_features: np.ndarray, test_events: bool, test_times: float,
        train_features: pd.DataFrame, train_events: np.ndarray, train_times: np.ndarray,
        torch_bnam_grid: ParameterGrid, nam_claz,
        categories_dict: Dict[str, Dict[float, str]],
        res_dir: Path
):
    res_dir.mkdir(exist_ok=True, parents=True)

    device = 'cpu'
    torch.set_default_device(device=device)
    draw_points_tsne(
        pt_groups=[
            train_features,
            test_features
        ],
        names=[
            'train background',
            'exp points'
        ],
        colors=[None] * 2,
        path=f'{res_dir}/tsne_exp_point.png'
    )

    bnam_grid_results = []

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
            val_features=test_features, val_events=test_events, val_times=test_times,
            **grid_args, save_dir=grid_dir, fit=True
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

        bnam_dp_s = bnam_model.predict_survival(test_features).detach().cpu().numpy()

        with open(f'{grid_dir}/pred_points.json', 'w+') as fp:
            json.dump(
                obj=dict(explainer_dp_s=bnam_dp_s.tolist()),
                fp=fp
            )

    for grid_res in bnam_grid_results:
        grid_dir = res_dir.joinpath(path_from_grid_args(grid_res['grid_args']))
        grid_dir.mkdir(exist_ok=True, parents=True)
        bnam_model = grid_res['model']

        fkeys = train_features.keys()
        train_feature_w_y = train_features.copy()
        train_feature_w_y['y_time'] = train_times
        train_feature_w_y['y_pred_exp'] = bnam_model.predict_survival(train_features).detach().cpu().numpy() \
            .sum(axis=-1)

        fig, axes = plt.subplots(nrows=3, ncols=len(fkeys), figsize=(9, 3))
        for row_i, y_key in enumerate(['y_time', 'y_pred_exp']):
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
            mnames = ['loss', 'cindex', 'r2', 'f1_r2']
            rename_m_dict = {'train': 'train&background', 'val': 'test'}
            fig, axes = plt.subplots(nrows=len(mnames), ncols=1, figsize=(len(mnames) * 3, 5))
            for ax, mname in zip(axes, mnames):
                for key in ['train', 'val']:
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

        draw_surv_yo_w_ax(
            time_points=bnam_model.unique_times_,
            pred_surv=np.array([StepFunction(x=bnam_model.event_times_, y=sample) for sample in bnam_dp_s[:5]]),
            ax=plt.gca(),
            draw_args=[dict()] * len(bnam_dp_s),
            actual_et=[None] * len(bnam_dp_s)
        )
        plt.title('bnam')
        plt.tight_layout()
        plt.savefig(f"{grid_dir}/surv_functions.png")
        plt.clf()


def run_main_st_3(ds_dir: Path, nam_claz, torch_bnam_grid: ParameterGrid):
    res_dir = ds_dir.joinpath(nam_claz.__name__.lower() + "_st3")
    res_dir.mkdir(exist_ok=True, parents=True)
    print(res_dir)

    with open(f"{ds_dir}/dataset.json") as fp:
        ds_json = json.load(fp)

        train_features = pd.DataFrame(ds_json['train_features'])
        train_events = np.array(ds_json['train_events'])
        train_times = np.array(ds_json['train_times'])

        train_pairs = Surv.from_arrays(event=train_events, time=train_times)
        all_train = [train_features, train_pairs]

        test_features = pd.DataFrame(ds_json['test_features'])
        test_events = np.array(ds_json['test_events'])
        test_times = np.array(ds_json['test_times'])

        test_pairs = Surv.from_arrays(event=test_events, time=test_times)
        all_test = [test_features, test_pairs]

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

    random_test_ids = np.linspace(0, len(test_events) - 1, min(100, len(test_events)), dtype=np.int_)
    random_train_ids = np.linspace(0, len(train_events) - 1, min(1000, len(train_events)), dtype=np.int_)

    explain_points_set(
        test_features=test_features.to_numpy()[random_test_ids],
        test_events=test_events[random_test_ids],
        test_times=test_times[random_test_ids],

        train_features=train_features.iloc[random_train_ids],
        train_events=train_events[random_train_ids],
        train_times=train_times[random_train_ids],

        nam_claz=nam_claz, torch_bnam_grid=torch_bnam_grid,
        categories_dict=categories_dict,
        res_dir=res_dir
    )
