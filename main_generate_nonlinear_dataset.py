import json
from pathlib import Path
import sklearn.utils
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sksurv.util import Surv
from core.drawing import draw_points_tsne


def get_func_from_str(s):
    if s == 'x^2':
        return lambda x: x ** 2
    elif s == '|x|':
        return lambda x: np.abs(x)
    elif s == 'relu':
        return lambda x: np.maximum(x, 0)
    elif s == 'no_imp':
        return lambda x: 1e-20 * x
    else:
        raise Exception(f"Undefined func name = {s}")


def get_nonlinear_data(coefs: np.ndarray, size: int, censored_part=0.2):
    assert 0 <= censored_part < 0.5
    x = np.random.random((size, len(coefs)))
    x = MinMaxScaler(feature_range=(-5, 5)).fit_transform(x)
    funcs = [get_func_from_str(coef) for coef in coefs]
    event_times = np.array([func(x[:, i]) for i, func in enumerate(funcs)]).sum(axis=0)
    events = np.ones(len(event_times))
    censored_ids = np.random.choice(list(range(size)), int(size * censored_part))
    events[censored_ids] = 0

    x, y = x, Surv.from_arrays(event=events, time=event_times)
    order = x[:, 0].argsort()
    # order = event_times.argsort()

    x_sorted = x[order]
    y_sorted = y[order]
    test_mask = np.array([0.7 * len(x_sorted) < i < 0.8 * len(x_sorted) for i in range(len(x_sorted))], dtype=bool)
    train_mask = ~test_mask

    x_train, y_train = sklearn.utils.shuffle(x_sorted[train_mask], y_sorted[train_mask])
    x_test, y_test = sklearn.utils.shuffle(x_sorted[test_mask], y_sorted[test_mask])

    x_to_df = lambda x: pd.DataFrame(x, columns=[f'f{i + 1}' for i in range(len(coefs))])

    # wrong way
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    return [x_to_df(x_train), y_train], [x_to_df(x_test), y_test]


additive_funcs_all = np.array(
    [
        [*['x^2', 'relu', '|x|'], *['no_imp'] * 2]
    ]
)
cl_size = 400

res_dir = Path(
    f'bnam/additive_funcs_fair_split_dataset_clnum={len(additive_funcs_all)}_fnum={len(additive_funcs_all[0])}_cl_size={cl_size}')
res_dir.mkdir(exist_ok=True, parents=True)

cox_clusters = [get_nonlinear_data(coefs=cox_coefs, size=cl_size) for cox_coefs in additive_funcs_all]

cox_clusters = [
    (
        [cox_cluster[0][0] + 2.0 / len(cox_clusters) * cl_i, cox_cluster[0][1]],
        [cox_cluster[1][0] + 2.0 / len(cox_clusters) * cl_i, cox_cluster[1][1]]
    )
    for cl_i, cox_cluster in enumerate(cox_clusters)
]

all_train = [
    pd.concat([cox_cluster[0][0] for cox_cluster in cox_clusters]),
    np.hstack([cox_cluster[0][1] for cox_cluster in cox_clusters])
]

all_test = [
    pd.concat([cox_cluster[1][0] for cox_cluster in cox_clusters]),
    np.hstack([cox_cluster[1][1] for cox_cluster in cox_clusters])
]

# Use SurvLimeExplainer class to find the feature importance
training_events = np.array([event for event, _ in all_train[1]])
training_times = np.array([time for _, time in all_train[1]])
training_features = all_train[0]

test_events = np.array([event for event, _ in all_test[1]])
test_times = np.array([time for _, time in all_test[1]])
test_features = all_test[0]

with open(f'{res_dir}/dataset.json', 'w+') as fp:
    json.dump(fp=fp, obj=dict(
        train_features=training_features.to_dict(orient='dict'),
        train_events=training_events.tolist(),
        train_times=training_times.tolist(),
        train_importances=additive_funcs_all.repeat(len(training_features) / len(additive_funcs_all), 0).tolist(),
        test_features=test_features.to_dict(orient='dict'),
        test_events=test_events.tolist(),
        test_times=test_times.tolist(),
        test_importances=additive_funcs_all.repeat(len(test_features) / len(additive_funcs_all), 0).tolist(),
    ))

with open(f"{res_dir}/config.json", 'w+') as fp:
    json.dump(
        obj=dict(cox_coefs_all=additive_funcs_all.tolist(), cl_size=cl_size),
        fp=fp
    )

draw_points_tsne(
    pt_groups=[training_features, test_features],
    names=['train points', 'test_points'],
    colors=[None] * 2,
    path=f"{res_dir}/tsne.png"
)

fig, axes = plt.subplots(nrows=1, ncols=len(training_features.keys()), figsize=(3 * len(training_features.keys()), 3))

for fi, fname in enumerate(training_features.keys()):
    ax = axes[fi]
    feature_order = np.argsort(training_features.to_numpy()[:, fi])
    ax.scatter(training_features.iloc[feature_order][fname], training_times[feature_order], label='train')
    feature_order = np.argsort(test_features.to_numpy()[:, fi])
    ax.scatter(test_features.iloc[feature_order][fname], test_times[feature_order], label='test')
    ax.set_xlabel(fname)
    ax.set_ylabel('time')
    ax.legend()

fig.tight_layout()
fig.savefig(f"{res_dir}/dependecies.png")
fig.clf()
