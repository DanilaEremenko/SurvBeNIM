import json
from pathlib import Path
from typing import Dict
import sksurv.datasets
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_string_dtype, is_categorical_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
from core.drawing import draw_points_tsne


def prepare_ds(ds):
    x_data, y_data = ds[0], ds[1]
    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in x_data.keys()
        if is_string_dtype(x_data[key]) or is_categorical_dtype(x_data[key])
    }

    for key, coder in le_dict.items():
        x_data[key] = coder.fit_transform(x_data[key])

    with open(f"{res_dir}/categories.json", 'w+') as fp:
        json.dump(fp=fp, obj={key: val.classes_.tolist() for key, val in le_dict.items()})

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8)
    return [x_train, y_train], [x_test, y_test]


avilabel_datasets = {
    'gbsg2': lambda: prepare_ds(sksurv.datasets.load_gbsg2()),
    'whas500': lambda: prepare_ds(sksurv.datasets.load_whas500()),
    'veterans': lambda: prepare_ds(sksurv.datasets.load_veterans_lung_cancer()),
    'breast_cancer': lambda: prepare_ds(sksurv.datasets.load_breast_cancer())
}

ds_name = 'veterans'
assert ds_name in avilabel_datasets.keys()

res_dir = Path(f'bnam/real_ds={ds_name}')
res_dir.mkdir(exist_ok=True, parents=True)

all_train, all_test = avilabel_datasets[ds_name]()

scaler = MinMaxScaler(feature_range=(1e-5, 1 - 1e-5))
scaler.fit(pd.concat([all_train[0], all_test[0]]))
feature_keys = all_test[0].keys()
all_train[0] = pd.DataFrame(scaler.transform(all_train[0]), columns=feature_keys)
all_test[0] = pd.DataFrame(scaler.transform(all_test[0]), columns=feature_keys)

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
        test_features=test_features.to_dict(orient='dict'),
        test_events=test_events.tolist(),
        test_times=test_times.tolist()
    ))

draw_points_tsne(
    pt_groups=[all_train[0], all_test[0]],
    names=['train', 'test'],
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
