import json
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from core.cox_generator import CoxGenerator
from core.drawing import draw_points_tsne


def get_cox_data(coefs: np.ndarray, size: int):
    cox_generator = CoxGenerator(coefs=coefs)
    x_cox_train, x_cox_test, y_cox_train, y_cox_test = train_test_split(
        *cox_generator.generate_data(size=cl_size, censored_part=0.2),
        train_size=0.7
    )

    x_cox_train = pd.DataFrame(x_cox_train, columns=[f'f{i + 1}' for i in range(len(coefs))])
    x_cox_test = pd.DataFrame(x_cox_test, columns=[f'f{i + 1}' for i in range(len(coefs))])

    return [x_cox_train, y_cox_train], [x_cox_test, y_cox_test]


cox_coefs_all = np.array(
    [
        [0.50, 0.25, 0.12, *[0] * 17],
        [*[0] * 17, 0.12, 0.25, 0.50],

    ]
)

# no_imp = 1e-20
# cox_coefs_all = np.array(
#     [
#         sorted([*[0.80, 0.70, 0.40, 0.35], *[no_imp] * 16], key=lambda x: random.random())
#         for i in range(10)
#     ]
# )

cl_size = 200

res_dir = Path(f'bnam/cox_dataset_clnum={len(cox_coefs_all)}_fnum={len(cox_coefs_all[0])}_cl_size={cl_size}')
res_dir.mkdir(exist_ok=True, parents=True)

cox_clusters = [get_cox_data(coefs=cox_coefs, size=cl_size) for cox_coefs in cox_coefs_all]

cox_clusters = [
    (
        [cox_cluster[0][0] + len(cox_coefs_all) / len(cox_clusters) * cl_i, cox_cluster[0][1]],
        [cox_cluster[1][0] + len(cox_coefs_all) / len(cox_clusters) * cl_i, cox_cluster[1][1]]
    )
    for cl_i, cox_cluster in enumerate(cox_clusters)
]

for cox_cluster in cox_clusters:
    test_x, test_y = cox_cluster[1]
    cl_centroid = test_x.mean()
    distances = np.mean((test_x - cl_centroid) ** 2, axis=1)
    cl_order = np.argsort(distances)
    cox_cluster[1][0] = test_x.iloc[cl_order]
    cox_cluster[1][1] = test_y[cl_order]

all_train = [
    pd.concat([cox_cluster[0][0] for cox_cluster in cox_clusters]),
    np.hstack([cox_cluster[0][1] for cox_cluster in cox_clusters])
]

all_test = [
    pd.concat([cox_cluster[1][0] for cox_cluster in cox_clusters]),
    np.hstack([cox_cluster[1][1] for cox_cluster in cox_clusters])
]

scaler = MinMaxScaler(feature_range=(1e-5, 1 - 1e-5))
scaler.fit(pd.concat([all_train[0], all_test[0]]))
feature_keys = all_test[0].keys()
all_train[0] = pd.DataFrame(scaler.transform(all_train[0]), columns=feature_keys)
all_test[0] = pd.DataFrame(scaler.transform(all_test[0]), columns=feature_keys)

cox_clusters = [
    (
        [pd.DataFrame(scaler.transform(cox_cluster[0][0]), columns=feature_keys), cox_cluster[0][1]],
        [pd.DataFrame(scaler.transform(cox_cluster[1][0]), columns=feature_keys), cox_cluster[1][1]]
    )
    for cox_cluster in cox_clusters
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
        train_features=training_features.to_dict(orient='records'),
        train_events=training_events.tolist(),
        train_times=training_times.tolist(),
        train_importances=cox_coefs_all.repeat(len(training_features) / len(cox_coefs_all), 0).tolist(),
        test_features=test_features.to_dict(orient='records'),
        test_events=test_events.tolist(),
        test_times=test_times.tolist(),
        test_importances=cox_coefs_all.repeat(len(test_features) / len(cox_coefs_all), 0).tolist(),
    ))

with open(f"{res_dir}/config.json", 'w+') as fp:
    json.dump(
        obj=dict(cox_coefs_all=cox_coefs_all.tolist(), cl_size=cl_size),
        fp=fp
    )

draw_points_tsne(
    pt_groups=[cox_cluster[0][0] for cox_cluster in cox_clusters],
    names=[f"cl_i={i}" for i, cox_cluster in enumerate(cox_clusters)],
    colors=[None] * len(cox_clusters),
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
