from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
from sksurv.nonparametric import nelson_aalen_estimator
import torch.nn.functional as F


def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ActivationLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        raise NotImplementedError("abstract method called")


class ExULayer(ActivationLayer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        truncated_normal_(self.weight, mean=4.0, std=0.5)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        exu = (x - self.bias) @ torch.exp(self.weight)
        return torch.clip(exu, 0, 1)


class ReLULayer(ActivationLayer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return F.relu((x - self.bias) @ self.weight)


class FeatureNN(torch.nn.Module):
    def __init__(self,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 dropout: float = .5,
                 ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            hidden_layer(shallow_units if i == 0 else hidden_units[i - 1], hidden_units[i])
            for i in range(len(hidden_units))
        ])
        self.layers.insert(0, shallow_layer(1, shallow_units))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(shallow_units if len(hidden_units) == 0 else hidden_units[-1], 1, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.linear(x)


class MyFeatureNN(torch.nn.Module):
    def __init__(self,
                 layers: List[ActivationLayer],
                 layers_args: List[dict],
                 dropout: float = .5,
                 output_size=1
                 ):
        assert len(layers) == len(layers_args)
        super().__init__()
        self.layers = torch.nn.ModuleList([
            layer(**layer_args) for layer, layer_args in zip(layers, layers_args)
        ])
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(layers_args[-1]['out_features'], output_size, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        if x.shape[-1] != self.linear.in_features:
            return self.linear(x @ torch.ones(x.shape[-1], self.linear.in_features))
        else:
            return self.linear(x)


class MyNeuralAdditiveModel(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 layers: List[ActivationLayer],
                 layers_args: List[dict],
                 feature_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 ):
        super().__init__()
        self.input_size = input_size
        assert len(layers) == len(layers_args)
        self.feature_nns = torch.nn.ModuleList([
            MyFeatureNN(
                layers=layers,
                layers_args=layers_args,
                dropout=hidden_dropout
            )
            for i in range(input_size)
        ])
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f_out = torch.cat(self._feature_nns(x), dim=-1)
        f_out = self.feature_dropout(f_out)

        return f_out.sum(axis=-1) + self.bias, f_out

    def _feature_nns(self, x):
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]


class SigmoidLayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return F.sigmoid((x - self.bias) @ self.weight)


def get_last_layer_from_str(last_layer: str):
    if last_layer == 'exu':
        return ExULayer
    elif last_layer == 'sigmoid':
        return SigmoidLayer
    elif last_layer == 'relu':
        return ReLULayer
    else:
        raise Exception(f"Undefined last_layer = {last_layer}")


def get_no_zero_fn_from_str(no_zero_fn: str):
    if no_zero_fn == 'abs':
        return torch.abs
    elif no_zero_fn == 'square':
        return torch.square
    else:
        raise Exception(f"Undefined last_layer = {no_zero_fn}")


class BNAMCommon:
    def __init__(self, kernel_width: float, kernel_name: str, last_layer: str, no_zero_fn: str, log_epsilon=0.001):
        self.log_epsilon = log_epsilon
        self.kernel_width = kernel_width
        self.last_layer = get_last_layer_from_str(last_layer=last_layer)
        self.no_zero_fn = get_no_zero_fn_from_str(no_zero_fn=no_zero_fn)

    def parameters(self):
        return self.nam.parameters()

    def fit(self, X: np.ndarray, y_event_times: np.ndarray, y_events: np.ndarray):
        # data should be normalized
        # for fi in range(X.shape[1]):
        #     assert X[:, fi].min() >= 0
        #     assert X[:, fi].max() <= 1

        sort_idx = np.argsort(y_event_times)
        self.X_sorted = X[sort_idx]
        self.y_ets_sorted = y_event_times[sort_idx]
        self.y_events_sorted = y_events[sort_idx]

        self.X_sorted_torch = torch.tensor(self.X_sorted, dtype=torch.float32)
        self.y_ets_sorted_torch = torch.tensor(y_event_times[sort_idx])
        self.y_events_sorted_torch = torch.tensor(y_events[sort_idx])

        self.event_times_ = np.unique(self.y_ets_sorted)
        self.unique_times_ = self.event_times_

        _, idx_start, count = np.unique(self.y_ets_sorted, return_counts=True, return_index=True)
        self.ids_with_repeats = idx_start + (count - 1)
        self.nam = MyNeuralAdditiveModel(
            input_size=X.shape[-1],
            layers=[ReLULayer, ReLULayer, self.last_layer],
            layers_args=[
                dict(in_features=1, out_features=64),
                dict(in_features=64, out_features=32),
                dict(in_features=32, out_features=512)
            ],
            hidden_dropout=0.0,
            feature_dropout=0.0
        )

    def _StxTorchOnKernels(self, kernel_preds):
        w_cumsum = torch.cumsum(kernel_preds, dim=1)
        shifted_w_cumsum = w_cumsum - kernel_preds
        ones = torch.ones_like(shifted_w_cumsum)
        anomaly_mask = torch.isclose(shifted_w_cumsum, ones) | torch.isclose(w_cumsum, ones)
        shifted_w_cumsum[anomaly_mask] = 0.0
        w_cumsum[anomaly_mask] = 0.0

        xi = torch.log(1.0 - w_cumsum) - torch.log(1.0 - shifted_w_cumsum)

        filtered_xi = self.y_events_sorted_torch.unsqueeze(0) * xi

        hazards = torch.cumsum(filtered_xi, dim=1)
        return torch.exp(hazards)

    def predict_survival(self, xps: np.ndarray) -> np.ndarray:
        raise Exception("Not implemented")


class BNAMImp1(BNAMCommon):
    def predict_survival(self, xps: np.ndarray) -> np.ndarray:
        if isinstance(xps, pd.DataFrame):
            xps = xps.to_numpy()
        xps_torch = torch.tensor(xps, dtype=torch.float32)

        nam_b_agg, nam_b = self.nam(xps_torch)
        nam_b = self.no_zero_fn(nam_b)
        xps_torch_rep = xps_torch.repeat_interleave(len(self.X_sorted_torch), 0)
        b_torch_rep = nam_b.repeat_interleave(len(self.X_sorted_torch), 0)
        x_train_rep = self.X_sorted_torch.repeat(len(xps_torch), 1)
        kernel_preds = self.kernel_width * b_torch_rep * self.no_zero_fn(xps_torch_rep - x_train_rep)
        kernel_preds = torch.exp(-torch.mean(kernel_preds, axis=-1)) \
            .reshape(len(xps), len(self.X_sorted_torch))

        kernel_preds_norm = kernel_preds / kernel_preds.sum(axis=1)[:, None]
        S = self._StxTorchOnKernels(kernel_preds_norm)
        if len(self.ids_with_repeats) != len(self.y_ets_sorted):
            return S[:, self.ids_with_repeats]
        else:
            return S


class BNAMImp2(BNAMCommon):
    def predict_survival(self, xps: np.ndarray) -> np.ndarray:
        if isinstance(xps, pd.DataFrame):
            xps = xps.to_numpy()
        xps_torch = torch.tensor(xps, dtype=torch.float32)
        _, nam_b_xps = self.nam(xps_torch)
        _, nam_b_train = self.nam(self.X_sorted_torch)
        nam_b_xps_rep = nam_b_xps.repeat_interleave(len(nam_b_train), 0)
        nam_b_train_rep = nam_b_train.repeat(len(nam_b_xps), 1)
        # xps_torch_rep = xps_torch.repeat_interleave(len(self.X_sorted_torch), 0)
        # x_train_rep = self.X_sorted_torch.repeat(len(xps_torch), 1)

        kernel_preds = self.kernel_width * torch.abs(nam_b_xps_rep - nam_b_train_rep)
        kernel_preds = torch.exp(-torch.mean(kernel_preds, axis=-1)) \
            .reshape(len(xps), len(self.X_sorted_torch))

        kernel_preds_norm = kernel_preds / kernel_preds.sum(axis=1)[:, None]
        S = self._StxTorchOnKernels(kernel_preds_norm)
        if len(self.ids_with_repeats) != len(self.y_ets_sorted):
            return S[:, self.ids_with_repeats]
        else:
            return S


class BaselineImportancesMLP(BNAMCommon):
    def fit(self, X: np.ndarray, y_event_times: np.ndarray, y_events: np.ndarray):
        # data should be normalized
        # for fi in range(X.shape[1]):
        #     assert X[:, fi].min() >= 0
        #     assert X[:, fi].max() <= 1

        sort_idx = np.argsort(y_event_times)
        self.X_sorted = X[sort_idx]
        self.y_ets_sorted = y_event_times[sort_idx]
        self.y_events_sorted = y_events[sort_idx]

        self.X_sorted_torch = torch.tensor(self.X_sorted, dtype=torch.float32)
        self.y_ets_sorted_torch = torch.tensor(y_event_times[sort_idx])
        self.y_events_sorted_torch = torch.tensor(y_events[sort_idx])

        self.event_times_ = np.unique(self.y_ets_sorted)
        self.unique_times_ = self.event_times_

        _, idx_start, count = np.unique(self.y_ets_sorted, return_counts=True, return_index=True)
        self.ids_with_repeats = idx_start + (count - 1)

        self.nn = MyFeatureNN(
            layers=[ReLULayer, ReLULayer, self.last_layer],
            layers_args=[
                dict(in_features=X.shape[-1], out_features=64),
                dict(in_features=64, out_features=32),
                dict(in_features=32, out_features=512)
            ],
            dropout=0.0,
            output_size=X.shape[-1]
        )

    def parameters(self):
        return self.nn.parameters()

    @property
    def nam(self):
        return self.nn

    def predict_survival(self, xps: np.ndarray) -> np.ndarray:
        if isinstance(xps, pd.DataFrame):
            xps = xps.to_numpy()
        xps_torch = torch.tensor(xps, dtype=torch.float32)

        nn_b = self.nn(xps_torch)[:, 0, :, ]
        nn_b = self.no_zero_fn(nn_b)
        xps_torch_rep = xps_torch.repeat_interleave(len(self.X_sorted_torch), 0)
        b_torch_rep = nn_b.repeat_interleave(len(self.X_sorted_torch), 0)
        x_train_rep = self.X_sorted_torch.repeat(len(xps_torch), 1)
        kernel_preds = self.kernel_width * b_torch_rep * self.no_zero_fn(xps_torch_rep - x_train_rep)
        kernel_preds = torch.exp(-torch.mean(kernel_preds, axis=-1)) \
            .reshape(len(xps), len(self.X_sorted_torch))

        kernel_preds_norm = kernel_preds / kernel_preds.sum(axis=1)[:, None]
        S = self._StxTorchOnKernels(kernel_preds_norm)
        if len(self.ids_with_repeats) != len(self.y_ets_sorted):
            return S[:, self.ids_with_repeats]
        else:
            return S


class BaselineNAM:
    def __init__(self, last_layer: str):
        self.last_layer = get_last_layer_from_str(last_layer)

    def fit(self, X: np.ndarray, y_event_times: np.ndarray, y_events: np.ndarray):
        # data should be normalized
        # for fi in range(X.shape[1]):
        #     assert X[:, fi].min() >= 0
        #     assert X[:, fi].max() <= 1

        sort_idx = np.argsort(y_event_times)
        self.X_sorted = X[sort_idx]
        self.y_ets_sorted = y_event_times[sort_idx]
        self.y_events_sorted = y_events[sort_idx]

        self.X_sorted_torch = torch.tensor(self.X_sorted)
        self.y_ets_sorted_torch = torch.tensor(y_event_times[sort_idx])
        self.y_events_sorted_torch = torch.tensor(y_events[sort_idx])

        _, idx_start, count = np.unique(self.y_ets_sorted, return_counts=True, return_index=True)
        self.ids_with_repeats = idx_start + (count - 1)

        self.nam = MyNeuralAdditiveModel(
            input_size=X.shape[-1],
            layers=[ReLULayer, ReLULayer, self.last_layer],
            layers_args=[
                dict(in_features=1, out_features=64),
                dict(in_features=64, out_features=32),
                dict(in_features=32, out_features=512)
            ],
            hidden_dropout=0.0,
            feature_dropout=0.0
        )

        self.H0 = torch.tensor(nelson_aalen_estimator(event=self.y_events_sorted, time=self.y_ets_sorted)[1],
                               dtype=torch.float32)

        self.event_times_ = np.unique(self.y_ets_sorted)
        self.unique_times_ = self.event_times_

    def parameters(self):
        return self.nam.parameters()

    def predict_survival(self, X: np.ndarray):
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            pass
        else:
            raise Exception(f"Undefined class of X = {X.__class__.__name__}")
        nam_b_agg, nam_b = self.nam(X)
        linear_predictor = nam_b
        # linear_predictor = X * (nam_b.mean(axis=-1)[:, np.newaxis] @ torch.ones((1, 5)))
        risk_score = torch.exp(linear_predictor)
        H0_proper = self.H0[:, np.newaxis] @ torch.ones((len(X)))[np.newaxis]
        return torch.exp(-(H0_proper * risk_score.mean(axis=-1)).T)
