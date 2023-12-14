import math
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from matplotlib import patches
from sklearn.preprocessing import MinMaxScaler
from survbenim.nam_esimators import BNAMImp1, BaselineNAM, BNAMImp2


def draw_shape_functions(
        ds: np.ndarray, funcs, fnames: List[str],
        categories_dict: Dict[str, Dict[float, str]], shift_mean: bool, derivative=False,
        max_cols=5
) -> Tuple[list, list]:
    if isinstance(funcs, (BNAMImp1, BNAMImp2, BaselineNAM)):
        assert ds.shape[-1] == len(funcs.nam.feature_nns) == len(fnames)
        fxs = [np.sort(np.unique(ds[:, fi])) for fi in range(len(fnames))]
        if derivative:
            fxs_tensor = [torch.tensor(fx, dtype=torch.float32, requires_grad=True) for fx in fxs]
            fys = []
            for fx, feature_nn in zip(fxs_tensor, funcs.nam.feature_nns):
                fy = feature_nn(fx)
                fy.sum().backward()
                fys.append(fx.grad.detach().numpy().flatten())
        else:
            fys = [feature_nn(torch.tensor(fx, dtype=torch.float32)).detach().numpy().flatten()
                   for fx, feature_nn in zip(fxs, funcs.nam.feature_nns)]

        if isinstance(funcs, BNAMImp1):
            fys = [np.abs(fy) for fy in fys]
    else:
        assert ds.shape[-1] == len(funcs) == len(fnames)
        fxs = [np.sort(ds[:, fi]) for fi in range(len(fnames))]
        fys = [func(fx) for func, fx in zip(funcs, fxs)]
    if shift_mean:
        fys = [fy - fy.mean() for fy in fys]

    # BNIM normalization
    if isinstance(funcs, BNAMImp1):
        max_divisor = sum([fy.max() for fy in fys])
        fys = [fy / max_divisor for fy in fys]

    nrows = math.ceil(len(fnames) / max_cols)
    ncols = max_cols if len(fnames) > max_cols else len(fnames)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, 3 * nrows))

    y_shared_min = min([fy.min() for fy in fys])
    y_shared_max = max([fy.max() for fy in fys])
    y_shared_min -= abs(y_shared_min) * 0.1
    y_shared_max += abs(y_shared_max) * 0.1

    for fi, (fx, fy, fname) in enumerate(zip(fxs, fys, fnames)):
        ax = axes[fi] if nrows == 1 else axes[int(fi / max_cols), fi % max_cols]

        if fname in categories_dict.keys():
            categorie_dict = categories_dict[fname]
            ax.set_xticks(list(categorie_dict.keys()))
            ax.set_xticklabels(list(categorie_dict.values()), rotation=45)
            bar_width = abs(list(categorie_dict.keys())[0] - list(categorie_dict.keys())[1])

            # shape function
            ax.step(x=fx, y=fy, where='mid')

            # density bars
            fx_ds = ds[:, fi]
            for x_val in categorie_dict.keys():
                val_samples = fx_ds[fx_ds == x_val]
                assert len(val_samples) > 0
                density = len(val_samples) / len(fx_ds)
                ax.add_patch(
                    patches.Rectangle((x_val - bar_width / 2, y_shared_min), bar_width, y_shared_max - y_shared_min,
                                      alpha=density, color='red'),
                )
            ax.set_xlim((fx.min() - bar_width / 2, fx.max() + bar_width / 2))
        else:
            # shape function
            ax.plot(fx, fy)

            # density bars
            x_ranges = np.linspace(fx.min(), fx.max(), num=10)
            fx_ds = ds[:, fi]
            for x_start, x_end in zip(x_ranges, x_ranges[1:]):
                val_samples = fx_ds[(x_start < fx_ds) & (fx_ds < x_end)]
                # assert len(val_samples) > 0
                density = len(val_samples) / len(fx_ds)
                ax.add_patch(
                    patches.Rectangle((x_start, y_shared_min), x_end - x_start, y_shared_max - y_shared_min,
                                      alpha=density, color='red'),
                )
            ax.set_xlim((fx.min(), fx.max()))

        if y_shared_min != y_shared_max:
            ax.set_ylim((y_shared_min, y_shared_max))
        if isinstance(funcs, BNAMImp1):
            ax.set_ylabel('importance')
        else:
            ax.set_ylabel('contribution')

        ax.set_xlabel(fname)

    plt.tight_layout()
    return fxs, fys


def main():
    ds1 = np.random.normal(loc=0.5, scale=0.25, size=(1000, 3))
    ds1 = MinMaxScaler(feature_range=(-5, 5)).fit_transform(ds1)
    ds2 = np.random.uniform(low=0, high=1, size=(100, 3))
    ds2 = MinMaxScaler(feature_range=(-5, 5)).fit_transform(ds2)
    funcs = [lambda x: x ** 2, lambda x: x, lambda x: x * 1e-20]

    draw_shape_functions(ds1, funcs, fnames=[f'f{i}' for i in range(len(funcs))], shift_mean=True)
    draw_shape_functions(ds2, funcs, fnames=[f'f{i}' for i in range(len(funcs))], shift_mean=True)


if __name__ == '__main__':
    main()
