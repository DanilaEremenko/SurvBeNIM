from pathlib import Path
from sklearn.model_selection import ParameterGrid
from survbenim.nam_esimators import BaselineImportancesMLP
from survbenim.utils_strategy_2 import run_main_st_2

ds_dirs = [
    Path(f"bnam/cox_dataset_clnum=2_fnum=5_cl_size=200"),
    Path(f"bnam/additive_funcs_dataset_clnum=1_fnum=5_cl_size=400"),
    Path(f"bnam/cox_nonlinear=1_fnum=5_cl_size=400"),
    Path(f"bnam/real_ds=gbsg2"),
    Path(f"bnam/real_ds=veterans"),
    Path(f"bnam/real_ds=whas500"),
    Path(f"bnam/real_ds=breast_cancer")
]
for ds_dir in ds_dirs:
    run_main_st_2(
        ds_dir=ds_dir,
        bbox_name='rf',
        nam_claz=BaselineImportancesMLP,
        torch_bnam_grid=ParameterGrid(
            [
                dict(
                    criterion=['mse'],
                    optimizer=['adam'],
                    max_epoch=[10],
                    optimizer_args=[
                        dict(lr=lr, weight_decay=weight_decay)
                        for lr in [1e-4]
                        for weight_decay in [0, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
                        # for weight_decay in [0]
                    ],
                    last_layer=[
                        'relu',
                        # 'exu', 'sigmoid'
                    ],
                    mode=['surv'],
                    v_mode=['no_surv'],
                    nam_args=[
                        dict(kernel_width=1e-1, kernel_name='nn', no_zero_fn='square'),
                        dict(kernel_width=1e-1, kernel_name='nn', no_zero_fn='abs')
                    ]
                )
            ]
        )
    )
