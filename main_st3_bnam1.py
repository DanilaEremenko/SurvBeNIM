from pathlib import Path
from sklearn.model_selection import ParameterGrid
from survbenim.nam_esimators import BNAMImp1
from survbenim.utils_strategy_3 import run_main_st_3

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
    run_main_st_3(
        ds_dir=ds_dir,
        nam_claz=BNAMImp1,
        torch_bnam_grid=ParameterGrid(
            [
                dict(
                    criterion=['mse'],
                    optimizer=['adam'],
                    max_epoch=[20],
                    optimizer_args=[
                        dict(lr=lr, weight_decay=weight_decay)
                        for lr in [1e-4]
                        for weight_decay in [0]
                        # for weight_decay in [0]
                    ],
                    last_layer=[
                        'relu',
                        # 'exu', 'sigmoid'
                    ],
                    mode=['surv'],
                    nam_args=[
                        dict(kernel_width=1e-1, kernel_name='nn', no_zero_fn='square'),
                        dict(kernel_width=1e-1, kernel_name='nn', no_zero_fn='abs')
                    ]
                )
            ]
        )
    )
