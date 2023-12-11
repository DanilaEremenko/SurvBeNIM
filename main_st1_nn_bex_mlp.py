from pathlib import Path
from survbenim.nam_esimators import BaselineImportancesMLP
from survbenim.utils_strategy_1_nams import run_main_st_1_nams
from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':
    cl_num = 5
    f_num = 10
    test_cl_size = 60
    run_main_st_1_nams(
        exp_dir=Path(f"bnam/cox_dataset_clnum={cl_num}_fnum={f_num}_cl_size=200"),
        test_ids=[cl_i * test_cl_size + pt_i for cl_i in range(cl_num) for pt_i in range(0, 10)],
        bbox_name='rf',
        bnam_claz=BaselineImportancesMLP,
        bnam_args=dict(kernel_name='nn', kernel_width=1e-1, last_layer='relu', no_zero_fn='abs'),
        radius=0.1,
        param_grid=ParameterGrid(
            [
                dict(
                    criterion=['mse'],
                    optimizer=['adam'],
                    optimizer_args=[
                        dict(lr=lr, weight_decay=weight_decay)
                        for lr in [1e-2]
                        for weight_decay in [0]
                    ],
                    mode=['surv'],
                    v_mode=['no_surv']
                )
            ]
        )
    )
