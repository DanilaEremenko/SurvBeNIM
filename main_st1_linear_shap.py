from pathlib import Path
from survbenim.utils_strategy_1_linear import run_main_st_1_linears

if __name__ == '__main__':
    cl_num = 2
    for f_num in [5, 10, 20]:
        for cl_size in [200]:
            test_cl_size = int(cl_size * 0.3)
            run_main_st_1_linears(
                exp_dir=Path(f"bnam/cox_dataset_clnum={cl_num}_fnum={f_num}_cl_size={cl_size}"),
                test_ids=[cl_i * test_cl_size + pt_i for cl_i in range(cl_num) for pt_i in range(0, 10)],
                bbox_name='rf',
                solver_name='shap',
                radius=0.2
            )
