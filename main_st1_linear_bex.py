from pathlib import Path
from survbenim.utils_strategy_1_linear import run_main_st_1_linears

if __name__ == '__main__':
    cl_num = 2
    f_num = 20
    test_cl_size = 60
    run_main_st_1_linears(
        exp_dir=Path(f"bnam/cox_dataset_clnum={cl_num}_fnum={f_num}_cl_size=200"),
        test_ids=[cl_i * test_cl_size + pt_i for cl_i in range(cl_num) for pt_i in range(0, 10)],
        bbox_name='rf',
        solver_name='survbex',
        radius=0.2
    )
