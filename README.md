# Unbalanced Optimal Transport through Non-negative Penalized Linear Regression
Contains the code relative to the paper Unbalanced Optimal Transport through Non-negative Penalized Linear Regression https://arxiv.org/abs/2106.04145

```
@inproceedings{chapel2021unbalanced,
  title={Unbalanced Optimal Transport through Non-negative Penalized Linear Regression},
  author={Chapel, Laetitia and Flamary, R{\'e}mi and Wu, Haoran and F{\'e}votte, C{\'e}dric and Gasso, Gilles},
  booktitle={Advances in Neural Information Processing Systems 34},
  year={2021}
}
```
![L2 UOT](https://github.com/lchapel/UOT-though-penalized-linear-regression/blob/main/regpath_l2.jpg "L2 UOT")

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running

The regularization path algorithm is implemented in the [Python Optimal Transport (POT)](https://pythonot.github.io/) toolbox  and is the latest version up-to-date.

The functions for running the algorithms can be found in the following files:
- ``solvers\solver_kl_UOT.py`` contains the functions that allows solving the KL-penalized OUT, that is
	- ``ot_ukl_solve_BFGS`` to run the BFGS algorithm 
	- ``ot_uklreg_solve_mm`` to run our multiplicative algorithm 
- ``solvers\solvers_L2_UOT.py`` contains the functions that allows solving the L2-penalized OUT, that is
	- ``ot_ul2_solve_BFGS`` to run the BFGS algorithm 
	- ``ot_ul2_reg_path`` to compute our regularization path
	- ``ot_ul2_solve_lasso_celer`` to solve the UOT reformulated as a Lasso problem using the Celer implementation
	- ``ot_ul2_solve_lasso_cd`` to solve the UOT reformulated as a Lasso problem using the Scikit-learn implementation
	- ``ot_ul2_solve_mu`` to run our multiplicative algorithm 
- ``solvers\solver_semirelax_L2_UOT.py`` contains the functions that allows solving the semi-relaxed L2-penalized OUT, that is
	- ``ot_semi_relaxed_ul2_reg_path`` to solve our regularization path algorithm

## Results

Our results given in Figures 1 to 5 can be reproduced by running the following notebooks:

- Figure 1 can be reproduced by running the notebook [Figure 1.ipynb](notebooks/Figure1.ipynb)

![Figure 1](evol_pi.jpg "Figure 1") 

- Figure 2 can be reproduced by running the notebook [Figure 2.ipynb](notebooks/Figure2.ipynb)

![Figure 2](regpath_l2.jpg "Figure 2") 

- Figure 3 can be reproduced by running the notebook [Figure 3.ipynb](notebooks/Figure3.ipynb)

![Figure 2](simu.jpg "Figure 3") 

- Figure 4 can be reproduced by running the notebook [Figure 4.ipynb](notebooks/Figure4.ipynb)

![Figure 4](Classif_expe.jpg "Figure 4") 

- Figure 5 can be reproduced by running the notebook [Figure 5.ipynb](notebooks/Figure5.ipynb)

![Figure 5](simu_cpu_gpu.jpg "Figure 5") 
