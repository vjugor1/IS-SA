# IS-SA
 Scenario approximation based on importance sampling for DC approximation of AC-OPF

# Guidlines on usage
## Running the code to conduct experiments
* Use `main.py`. Here you can choose a power system to test in row 44, .e.g, `grid_name = "grid30"`.
* Lines `56`, `57` allow one to alter uncertainty parameters: mean and covariance matrix, respectively.
* Line `67` - choose `eta` -- confidedence from the paper
* Lines `79-81` choose parameters for the size of scenario approximations. `N0` and `ks` are used to choose sequence of number of samples to solve. E.g., if one chooses `N0=1` and `ks=list(range(1, 5))`, 4 x 3 instances of scenario approximations will be formed and solved. 3 methods - SA, IS-SA, SA-O with `1 x 1`, `1 x 2`, `1 x 3`, `1 x 4` samples used. Parameters `L` controls how much times should program solve each scenario approximation (with samples are independently samples for different re-solvings). The need for that is explained in the paper and in the end of the notebook `notebooks/sampling_comparison_grid.ipynb`.
* Upon running `main.py`, results will be saved into `saves` folder. There will be folder with grid name, e.g., `grid30`. Inside, one would find `.json` file with logs of all of the solution and folder `figures` with plots of statistics.

## Playgrounds
If you don't want to run full experiment at once with `main.py`, you can considering playing with notebooks in `notebooks` folder. Below one can find summary for the notebooks.
### `sampling_comparison_grid.ipynb`
Notebook version of `main.py`. Here all of the steps are done in separate steps, so one can play with itermediate results.
### `sampling_comparison.ipynb`
The same but with regular polytope.
### `summary_grids.ipynb`
Technical notebook to collect all of the results and prepare coherent plots.
### `1dProblemDemo.ipynb`
One dimensional code illustration of the proposed approach. Here a simple to catch illustration of proposed scneario approximation and interrelations between them is provided. A check on samples severity of IS is provided at the bottom of the notebook.
### `ClassicCampi.ipynb`
A demonstration of how a scenario approximation is formed for one dimensional problem is provided in this noteobook. Step-by-step, the initial deterministic feasibility set is defined, then samples, then scenario approximation using CVXPY, and, finally, the solution obtaining.
### `GridData.ipynb`
Demonstrates how to use griddata parser. This parser scraps equivalent formulation of DC-OPF in term of inequalities only for the corresponding grid. In other words, one obtains formulation


![image](https://user-images.githubusercontent.com/18471262/229287922-208d8788-e6c3-4b95-b6c3-fd0e10741cd5.png)


which is equivalent to the formulation with equality constraints that is used inside of `pandapower`. One can pass `check_pp_vs_new_form=True` to check that the solutions of the reformulation above are idential with `pandapower`'s.
### `ImportanceSampler.ipynb`
Demonstration how to work with importance sampling via `ConditionedPolytopeGaussianSampler` class. Demonstrates that all of the samples generated are outside of the polytope specified.
