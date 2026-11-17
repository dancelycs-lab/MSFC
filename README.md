# Motion-separated force control

Reproduction code for **Adaptive Force Control of Moving Compliant Workpieces Through Motion-Separated Contact Estimation**.

The experiments use a torque-actuated, two-joint robot in MuJoCo. The robot follows a tangential trajectory while regulating force against a moving compliant workpiece. The evaluation contains 32 tasks and 16 controller configurations, including MSFC, seven published comparisons, four additional baselines, and four ablations.

## Installation

Use Python 3.11. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run the experiments

A short check runs all 16 controllers on one complete 8.4-second task and compares seven metrics with the recorded results:

```bash
python reproduce.py smoke --workers 4
```

Run the complete evaluation, generate figures and tables, and perform the analytical, integration-step, constant-force, and execution-time checks:

```bash
python reproduce.py all --workers 4
```

Results are written to `output/`; the short check uses `smoke_output/`. Increase or reduce `--workers` to suit the available CPU and memory. Each simulation uses one BLAS thread. The complete run requires substantially more time than the short check, particularly for the learned residual controller. Interrupted simulations resume when the same command is run again.

The stages can also be run separately:

```bash
python reproduce.py run --workers 4
python reproduce.py analyze
python reproduce.py theory
python reproduce.py validate --workers 4
python reproduce.py timing --workers 4
```

Use `--output path/to/results` to choose a different output directory. Use a new output directory after changing controller parameters or task configurations.

## Results

- `results.json`: all 512 evaluation runs and their scalar metrics.
- `tables/`: comparison, ablation, reference-waveform, paired-bootstrap, and individual-run results in CSV format.
- `figures/`: seven PDF plots of force responses, integration-rate contact force, configuration-wise errors, ramp tracking, ablations, tangential error, and joint torque.
- `traces/`: trajectories used by the plots, saved as compressed NumPy archives.
- `reference_comparison.csv`: differences from the recorded evaluation metrics.
- `theory.json`: symbolic identities, exact Lyapunov-vertex checks, and the physical/reduced-dynamics comparison.
- `constant_force_summary.csv` and `refinement_summary.csv`: constant-force regulation and integration-step comparisons.
- `controller_timing.json`: measured controller execution times and command-replay agreement.

The trajectory archives contain `trace`, `reference`, and `columns`. Read `columns` for the signal order. Positions are in metres, velocities in metres per second, forces in newtons, torques in newton-metres, and time in seconds.

Force-error metrics use the physical contact force after 1.2 s. Normalized mean absolute error is divided by the mean desired force over the same interval. Peak force ratio includes the complete trajectory. Contact-loss duration counts physical force below 0.5 N. Tangential RMSE and torque RMS use the post-1.2-second interval. The force-response plots show the filtered feedback force; the short-window contact-force plots show the unfiltered physical force. The paired confidence intervals use 10,000 resamples of the 16 matched physical configurations in each contact setting.

## Controllers

The method identifiers in `config/controllers.json` select the implementations below. Parameters are fixed during evaluation.

| Identifier | Method | Implementation |
|---|---|---|
| `reference_inverse` | MSFC | Motion-separated estimation, equilibrium-motion correction, and reference inverse |
| `force_cbf` | SC3 | Scalar force-observer and velocity-projection adaptation of [Wang et al.](https://doi.org/10.1109/LRA.2025.3636028) |
| `oidob` | OIDOBt | Outer-loop admittance/DOB interconnection; [Samuel et al.](https://doi.org/10.1109/TCST.2023.3340086) and the [unified observer formulation](https://doi.org/10.1109/LRA.2024.3406055) |
| `mob` | MOB | Multi-function observer of [Samuel et al.](https://doi.org/10.1109/TIE.2023.3317843), expressed through the unified observer formulation |
| `cdyob` | CDYOB | Coupled-dynamics observer from the unified observer formulation |
| `aftac_outer` | AFTAC-OL | Adaptive force outer loop of [Liu et al.](https://doi.org/10.1109/TIE.2024.3383029) |
| `ph_force_tcst` | PH-FI | Force-impedance torque law of [Ma et al.](https://doi.org/10.1109/TCST.2025.3606519) |
| `oracle_tcst` | ORACLE | Learned incremental model and bounded residual position search of [Petrone et al.](https://doi.org/10.1109/TCST.2025.3635172) |
| `pi` | PI force control | Force-error PI outer loop |
| `fixed_observer` | Fixed contact model | Fixed stiffness and damping in the shared inverse |
| `two_observer_estimator` | Two-observer estimate | Two-candidate interpolation adapted from [Coutinho and Cortesão](https://doi.org/10.1016/j.conengprac.2013.11.002) |
| `oracle_dfc` | DFC | Direct force controller underlying ORACLE |
| `retained` | No reference inverse | MSFC without reference compensation |
| `calibrate_only` | Initial fit only | Estimation stops after 1.2 s |
| `no_time` | No time projection | Polynomial time terms removed from the regression |
| `no_motion` | No motion correction | Equilibrium-motion term removed from the command |

The published controllers are adapted to the common test plant and measurement interface. The velocity-output methods share a Cartesian PI servo. PH-FI, ORACLE, and DFC retain torque interfaces with delayed outer force corrections or position requests and active local impedance feedback. Their implementation details and source equations are documented in `src/`.

## ORACLE model

`models/oracle_ensemble.npz` contains the trained three-network ensemble used in the comparisons. Evaluation loads these weights directly.

To regenerate the 12 development trajectories and retrain the ensemble:

```bash
python reproduce.py train --workers 4 --output training_output
```

Training uses ten complete trajectories; trajectory indices 2 and 9 are held out for model validation. The command writes `training_output/oracle_ensemble.npz` and prediction errors in `model_validation.json`. To evaluate the retrained model, set `oracle_tcst.controller_parameters.model_path` in `config/controllers.json` to `training_output/oracle_ensemble.npz` and use a new evaluation output directory.

## Files and numerical settings

`config/` contains the task definitions, selected controller gains, verification cases, certificate matrices, and reference metrics. `src/` contains the plant, controllers, analysis, and plotting routines. The MuJoCo model is generated by `src/plant.py`.

The main evaluation uses a 1-ms control period and a 50-microsecond integration interval. Refinement runs use 25 microseconds. Measurement-noise realizations are fixed by the task configurations. The analytical certificate checks concern the continuous persistent-contact model stated in the paper. Execution times depend on the processor and system load; floating-point and contact-solver differences can affect results across software versions and platforms.
