# IPW comparator at target_N=50 (for the IW-Learn paper)

- `run_ipw_t50.py`: propensity-score odds-weighted Hajek IPW targeting the current-trial population.
  Linear (main) / quadratic (sensitivity) logistic PS, full-pipeline bootstrap (B=1000).
- Data generation must match IW-Learn `scripts/nonlinear_gamma_experiment.py::generate_data`
  (same seeding → same trial i data). The script imports it via `sys.path` from the IW-Learn checkout;
  adjust `PROJECT_ROOT` or copy `generate_data` when running from this repo.
- `results/`: 45 conditions (3 scenarios × n_h∈{20,50,100,200,500} × γ∈{0,0.25,0.5}), 1000 trials each,
  `trials_*.csv` + `summary.csv` + run logs. `summary.csv` is mirrored in IW-Learn as
  `results/additional/nonlinear_gamma_target50/external/ipw_t50_summary.csv`.
