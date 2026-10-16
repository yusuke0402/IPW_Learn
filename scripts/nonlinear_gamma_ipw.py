"""Nonlinear gamma sweep for IPW (Hájek-type weighted mean difference) with selectable weights.

真のアウトカムモデルに gamma * x1^2 を加えた DGP(両群共通、真の効果は 1 のまま)で、
過去対照群を現在試験の共変量分布へ輸送する重み付き平均差

    Δ̂ = ȳ_c − Σ_j w_j y_j^h / Σ_j w_j

を評価する。重み w は weight.py の compute_weights で選ぶ:
    propensity     : ロジスティック傾向スコアのオッズ e/(1-e)(--propensity-penalty で罰則を選択)
    ulsif          : uLSIF の密度比(IW-Learn / AIPW-Learn と同一手順)
    uniform        : 一様(輸送なしの平均差)
    oracle         : 真のガウス分布からの密度比

IW-Learn の scripts/nonlinear_gamma_experiment.py と同一のシード・乱数消費順序で
データを生成するため、trial 番号で突き合わせれば IWL と同一データのペア比較になる。
計算はこのリポジトリ内で完結する(IW-Learn への依存なし)。

出力(IW-Learn の summary.csv と同じ列 + 重み診断)。出力先は
results/v2/<purpose>/ipw_<重み>_t<N>_B<B>/(results/README.md 参照):
    trials_<scenario>_N<n_h>_gamma<g>.csv
    summary.csv
既存 trials CSV にある trial は飛ばし、足りない trial だけ計算して追記する(試行単位のレジューム)。

Usage:
    .venv/bin/python scripts/nonlinear_gamma_ipw.py --weight-method propensity --purpose paper   # 既定重み
    .venv/bin/python scripts/nonlinear_gamma_ipw.py --weight-method ulsif --purpose paper
    .venv/bin/python scripts/nonlinear_gamma_ipw.py --scenarios data_scenario_2 --trials 200 --bootstraps 200  # dev
    .venv/bin/python scripts/nonlinear_gamma_ipw.py --smoke
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# numpy 2.0 + Accelerate の matmul 誤検知警告(値は正常)を joblib ワーカーでも抑止する。
# warnings.filterwarnings はワーカーに継承されないため、環境変数で指定する。
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")

import argparse
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from weight import WEIGHT_METHODS, ipw_estimate


# numpy 2.0 + Accelerate の matmul 誤検知警告を抑える(値は正常)
warnings.filterwarnings("ignore", message=".*encountered in matmul")

TRUE_VALUE = 1.0
DEFAULT_TARGET_N = 50
METHOD_LABEL = "IPW"
PREFIX = "ipw"

# ---- IW-Learn scripts/nonlinear_gamma_experiment.py と同一のデータ生成 ----
MEANS = {
    "data_scenario_1": {"target": [0.0, 1.0], "source": [0.0, 0.0]},
    "data_scenario_2": {"target": [2.0, 3.0], "source": [0.0, 0.0]},
    "data_scenario_3": {"target": [0.0, 1.0], "source": [0.0, 0.0]},
}
COVS = {
    "data_scenario_1": {
        "target": [[1.5, -0.7], [-0.7, 1.5]],
        "source": [[1.7, 0.7], [0.7, 1.7]],
    },
    "data_scenario_2": {
        "target": [[1.5, -0.7], [-0.7, 1.5]],
        "source": [[1.7, 0.7], [0.7, 1.7]],
    },
    "data_scenario_3": {
        "target": [[1.5, -0.7], [-0.7, 1.5]],
        "source": [[3.5, 0.7], [0.7, 3.5]],
    },
}
COEFFICIENT = np.array([1.5, -2.0, 0.8])  # [intercept, x1, x2]


def generate_data(scenario, source_n, gamma, target_n):
    """IW-Learn main.py / nonlinear_gamma_experiment.py と同一の乱数消費順序。"""
    mean = MEANS[scenario]
    cov = COVS[scenario]

    x_t = np.random.multivariate_normal(mean["target"], cov["target"], target_n)
    target_x = np.insert(x_t, 0, 1, axis=1)
    eps_t = np.random.normal(0, 1, target_n)
    y_t = (
        COEFFICIENT @ target_x.T
        + gamma * target_x[:, 1] ** 2
        + TRUE_VALUE
        + eps_t
    )
    target_y = y_t.reshape(-1, 1)

    x_s = np.random.multivariate_normal(mean["source"], cov["source"], source_n)
    source_x = np.insert(x_s, 0, 1, axis=1)
    eps_s = np.random.normal(0, 1, source_n)
    y_s = COEFFICIENT @ source_x.T + gamma * source_x[:, 1] ** 2 + eps_s
    source_y = y_s.reshape(-1, 1)

    return target_x, target_y, source_x, source_y
# ---------------------------------------------------------------------------


def _true_dist(scenario):
    means = {k: np.array(v) for k, v in MEANS[scenario].items()}
    covs = {k: np.array(v) for k, v in COVS[scenario].items()}
    return means, covs


def _estimate(tx, ty, sx, sy, method, means, covs, wkw):
    result = ipw_estimate(tx, ty, sx, sy, method=method, means=means, covariances=covs, **wkw)
    return result["estimate"], result["source_weights"]





# ---------------------------------------------------------------------------
# 出力規約(4 リポジトリ共通):
#   results/v2/<purpose>/<run_name>/   purpose ∈ {paper, dev, smoke}
#   run_name 既定 = <手法>_<重み>_t<target_n>_B<bootstraps>(試行数は含めない)
# 試行単位のレジューム: 既存 trials CSV にある trial は飛ばし、足りない trial だけ計算して追記する。
# シードは trial 番号で固定なので、分割して計算しても一括計算と同じ結果になる。
# ---------------------------------------------------------------------------
def _git_commit(root):
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def resolve_out_dir(root, args, run_name_default):
    """--out-dir(results/ 相対の明示指定)があればそれ、なければ results/v2/<purpose>/<run_name>/。"""
    if args.out_dir:
        return root / "results" / args.out_dir
    purpose = "smoke" if args.smoke else args.purpose
    return root / "results" / "v2" / purpose / (args.run_name or run_name_default)


# The shared helpers reject incompatible cached runs before writing anything.
from experiment_io import write_run_meta, load_or_run_trials  # noqa: E402
from experiment_execution import (  # noqa: E402
    run_weighted_bootstrap, diagnostic_columns, variant_columns, summarize_trials,
)
from experiment_diagnostics import (  # noqa: E402
    DEFAULT_MIN_ESS, DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE, validate_min_ess,
)


def add_layout_args(parser):
    parser.add_argument("--purpose", choices=["paper", "dev", "smoke"], default="dev",
                        help="出力先 results/v2/<purpose>/。paper=論文・資料用の本番、"
                             "dev=動作確認・縮小規模、smoke=--smoke 用(git 管理外)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="results/v2/<purpose>/ 配下のディレクトリ名。"
                             "既定は <手法>_<重み>_t<target_n>_B<bootstraps>")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="results/ 相対の出力先を直接指定(規約を使わない場合のみ)")


def run_one_trial(i, scenario, source_n, gamma, n_bootstraps, target_n, method, wkw,
                  max_bootstrap_failure_rate=DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE):
    random.seed(i)
    np.random.seed(i)
    tx, ty, sx, sy = generate_data(scenario, source_n, gamma, target_n)
    means, covs = _true_dist(scenario)
    run = run_weighted_bootstrap(
        lambda a, b, c, d: ipw_estimate(a, b, c, d, method=method,
                                      means=means, covariances=covs, **wkw),
        tx, ty, sx, sy, n_bootstraps, i, max_failure_rate=max_bootstrap_failure_rate,
        min_ess=wkw.get("min_ess", DEFAULT_MIN_ESS))
    return {"trial": i, **diagnostic_columns(run, "ipw_"),
            **variant_columns(run, PREFIX),
            "ipw_max_weight": run["point"].get("max_weight", np.nan)}


def summarize(df, prefix):
    stats = summarize_trials(df, prefix, TRUE_VALUE)
    stats["max_weight_mean"] = float(df[f"{prefix}_max_weight"].mean())
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--min-ess", type=float, default=None)
    parser.add_argument("--max-bootstrap-failure-rate", type=float, default=None)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET_N)
    parser.add_argument("--source-ns", type=int, nargs="+",
                        default=[20, 50, 100, 200, 500])
    parser.add_argument("--scenarios", type=str, nargs="+",
                        default=["data_scenario_1", "data_scenario_2", "data_scenario_3"])
    parser.add_argument("--weight-method", choices=WEIGHT_METHODS, default="propensity")
    parser.add_argument("--propensity-penalty", choices=["l2", "none"], default="l2")
    parser.add_argument("--propensity-C", type=float, default=1.0)
    add_layout_args(parser)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    hp = config.get("hyperparameters", {})
    args.min_ess = validate_min_ess(args.min_ess if args.min_ess is not None else hp.get("min_ess", DEFAULT_MIN_ESS))
    args.max_bootstrap_failure_rate = (args.max_bootstrap_failure_rate if args.max_bootstrap_failure_rate is not None
                                       else hp.get("max_bootstrap_failure_rate", DEFAULT_MAX_BOOTSTRAP_FAILURE_RATE))
    if not np.isfinite(args.max_bootstrap_failure_rate) or not 0 <= args.max_bootstrap_failure_rate <= 1:
        parser.error("max-bootstrap-failure-rate must be between 0 and 1")
    if args.trials < 1 or args.bootstraps < 0 or args.target_n < 5 or min(args.source_ns) < 5:
        parser.error("Require trials >= 1, bootstraps >= 0 and sample sizes >= 5")

    scenarios, source_ns, gammas = args.scenarios, args.source_ns, args.gammas
    n_trials, n_boot = args.trials, args.bootstraps
    if args.smoke:
        scenarios, source_ns, gammas = ["data_scenario_2"], [100], [0.5]
        n_trials, n_boot = 4, 20
    wkw = {"propensity_penalty": args.propensity_penalty, "propensity_C": args.propensity_C,
           "min_ess": args.min_ess}

    weight_tag = args.weight_method
    if args.weight_method.startswith("propensity"):
        weight_tag += f"-{args.propensity_penalty}"
    out_dir = resolve_out_dir(PROJECT_ROOT, args, f"ipw_{weight_tag}_t{args.target_n}_B{n_boot}_ess{args.min_ess:g}_bf{args.max_bootstrap_failure_rate:g}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    write_run_meta(out_dir, PROJECT_ROOT, method="IPW", weight=args.weight_method,
                   propensity=wkw, target_n=args.target_n, n_trials=n_trials,
                   bootstraps=n_boot, min_ess=args.min_ess,
                   max_bootstrap_failure_rate=args.max_bootstrap_failure_rate, scenarios=scenarios, source_ns=source_ns, gammas=gammas)
    print(f"output: {out_dir}", flush=True)

    # scenario 2(シフト最大)を先に回す(IW-Learn と同じ順序)
    conditions = [
        (sc, sn, g)
        for sc in ["data_scenario_2", "data_scenario_1", "data_scenario_3"]
        if sc in scenarios
        for g in gammas
        for sn in source_ns
    ]

    summary_rows = []
    for cond_i, (sc, sn, g) in enumerate(conditions, start=1):
        t0 = time.time()
        csv_path = out_dir / f"trials_{sc}_N{sn}_gamma{g:g}.csv"
        print(f"[{cond_i}/{len(conditions)}] start {sc} N{sn} gamma={g:g} "
              f"({n_trials} trials x B={n_boot}, weights={args.weight_method})", flush=True)
        df, n_new = load_or_run_trials(
            csv_path, n_trials, f"{PREFIX}_boot_se",
            lambda i: run_one_trial(i, sc, sn, g, n_boot, args.target_n,
                                    args.weight_method, wkw, args.max_bootstrap_failure_rate),
            args.n_jobs, args.verbose,
        )
        status = "skip (cached)" if n_new == 0 else f"done (+{n_new} trials)"
        summary_rows.append({"scenario": sc, "source_n": sn, "gamma": g,
                             "method": METHOD_LABEL, "weight_method": args.weight_method,
                             "propensity_penalty": args.propensity_penalty,
                             "n_trial": len(df), "B": n_boot,
                             **summarize(df, PREFIX)})
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        s = summary_rows[-1]
        print(f"[{cond_i}/{len(conditions)}] {status} {sc} N{sn} gamma={g:g} "
              f"bias={s['bias']:+.3f} cov={s['coverage']:.3f} ess={s['ess_median']:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(summary_path)


if __name__ == "__main__":
    main()
